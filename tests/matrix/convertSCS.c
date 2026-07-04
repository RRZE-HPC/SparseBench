// DL 2025.04.04
// Single rank test to convert MM to SCS format

#include "../../src/matrix.h"
#include "../common.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

int test_convertSCS(void *args, const char *dataDir)
{

  /* TODO: validate under _MPI — single-rank only; no commInit/commDistributeMatrix (full matrix on one process). */
  int rank = 0;
  int size = 1;

  // Ensure the directory for reported outputs exists (its contents are
  // gitignored, so it is absent on a fresh clone and fopen(...,"w") would
  // return NULL, causing a segfault in dumpMatrix_impl).
  char *pathToReportedDir = malloc(strlen(dataDir) + strlen("reported/") + 1);
  strcpy(pathToReportedDir, dataDir);
  strcat(pathToReportedDir, "reported/");
  mkdir(pathToReportedDir, 0775);
  free(pathToReportedDir);

  // Open the directory
  char *pathToMatrices = malloc(strlen(dataDir) + strlen("testMatrices/") + 1);
  strcpy(pathToMatrices, dataDir);
  strcat(pathToMatrices, "testMatrices/");

  DIR *dir = opendir(pathToMatrices);
  if (dir == NULL) {
    perror("Error opening directory");
    return 1;
  }

  // Read the directory entries
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strstr(entry->d_name, ".mtx") != NULL) {
      char *pathToMatrix = malloc(strlen(pathToMatrices) + strlen(entry->d_name) + 1);
      strcpy(pathToMatrix, pathToMatrices);
      strcat(pathToMatrix, entry->d_name);

      Matrix A; // this is the crs/sell matrix
      Args *arguments = (Args *)args;
      A.C             = arguments->C;
      A.sigma         = arguments->sigma;

      // String preprocessing
      char C_str[STR_LEN];
      char sigma_str[STR_LEN];
      FORMAT_AND_STRIP_MATRIX_FILE(A, entry, C_str, sigma_str)

      // This is the external file to check against
      char *pathToExpectedData = malloc(STR_LEN);
      BUILD_MATRIX_FILE_PATH(
          entry, "expected/", ".in", C_str, sigma_str, pathToExpectedData);

      // Validate against expected data, if it exists
      FILE *fptr = fopen(pathToExpectedData, "r");
      if (fptr) {

        MMMatrix m;
        MMMatrixRead(&m, pathToMatrix);

        GMatrix gm;
        matrixConvertfromMM(&m, &gm);

        // Set single rank defaults for MmMatrix
        // m.startRow = 0;
        // m.stopRow = m.nr;
        // m.totalNr = m.nr;
        // m.totalNnz = m.nnz;

        convertMatrix(&A, &gm);

        // Dump to this external file
        char *pathToReportedData = malloc(STR_LEN);
        BUILD_MATRIX_FILE_PATH(
            entry, "reported/", ".out", C_str, sigma_str, pathToReportedData);
        FILE *reportedData = fopen(pathToReportedData, "w");
        if (reportedData == NULL) {
          perror("Error opening reported data file");
          free(pathToReportedData);
          free(pathToExpectedData);
          free(pathToMatrix);
          closedir(dir);
          return 1;
        }

        dumpMatrix_impl(&A, reportedData);
        fclose(reportedData);

        // If the expect and reported data differ in some way
        if (diff_files(pathToExpectedData, pathToReportedData)) {
          free(pathToReportedData);
          free(pathToExpectedData);
          free(pathToMatrix);

          closedir(dir);
          return 1;
        }
      }
      if (fptr)
        fclose(fptr);
      free(pathToExpectedData);
      free(pathToMatrix);
    }
  }

  free(pathToMatrices);
  closedir(dir);

  return 0;
}