// DL 2025.04.07
// Single rank SpMV test

#include "../../src/allocate.h"
#include "../../src/debugger.h"
#include "../../src/matrix.h"
#include "../../src/solver.h"
#include "../common.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include "../../src/affinity.h"
#include <omp.h>
#endif

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "../../src/cuda_kernels.h"
#endif

int test_spmvSCS(void *args, const char *dataDir)
{

  int rank           = 0;
  int size           = 1;
  int validFileCount = 0;

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

      printf("pathToMatrix = %s\n", pathToMatrix);

      Matrix A;
      Args *arguments         = (Args *)args;
      int repeat_count        = arguments->run_count;
      char C_str[STR_LEN]     = "";
      char sigma_str[STR_LEN] = "";
#ifdef SCS
      A.C     = arguments->C;
      A.sigma = arguments->sigma;
#endif
      sprintf(C_str, "%d", arguments->C);
      sprintf(sigma_str, "%d", arguments->sigma);

      // String preprocessing
      FORMAT_AND_STRIP_VECTOR_FILE(entry)

      // This is the external file to check against
      char *pathToExpectedData = malloc(STR_LEN);

      char in_file_name[64];
      snprintf(in_file_name, sizeof(in_file_name), "_spmv_x_%d.in", repeat_count);

      BUILD_VECTOR_FILE_PATH(entry, "expected/", in_file_name, pathToExpectedData);

      printf("pathToExpectedData = %s\n", pathToExpectedData);

      // Validate against expected data, if it exists
      FILE *fptr = fopen(pathToExpectedData, "r");
      if (fptr) {
        ++validFileCount;

        MMMatrix m;
        MMMatrixRead(&m, pathToMatrix);

        GMatrix gm;
        matrixConvertfromMM(&m, &gm);

        // Set single rank defaults for MmMatrix
        m.startRow = 0;
        m.stopRow  = m.nr;
        m.totalNr  = m.nr;
        m.totalNnz = m.nnz;

        int vectorSize;
        char *matrixFormat = (char *)malloc(4 * sizeof(char));

#ifdef SCS
        convertMatrix(&A, &gm);
        vectorSize = A.nrPadded;
        strcpy(matrixFormat, "SCS");
#else
        convertMatrix(&A, &gm);
        vectorSize = A.nr;
        strcpy(matrixFormat, "CRS");
#endif
        VALIDATE_MATRIX_FORMAT(matrixFormat);
        // A.matrixFormat = matrixFormat;

        V_ELE *x = (V_ELE *)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(V_ELE));
        V_ELE *y = (V_ELE *)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(V_ELE));

        // Fix x = 1 for now
        for (int i = 0; i < vectorSize; ++i) {
          x[i] = (CG_FLOAT)1.0;
          y[i] = (CG_FLOAT)0.0;
        }

#ifdef SCS

        V_ELE *x_perm = (V_ELE *)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(V_ELE));
        V_ELE *y_perm = (V_ELE *)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(V_ELE));

        // Permute x into SCS ordering once (colInd already remapped)
        permute_vector(A.oldToNewPerm, x, x_perm, A.nr);

        for (size_t i = 0; i < repeat_count; i++) {
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
          gpu_spMVM(&A, x_perm, y_perm);
#else
          spMVM(&A, x_perm, y_perm);
#endif
          if (i < repeat_count - 1) {
            swap_ptrs(&x_perm, &y_perm);
          }
        }

        // Unpermute y back to original ordering
        permute_vector(A.newToOldPerm, y_perm, y, A.nr);
#else

        for (size_t i = 0; i < repeat_count; i++) {
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
          gpu_spMVM(&A, x, y);
#else
          spMVM(&A, x, y);
#endif
          if (i < repeat_count - 1) {
            swap_ptrs(&x, &y);
          }
        }
#endif

        // Dump to this external file
        char *pathToReportedData = malloc(STR_LEN);
        char out_file_name[64];
        snprintf(out_file_name, sizeof(out_file_name), "_spmv_x_%d.out", repeat_count);
        BUILD_MATRIX_FILE_PATH(
            entry, "reported/", out_file_name, C_str, sigma_str, pathToReportedData);
        FILE *reportedData = fopen(pathToReportedData, "w");
        if (reportedData == NULL) {
          perror("Error opening reported data file");
          printf("pathToReportedData = %s\n", pathToReportedData);
          exit(EXIT_FAILURE);
        }

        printf("pathToReportedData = %s\n", pathToReportedData);

        dumpVectorToFile(y, A.nr, reportedData);
        fclose(reportedData);

        // If the expect and reported data differ in some way
        int diff_result = diff_files(pathToExpectedData, pathToReportedData);

        // Free per-iteration allocations
        free(matrixFormat);

        deallocate(x);
        deallocate(y);
#ifdef SCS
        deallocate(x_perm);
        deallocate(y_perm);
#endif
        free(pathToReportedData);

        freeMatrix(&A);
        freeGMatrix(&gm);
        freeMMMatrix(&m);

        if (diff_result) {
          fclose(fptr);
          free(pathToExpectedData);
          free(pathToMatrix);
          free(pathToMatrices);
          closedir(dir);
          return 1;
        }

        fclose(fptr);
      }
      free(pathToExpectedData);
      free(pathToMatrix);
    }
  }

  closedir(dir);

  if (!validFileCount) {
    fprintf(stderr, "No valid files found in %s\n", pathToMatrices);
    free(pathToMatrices);
    return 1;
  } else {
    free(pathToMatrices);
    return 0;
  }
}