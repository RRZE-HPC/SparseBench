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

void swap_DMatrix(DMatrix *x_perm, DMatrix *y_perm)
{
  DMatrix tmp = *x_perm;
  *x_perm     = *y_perm;
  *y_perm     = tmp;
}

int test_spmmvSCS(void *args, const char *dataDir)
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

  const int test_blockwidth = 3;

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
      char C_str[STR_LEN]     = "";
      char sigma_str[STR_LEN] = "";
#ifdef SCS
      A.C     = arguments->C;
      A.sigma = arguments->sigma;
#endif
      sprintf(C_str, "%d", arguments->C);
      sprintf(sigma_str, "%d", arguments->sigma);
      int repeat_count = arguments->run_count;

      // String preprocessing
      FORMAT_AND_STRIP_VECTOR_FILE(entry)

      // This is the external file to check against
      char *pathToExpectedData = malloc(STR_LEN);

      char in_file_name[64];
      snprintf(in_file_name, sizeof(in_file_name), "_spmmv_x_%d.in", repeat_count);

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

        // Use vectorSize (= nrPadded for SCS) so dimensions match
        // permute_DMatrix requires src.nr == dst.nr
        DMatrix x = { .nr = A.nc, .nc = test_blockwidth, .entries = NULL };
        DMatrix y = { .nr = A.nr, .nc = test_blockwidth, .entries = NULL };

        x.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, x.nr * x.nc * sizeof(V_ELE));
        y.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, y.nr * y.nc * sizeof(V_ELE));

        // Initialize x: sequential values for real entries, 0 for padding
        for (int i = 0; i < (x.nr * x.nc); ++i) {
          x.entries[i] = (CG_FLOAT)i;
        }
        for (int i = 0; i < (y.nr * y.nc); ++i) {
          y.entries[i] = (CG_FLOAT)0.0;
        }

// NOTE : since we switch the vectors around making them bigger is necessary to
// prevent accessing garbage data
#ifdef SCS
        DMatrix x_perm = { .nr = vectorSize, .nc = test_blockwidth, .entries = NULL };
        DMatrix y_perm = { .nr = vectorSize, .nc = test_blockwidth, .entries = NULL };

        x_perm.entries =
            (V_ELE *)allocate(ARRAY_ALIGNMENT, x_perm.nr * x_perm.nc * sizeof(V_ELE));
        y_perm.entries =
            (V_ELE *)allocate(ARRAY_ALIGNMENT, y_perm.nr * y_perm.nc * sizeof(V_ELE));

        // Zero-fill permuted vectors (essential for padded rows)
        for (int i = 0; i < (x_perm.nr * x_perm.nc); ++i)
          x_perm.entries[i] = (CG_FLOAT)0.0;
        for (int i = 0; i < (y_perm.nr * y_perm.nc); ++i)
          y_perm.entries[i] = (CG_FLOAT)0.0;

        // Forward permutation: scatter x into SCS ordering
        permute_DMatrix(A.oldToNewPerm, &x, &x_perm);
        for (size_t i = 0; i < repeat_count; i++) {
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
          gpu_spMMVM(&A, &x_perm, &y_perm);
#else
          spMMVM(&A, &x_perm, &y_perm);
#endif
          if (i < repeat_count - 1) {
            swap_DMatrix(&x_perm, &y_perm);
          }
        }
        // Inverse permutation: gather y from SCS ordering back to original
        permute_DMatrix(A.newToOldPerm, &y_perm, &y);
#else
        for (size_t i = 0; i < repeat_count; i++) {
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
          gpu_spMMVM(&A, &x, &y);
#else
          spMMVM(&A, &x, &y);
#endif
          if (i < repeat_count - 1) {
            swap_DMatrix(&x, &y);
          }
        }
#endif
        // Dump to this external file
        char *pathToReportedData = malloc(STR_LEN);
        char out_file_name[64];
        snprintf(out_file_name, sizeof(out_file_name), "_spmmv_x_%d.out", repeat_count);
        BUILD_MATRIX_FILE_PATH(
            entry, "reported/", out_file_name, C_str, sigma_str, pathToReportedData);
        FILE *reportedData = fopen(pathToReportedData, "w");
        if (reportedData == NULL) {
          perror("Error opening reported data file");
          printf("pathToReportedData = %s\n", pathToReportedData);
          exit(EXIT_FAILURE);
        }

        printf("pathToReportedData = %s\n", pathToReportedData);

        dumpDMatrix_impl(&y, reportedData);
        fclose(reportedData);

        // If the expect and reported data differ in some way
        int diff_result = diff_files(pathToExpectedData, pathToReportedData);

        // Free per-iteration allocations
        free(matrixFormat);
        free(pathToReportedData);

        deallocate(x.entries);
        deallocate(y.entries);

#ifdef SCS
        deallocate(x_perm.entries);
        deallocate(y_perm.entries);
#endif

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