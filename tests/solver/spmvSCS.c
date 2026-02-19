// DL 2025.04.07
// Single rank SpMV test

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "../../src/matrix.h"
#include "../../src/solver.h"
#include "../../src/debugger.h"
#include "../../src/allocate.h"
#include "../common.h"

#ifdef _OPENMP
#include "../../src/affinity.h"
#include <omp.h>
#endif


void swap_ptrs(CG_FLOAT** x_perm, CG_FLOAT** y_perm){
	CG_FLOAT* tmp = *x_perm;
	*x_perm = *y_perm;
	*y_perm = tmp;
}

int test_spmvSCS(void* args, const char* dataDir){

	int rank = 0;
	int size = 1;
	int validFileCount = 0;

	// Open the directory
	char *pathToMatrices = malloc(strlen(dataDir) + strlen("testMatrices/") + 1);
	strcpy(pathToMatrices, dataDir);	
	strcat(pathToMatrices, "testMatrices/");
	DIR *dir = opendir( pathToMatrices );
	if (dir == NULL) {
			perror("Error opening directory");
			return 1;
	}

	// Read the directory entries
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strstr(entry->d_name, ".mtx") != NULL){
			char *pathToMatrix = malloc(strlen(pathToMatrices) + strlen(entry->d_name) + 1);
			strcpy(pathToMatrix, pathToMatrices);	
			strcat(pathToMatrix, entry->d_name);

			printf("pathToMatrix = %s\n", pathToMatrix);

			Matrix A;
			Args* arguments = (Args*)args;
			A.C = arguments->C;
			A.sigma = arguments->sigma;
			int repeat_count = arguments->run_count;
			char C_str[STR_LEN];                           
			char sigma_str[STR_LEN];
			sprintf(C_str, "%d", A.C);
			sprintf(sigma_str, "%d", A.sigma);

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
			if(fptr){
				++validFileCount;

				MMMatrix m;
				MMMatrixRead( &m, pathToMatrix );

				GMatrix gm;
				matrixConvertfromMM(&m, &gm);

				// Set single rank defaults for MmMatrix
				m.startRow = 0;
				m.stopRow = m.nr;
				m.totalNr = m.nr;
				m.totalNnz = m.nnz;
			
				int vectorSize;
				char* matrixFormat = (char*)malloc(4*sizeof(char)); 

				if(A.C == 0 || A.sigma == 0){
					convertMatrix(&A, &gm);
					vectorSize = A.nr;
					strcpy(matrixFormat, "CRS");
				}
				else{
					convertMatrix(&A, &gm);
					vectorSize = A.nrPadded;
					strcpy(matrixFormat, "SCS");
				}
				VALIDATE_MATRIX_FORMAT(matrixFormat);
				// A.matrixFormat = matrixFormat;

				CG_FLOAT* x = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));
				CG_FLOAT* y = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));

				// Fix x = 1 for now
				for(int i = 0; i < vectorSize; ++i){
					x[i] = (CG_FLOAT)1.0;
					y[i] = (CG_FLOAT)0.0;
				}

				CG_FLOAT* x_perm = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));
				CG_FLOAT* y_perm = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));

				permute_vector(A.oldToNewPerm, x, x_perm, A.nr);
				
				for (size_t i = 0; i < repeat_count; i++)
				{
					spMVM(&A, x_perm, y_perm);
					
					if (i < repeat_count - 1) {
						swap_ptrs(&x_perm, &y_perm);
					}
				}
				
				permute_vector(A.newToOldPerm, y_perm, y, A.nr);
					

				// Dump to this external file
				char *pathToReportedData = malloc(STR_LEN);
      			char out_file_name[64];
      			snprintf(out_file_name, sizeof(out_file_name), "_spmv_x_%d.out", repeat_count);
				BUILD_MATRIX_FILE_PATH(entry, "reported/", out_file_name, C_str, sigma_str, pathToReportedData);
				FILE *reportedData = fopen(pathToReportedData, "w");
				
				printf("pathToReportedData = %s\n", pathToReportedData);
				
				dumpVectorToFile(y, A.nr, reportedData);
				fclose(reportedData);
			
				// If the expect and reported data differ in some way
				int diff_result = diff_files(pathToExpectedData, pathToReportedData);

				// Free per-iteration allocations
				free(matrixFormat);
				free(x);
				free(y);
				free(x_perm);
				free(y_perm);
				free(pathToReportedData);

				if(diff_result){
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

	if(!validFileCount){
		fprintf(stderr, "No valid files found in %s\n", pathToMatrices);
		free(pathToMatrices);
		return 1;
	}
	else{
		free(pathToMatrices);
		return 0;
	}	
}