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
			char C_str[STR_LEN];                           
			char sigma_str[STR_LEN];
			sprintf(C_str, "%d", A.C);
			sprintf(sigma_str, "%d", A.sigma);

			// String preprocessing
			FORMAT_AND_STRIP_VECTOR_FILE(entry)

			// This is the external file to check against
			char *pathToExpectedData = malloc(STR_LEN);
			BUILD_VECTOR_FILE_PATH(entry, "expected/", "_spmv_x_1.in", pathToExpectedData);

			printf("pathToExpectedData = %s\n", pathToExpectedData);

			// Validate against expected data, if it exists
			
			if(fopen(pathToExpectedData, "r")){
				++validFileCount;

				MmMatrix m;
				matrixRead( &m, pathToMatrix );

				// Set single rank defaults for MmMatrix
				m.startRow = 0;
				m.stopRow = m.nr;
				m.totalNr = m.nr;
				m.totalNnz = m.nnz;
			
				int vectorSize;
				char* matrixFormat = (char*)malloc(4*sizeof(char)); 

				if(A.C == 0 || A.sigma == 0){
					matrixConvertMMtoCRS(&m, &A, rank, size);
					vectorSize = A.nr;
					strcpy(matrixFormat, "CRS");
				}
				else{
					matrixConvertMMtoSCS(&m, &A, rank, size);
					vectorSize = A.nrPadded;
					strcpy(matrixFormat, "SCS");
				}
				VALIDATE_MATRIX_FORMAT(matrixFormat);
				A.matrixFormat = matrixFormat;

				CG_FLOAT* x = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));
				CG_FLOAT* y = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, vectorSize * sizeof(CG_FLOAT));

				// Fix x = 1 for now
				for(int i = 0; i < vectorSize; ++i){
					x[i] = (CG_FLOAT)1.0;
					y[i] = (CG_FLOAT)0.0;
				}

				spMVM(&A, x, y);

				// Dump to this external file
				char *pathToReportedData = malloc(STR_LEN);
				BUILD_MATRIX_FILE_PATH(entry, "reported/", "_spmv_x_1.out", C_str, sigma_str, pathToReportedData);
				FILE *reportedData = fopen(pathToReportedData, "w");
				
				printf("pathToReportedData = %s\n", pathToReportedData);
				
				dumpVectorToFile(y, A.nr, reportedData);
				fclose(reportedData);
			
				// If the expect and reported data differ in some way
				if(diff_files(pathToExpectedData, pathToReportedData)){
					free(pathToReportedData);
					free(pathToExpectedData);
					free(pathToMatrix);

					closedir(dir);
					return 1;
				} 
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