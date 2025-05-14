// DL 2025.04.04
// Single rank test to convert MM to SCS format

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "../../src/matrix.h"
#include "../common.h"

int test_convertSCS(void* args, const char* dataDir){

	int rank = 0;
	int size = 1;

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

			Matrix A;
			Args* arguments = (Args*)args;
			A.C = arguments->C;
			A.sigma = arguments->sigma;

			// String preprocessing
			char C_str[STR_LEN];                           
			char sigma_str[STR_LEN];
			FORMAT_AND_STRIP_MATRIX_FILE(A, entry, C_str, sigma_str)

			// This is the external file to check against
			char *pathToExpectedData = malloc(STR_LEN);
			BUILD_MATRIX_FILE_PATH(entry, "expected/", ".in", C_str, sigma_str, pathToExpectedData);

			// Validate against expected data, if it exists
			if(fopen(pathToExpectedData, "r")){

				MmMatrix m;
				matrixRead( &m, pathToMatrix );

				// Set single rank defaults for MmMatrix
				m.startRow = 0;
				m.stopRow = m.nr;
				m.totalNr = m.nr;
				m.totalNnz = m.nnz;
			
				matrixConvertMMtoSCS(&m, &A, rank, size);

				// Dump to this external file
				char *pathToReportedData = malloc(STR_LEN);
				BUILD_MATRIX_FILE_PATH(entry, "reported/", ".out", C_str, sigma_str, pathToReportedData);
				FILE *reportedData = fopen(pathToReportedData, "w");

				dumpSCSMatrixToFile(&A, reportedData);
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

	free(pathToMatrices);
	closedir(dir);

	return 0;
}