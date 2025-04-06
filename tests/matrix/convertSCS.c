// DL 2025.04.04
// Single rank test to convert MM to SCS format

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "../../src/matrix.h"
#include "matrixCommon.h"

#define BUILD_PATH(entry, dir, expect, c_str, sigma_str, path)  \
	strcpy((path), "data/");                                      \
	strcat((path), (dir));                                        \
	strcat((path), (entry)->d_name);                              \
	strcat((path), "_c_");                                        \
	strcat((path), (c_str));                                      \
	strcat((path), "_sigma_");                                    \
	strcat((path), (sigma_str));                                  \
	strcat((path), (expect));              												\

	typedef struct {
		int c;
		int sigma;
	} Args;

int test_convertSCS(void* args, char* dataDir){

	Args* arguments = (Args*)args;
	int c = arguments->c;
	int sigma = arguments->sigma;
	int rank = 0;
	int size = 1;

	// Open the directory
	DIR *dir = opendir( strcat(dataDir, "testMatrices/") );
	if (dir == NULL) {
			perror("Error opening directory");
			return 1;
	}

	// Read the directory entries
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strstr(entry->d_name, ".mtx") != NULL){
			char *pathToMatrix = malloc(strlen(dataDir) + strlen(entry->d_name) + 1); // +1 for the null terminator
			strcpy(pathToMatrix, dataDir);	
			strcat(pathToMatrix, entry->d_name);

			MmMatrix m;
			matrixRead( &m, pathToMatrix );

			// Set single rank defaults for MmMatrix
			m.startRow = 0;
			m.stopRow = m.nr;
			m.totalNr = m.nr;
			m.totalNnz = m.nnz;
		
			SellCSigmaMatrix A;
			matrixConvertMMtoSCS(&m, &A, c, sigma, rank, size);

			// String preprocessing
			char c_str[20];                           
			sprintf(c_str, "%d", (c));                
			char sigma_str[20];                       
			sprintf(sigma_str, "%d", (sigma));        
			char *dot = strrchr((entry)->d_name, '.');
			if (dot != NULL) {                        
					*dot = '\0';                          
			}       

			// Dump to this external file
			char *pathToReportedData = malloc(100);
			BUILD_PATH(entry, "reported/", ".out", c_str, sigma_str, pathToReportedData);
			FILE *reportedData = fopen(pathToReportedData, "w");
			dumpSCSMatrixToFile(&A, reportedData);
			fclose(reportedData);

			// Check against this external file
			char *pathToExpectedData = malloc(100);
			BUILD_PATH(entry, "expected/", ".in", c_str, sigma_str, pathToExpectedData);

			// Validate against expected data
			if(diff_files(pathToExpectedData, pathToReportedData)) return 1;

			free(pathToMatrix);
		}
	}

	// Close the directory
	closedir(dir);

	return 0;
}