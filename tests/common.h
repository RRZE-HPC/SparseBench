#ifndef __COMMON_H_
#define __COMMON_H_

#include <stdio.h>
#include <string.h>

#define SET_ARGS(i, C_val, sigma_val) \
{                              				\
	args[i]->C = (C_val);         			\
	args[i]->sigma = (sigma_val);				\
}

typedef int (*TestFunc)(void* config, const char* dataDir);

typedef struct {
	const char* name;
	TestFunc func;
} Test;

typedef struct {
int C;
int sigma;
} Args;

#ifndef BUILD_MATRIX_FILE_PATH
#define BUILD_MATRIX_FILE_PATH(entry, dir, expect, C_str, sigma_str, path)  \
{																\
	strcpy((path), "data/");                                    \
	strcat((path), (dir));                                      \
	strcat((path), (entry)->d_name);                            \
	strcat((path), "_C_");                                      \
	strcat((path), (C_str));                                    \
	strcat((path), "_sigma_");                                  \
	strcat((path), (sigma_str));                                \
	strcat((path), (expect));              						\
}
#endif

#ifndef FORMAT_AND_STRIP_MATRIX_FILE
#define FORMAT_AND_STRIP_MATRIX_FILE(A, entry, C_str, sigma_str)	\
{                                                     				\
	sprintf((C_str), "%d", (A).C);                      			\
	sprintf((sigma_str), "%d", (A).sigma);              			\
	char *dot = strrchr((entry)->d_name, '.');          			\
	if (dot != NULL) {                                  			\
		*dot = '\0';                                      			\
	}                                                   			\
}
#endif

#ifndef BUILD_VECTOR_FILE_PATH
#define BUILD_VECTOR_FILE_PATH(entry, dir, expect, path) \
{														 \
	strcpy((path), "data/");                             \
	strcat((path), (dir));                               \
	strcat((path), (entry)->d_name);                     \
	strcat((path), (expect));              				 \
}
#endif

#ifndef FORMAT_AND_STRIP_VECTOR_FILE
#define FORMAT_AND_STRIP_VECTOR_FILE(entry)	\
{                                               \
	char *dot = strrchr((entry)->d_name, '.');  \
	if (dot != NULL) {                          \
		*dot = '\0';                            \
	}                                           \
}
#endif

#ifndef STR_LEN
#define STR_LEN 1024
#endif

// TODO: Read from config.mk
#ifndef ARRAY_ALIGNMENT
#define ARRAY_ALIGNMENT 64
#endif

static int diff_files(const char *expectedData, const char *reportedData) {
	FILE *f1 = fopen(expectedData, "r");
	FILE *f2 = fopen(reportedData, "r");

	if (f1 == NULL || f2 == NULL) {
			perror("Error opening file");
			return 1;
	}

	char line1[2048], line2[2048];
	int line_number = 1;  // Line number counter

	// Compare lines until one of the files ends
	while (fgets(line1, sizeof(line1), f1) != NULL && fgets(line2, sizeof(line2), f2) != NULL) {
			if (strcmp(line1, line2) != 0) {  // If the lines are different
					printf("Files differ at line %d:\n", line_number);
					printf("File 1 (%s): %s\n", expectedData, line1);
					printf("File 2 (%s): %s\n", reportedData, line2);
					fclose(f1);
					fclose(f2);
					return 1;  // Return 1 as soon as a difference is found
			}
			line_number++;
	}

	// Handle case where one file has more lines
	if (fgets(line1, sizeof(line1), f1) != NULL || fgets(line2, sizeof(line2), f2) != NULL) {
			printf("Files differ at line %d:\n", line_number);
			if (fgets(line1, sizeof(line1), f1) != NULL) {
					printf("File 1 (%s): %s\n", expectedData, line1);
			} else {
					printf("File 1 (%s): (no more lines)\n", expectedData);
			}

			if (fgets(line2, sizeof(line2), f2) != NULL) {
					printf("File 2 (%s): %s\n", reportedData, line2);
			} else {
					printf("File 2 (%s): (no more lines)\n", reportedData);
			}

			fclose(f1);
			fclose(f2);
			return 1;  // Files are different if one ends before the other
	}

	fclose(f1);
	fclose(f2);
	return 0;  // Files are identical
}

#endif //__COMMON_H_
