#ifndef __MATIRX_COMMON_H_
#define __MATIRX_COMMON_H_

#include <stdio.h>
#include <string.h>

int diff_files(const char *expectedData, const char *reportedData) {
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
					printf("File 1 (%s): %s", expectedData, line1);
					printf("File 2 (%s): %s", reportedData, line2);
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
					printf("File 1 (%s): %s", expectedData, line1);
			} else {
					printf("File 1 (%s): (no more lines)\n", expectedData);
			}

			if (fgets(line2, sizeof(line2), f2) != NULL) {
					printf("File 2 (%s): %s", reportedData, line2);
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

#endif //__MATIRX_COMMON_H_
