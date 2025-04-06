
#include "convertSCS.h"

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#define SET_ARGS(i, c_val, sigma_val) \
{                              				\
	args[i]->c = (c_val);         			\
	args[i]->sigma = (sigma_val);				\
}

typedef int (*TestFunc)(void* config, const char* dataDir);

typedef struct {
	const char* name;
	TestFunc func;
} Test;

typedef struct {
int c;
int sigma;
} Args;

int matrixTests(int argc, char** argv){
	// Hard-code data directory
	char* dataDir = malloc(6);
	if (dataDir) strcpy(dataDir, "data/");

	// Alternatively, if you want to get the data dir from the command line
  // Check if the user has provided the directory path
  //   if (argc != 2) {
	// 		fprintf(stderr, "Usage: %s <directory_path>\n", argv[0]);
	// 		return 1; // Exit with error code if not provided
	// }

	// Get the directory path from the command line argument
	// const char *dataDir = argv[1];

	Test tests[] = {
		{ "convertSell-1-1", test_convertSCS }
		// { "convertSell-1-2", test_convertSCS }
		// Add more here
	};

	int num_tests = sizeof(tests) / sizeof(tests[0]);
	int passed = 0;

	Args** args = (Args **)malloc(num_tests * sizeof(Args*));
	for (int i = 0; i < num_tests; ++i) {
		args[i] = (Args*)malloc(sizeof(Args));
		if (!args[i]) {
				printf("Memory allocation failed for test %d!\n", i);
				return 1;
		}
	}

	// Manually assign configurations
	SET_ARGS(0, 1, 1);

	printf("Running %d tests:\n", num_tests);
	for (int i = 0; i < num_tests; ++i) {
			printf("[%-2d/%-2d] %-20s ... \n", i+1, num_tests, tests[i].name);
			fflush(stdout);

			if (!(tests[i].func((void*)args[i], dataDir))) {
					printf("✅ PASS\n");
					passed++;
			} else {
					printf("❌ FAIL\n");
			}
	}

	printf("\nSummary: %d/%d tests passed.\n", passed, num_tests);

	free(dataDir);
	free(args);

	return (passed == num_tests) ? 0 : 1;
}