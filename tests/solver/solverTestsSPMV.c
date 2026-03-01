#include "../common.h"
#include "spmvSCS.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int solverTestsSPMV(int argc, char **argv)
{
  // Hard-code data directory
  char *dataDir = malloc(6);
  if (dataDir)
    strcpy(dataDir, "./data/");

  // Alternatively, if you want to get the data dir from the command line
  // Check if the user has provided the directory path
  //   if (argc != 2) {
  // 		fprintf(stderr, "Usage: %s <directory_path>\n", argv[0]);
  // 		return 1; // Exit with error code if not provided
  // }

  // Get the directory path from the command line argument
  // const char *dataDir = argv[1];

  Test tests[REP_COUNT * C_SIGMA_MAX * C_SIGMA_MAX] = { };

  int num_tests                                     = sizeof(tests) / sizeof(tests[0]);
  int passed                                        = 0;

  Args **args = (Args **)malloc(num_tests * sizeof(Args *));
  for (int i = 0; i < num_tests; ++i) {
    args[i] = (Args *)malloc(sizeof(Args));
    if (!args[i]) {
      printf("Memory allocation failed for test %d!\n", i);
      return 1;
    }
  }

  // Manually assign one configuration per test
  int idx = 0;
  for (int i = 1; i <= 3; ++i) {
    for (int sigma = 1; sigma <= C_SIGMA_MAX; sigma++) {
      for (int c = 1; c <= C_SIGMA_MAX; c++) {
        char buff[64];
        snprintf(buff, sizeof(buff), "SpMV_%d Sell-%d-%d", i, c, sigma);
        tests[idx] = (Test) { "", test_spmvSCS };
        strcpy(tests[idx].name, buff);
        SET_ARGS(idx, c, sigma, i); // Test idx, c, sigma, repeat
        ++idx;
      }
    }
  }

  printf("Running %d Solver tests:\n", num_tests);
  for (int i = 0; i < num_tests; ++i) {
    printf("[%-2d/%-2d] %-20s ... \n", i + 1, num_tests, tests[i].name);
    fflush(stdout);

    if (!(tests[i].func((void *)args[i], dataDir))) {
      printf("✅ PASS\n");
      passed++;
    } else {
      printf("❌ FAIL\n");
    }
  }

  printf("\nSummary: %d/%d Solver SPMV tests passed.\n", passed, num_tests);

  free(dataDir);
  free(args);

  return (passed == num_tests) ? 0 : 1;
}