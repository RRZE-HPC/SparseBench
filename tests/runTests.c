
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef SCS
#include "matrix/matrixTests.h"
#endif
#include "solver/chebFDGershgorinTests.h"
#include "solver/chebFDStreamTests.h"
#include "solver/chebFDUnitTests.h"
#include "solver/solverTestsSPMMV.h"
#include "solver/solverTestsSPMV.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "../src/cuda_kernels.h"
#endif

// genrator for test codes. One bit per suite.
enum { RC_BIT_BASE = __COUNTER__ };
#define RC_BIT (1 << (__COUNTER__ - RC_BIT_BASE - 1))

enum {
  RC_MATRIX_TESTS      = RC_BIT,
  RC_SOLVER_SPMV       = RC_BIT,
  RC_SOLVER_SPMMV      = RC_BIT,
  RC_CHEBFD_GERSHGORIN = RC_BIT,
  RC_CHEBFD_UNIT       = RC_BIT,
  RC_CHEBFD_STREAM     = RC_BIT,
  // Not a test suite failure
  RC_UNSUPPORTED_MPI   = RC_BIT,
};

// TODO : add tests for MPI cases
int main(int argc, char **argv)
{
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_init(0);
#endif

#if defined(_MPI)
  printf("tests do not support MPI recompile with ENABLE_MPI ?= false\n");
  return RC_UNSUPPORTED_MPI;
#endif

  // since tests needs a reported folder it made sense to centralize it here
  mkdir("./data", 0775);
  mkdir("./data/reported", 0775);

  // Every suite runs even if an earlier one fails, so rc is the OR of the bits
  // of all failing suites and a single run can name all of them at once.
  static const struct {
    const char *name;
    int (*run)(int, char **);
    int bit;
  } suites[] = {
#ifdef SCS
    { "matrixTests",           matrixTests,           RC_MATRIX_TESTS      },
#endif
    { "solverTestsSPMV",       solverTestsSPMV,       RC_SOLVER_SPMV       },
    { "solverTestsSPMMV",      solverTestsSPMMV,      RC_SOLVER_SPMMV      },
    { "chebFDGershgorinTests", chebFDGershgorinTests, RC_CHEBFD_GERSHGORIN },
    { "chebFDUnitTests",       chebFDUnitTests,       RC_CHEBFD_UNIT       },
    { "chebFDStreamTests",     chebFDStreamTests,     RC_CHEBFD_STREAM     },
  };
  const size_t numSuites = sizeof(suites) / sizeof(suites[0]);

  int rc                 = 0;
  for (size_t i = 0; i < numSuites; i++) {
    rc |= suites[i].run(argc, argv) ? suites[i].bit : 0;
  }
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_finalize();
#endif

  if (rc == 0) {
    printf("===>  ALL TESTS PASSED (rc=0)\n");
  } else {
    printf("===>  TESTS FAILED (rc=%d)\n", rc);
    for (size_t i = 0; i < numSuites; i++) {
      if (rc & suites[i].bit) {
        printf("        FAILED: %s (rc bit %d)\n", suites[i].name, suites[i].bit);
      }
    }
  }

  return rc;
}
