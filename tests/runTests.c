
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef SCS
#include "matrix/matrixTests.h"
#endif
#include "solver/chebFDGershgorinTests.h"
#include "solver/solverTestsSPMMV.h"
#include "solver/solverTestsSPMV.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "../src/cuda/cuda_kernels.h"
#endif

// TODO : add tests for MPI cases
int main(int argc, char **argv)
{
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_init(0);
#endif

#if defined(_MPI)
  printf("tests do not support MPI recompile with ENABLE_MPI ?= false\n");
  return -1;
#endif

  // since tests needs a reported folder it made sense to centralize it here
  mkdir("./data", 0775);
  mkdir("./data/reported", 0775);

#ifdef SCS
  matrixTests(argc, argv);
#endif
  solverTestsSPMV(argc, argv);
  solverTestsSPMMV(argc, argv);
  chebFDGershgorinTests(argc, argv);
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_finalize();
#endif
  return 0;
}