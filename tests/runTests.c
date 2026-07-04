
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef SCS
#include "matrix/matrixTests.h"
#endif
#include "solver/solverTestsSPMMV.h"
#include "solver/solverTestsSPMV.h"
#include "solver/chebFDGershgorinTests.h"

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