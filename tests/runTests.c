
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef SCS
#include "matrix/matrixTests.h"
#endif
#include "solver/solverTestsSPMMV.h"
#include "solver/solverTestsSPMV.h"
#include "solver/solverTestsSPMVSplit.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "../src/cuda/cuda_kernels.h"
#endif

int main(int argc, char **argv)
{
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_init(0);
#endif
#ifdef SCS
  matrixTests(argc, argv);
#endif
  // Self-contained and fast, so run it before the data-driven tests
  solverTestsSPMVSplit(argc, argv);
  solverTestsSPMV(argc, argv);
  solverTestsSPMMV(argc, argv);
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_finalize();
#endif
  return 0;
}