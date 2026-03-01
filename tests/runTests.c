
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef SCS
#include "matrix/matrixTests.h"
#endif
#include "solver/solverTestsSPMMV.h"
#include "solver/solverTestsSPMV.h"

int main(int argc, char **argv)
{
#ifdef SCS
  matrixTests(argc, argv);
#endif
  solverTestsSPMV(argc, argv);
  solverTestsSPMMV(argc, argv);
  return 0;
}