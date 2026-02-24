
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix/matrixTests.h"
#include "solver/solverTestsSPMV.h"
#include "solver/solverTestsSPMMV.h"

int main(int argc, char** argv){
	matrixTests(argc, argv);
	solverTestsSPMV(argc, argv);
	solverTestsSPMMV(argc, argv);
	return 0;
}