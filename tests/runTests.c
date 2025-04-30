
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix/matrixTests.h"
#include "solver/solverTests.h"

int main(int argc, char** argv){
	matrixTests(argc, argv);
	solverTests(argc, argv);
	
	return 0;
}