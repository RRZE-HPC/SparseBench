// DL 2025.04.04
// Single rank test to convert MM to SCS format

#include "../../src/matrix.h"

int main(int argc, char** argv){

	int rank = 0;
	int size = 1;

	MmMatrix m;
	matrixRead(&m, argv[1]);

	SellCSigmaMatrix A;
	matrixConvertMMtoSCS(&m, &A, rank, size);

}