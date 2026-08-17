#ifndef __CHEBFDUNITTESTS_H_
#define __CHEBFDUNITTESTS_H_

// Standalone unit tests for each internal step of the ChebFD algorithm
// (chebFilterInit, jacobiEigen, the fused spMMVM kernels, applyFilter,
// orthoMGS, rayleighRitz/residual, and an end-to-end solveChebFD sanity
// check). Returns 0 if all sections pass, 1 otherwise.
int chebFDUnitTests(int argc, char **argv);

#endif // __CHEBFDUNITTESTS_H_
