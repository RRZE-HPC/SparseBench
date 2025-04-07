/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "allocate.h"
#include "comm.h"
#include "matrix.h"
#include "parameter.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "util.h"
#include "debugger.h"

int main(int argc, char** argv)
{
  double timeStart, timeStop, ts;
  Parameter param;
  Solver s;
  Comm comm;

  commInit(&comm, argc, argv);
  initParameter(&param);

  if (argc != 2) {
    if (commIsMaster(&comm)) {
      printf("Usage: %s <configFile>\n", argv[0]);
    }
    commFinalize(&comm);
    exit(EXIT_SUCCESS);
  }

  readParameter(&param, argv[1]);
  commPrintBanner(&comm);

  // DL: For testing. Options are CRS or SCS
  char* matrixFormat = (char*)malloc(4*sizeof(char)); strcpy(matrixFormat, "CRS");
  VALIDATE_MATRIX_FORMAT(matrixFormat);

  CG_FLOAT eps = (CG_FLOAT)param.eps;
  int itermax  = param.itermax;
  initSolver(&s, &comm, &param, matrixFormat);
  profilerInit();
  commPartition(&comm, &s.A);
  commPrintConfig(&comm, s.A.nr, s.A.startRow, s.A.stopRow);
  // commMatrixDump(&comm, &s.A);
  // commFinalize(&comm);
  // return EXIT_SUCCESS;

  CG_UINT nrow = s.A.nr;
  CG_UINT ncol = s.A.nc;
  CG_FLOAT* r  = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT* p;
  CG_FLOAT* Ap;
  if(IS_EQUAL(matrixFormat, "CRS")){
    p  = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, ncol * sizeof(CG_FLOAT));
    Ap = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  }
  else if(IS_EQUAL(matrixFormat, "SCS")){
    p = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, s.A.nrPadded * sizeof(CG_FLOAT));
    Ap = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, s.A.nrPadded * sizeof(CG_FLOAT));
  }

  CG_FLOAT normr  = 0.0;
  CG_FLOAT rtrans = 0.0, oldrtrans = 0.0;

  int printFreq = itermax / 10;
  if (printFreq > 50) {
    printFreq = 50;
  }
  if (printFreq < 1) {
    printFreq = 1;
  }

  PROFILE(WAXPBY, waxpby(nrow, 1.0, s.x, 0.0, s.x, p));
  PROFILE(COMM, commExchange(&comm, &s.A, p));
  PROFILE(SPMVM, spMVM(&s.A, p, Ap));
  PROFILE(WAXPBY, waxpby(nrow, 1.0, s.b, -1.0, Ap, r));
  PROFILE(DDOT, ddot(nrow, r, r, &rtrans));

  normr = sqrt(rtrans);
  if (commIsMaster(&comm)) {
    printf("Initial Residual = %E\n", normr);
  }

  int k;
  timeStart = getTimeStamp();
  for (k = 1; k < itermax && normr > eps; k++) {
    if (k == 1) {
      PROFILE(WAXPBY, waxpby(nrow, 1.0, r, 0.0, r, p));
    } else {
      oldrtrans = rtrans;
      PROFILE(DDOT, ddot(nrow, r, r, &rtrans));
      double beta = rtrans / oldrtrans;
      PROFILE(WAXPBY, waxpby(nrow, 1.0, r, beta, p, p));
    }
    normr = sqrt(rtrans);

    if (commIsMaster(&comm) && (k % printFreq == 0 || k + 1 == itermax)) {
      printf("Iteration = %d Residual = %E\n", k, normr);
    }

    PROFILE(COMM, commExchange(&comm, &s.A, p));
    PROFILE(SPMVM, spMVM(&s.A, p, Ap));
    double alpha = 0.0;
    PROFILE(DDOT, ddot(nrow, p, Ap, &alpha));
    alpha = rtrans / alpha;
    PROFILE(WAXPBY, waxpby(nrow, 1.0, s.x, alpha, p, s.x));
    PROFILE(WAXPBY, waxpby(nrow, 1.0, r, -alpha, Ap, r));
  }
  timeStop = getTimeStamp();

  if (commIsMaster(&comm)) {
    printf("Solution performed %d iterations and took %.2fs\n",
        k,
        timeStop - timeStart);
  }

  profilerPrint(&comm, &s, k);
  solverCheckResidual(&s, &comm);
  profilerFinalize();
  commFinalize(&comm);

  // DL: For testing
  free(matrixFormat);

  return EXIT_SUCCESS;
}
