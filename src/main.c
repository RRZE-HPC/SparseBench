/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "comm.h"
#include "matrix.h"
#include "parameter.h"
#include "solver.h"
#include "timing.h"
#include "util.h"

int main(int argc, char** argv)
{
  double timeStart, timeStop;
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

  CG_FLOAT eps = (CG_FLOAT)param.eps;
  int itermax  = param.itermax;
  initSolver(&s, &comm, &param);
  // commMatrixDump(&comm, &s.A);
  // exit(EXIT_SUCCESS);
  commPartition(&comm, &s.A);
  // commPrintConfig(&comm);

  CG_UINT nrow = s.A.nr;
  CG_UINT ncol = s.A.nc;
  CG_FLOAT* r  = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT* p  = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, ncol * sizeof(CG_FLOAT));
  CG_FLOAT* Ap = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT normr  = 0.0;
  CG_FLOAT rtrans = 0.0, oldrtrans = 0.0;

  int print_freq = itermax / 10;
  if (print_freq > 50) {
    print_freq = 50;
  }
  if (print_freq < 1) {
    print_freq = 1;
  }

  waxpby(nrow, 1.0, s.x, 0.0, s.x, p);
  commExchange(&comm, &s.A, p);
  spMVM(&s.A, p, Ap);
  waxpby(nrow, 1.0, s.b, -1.0, Ap, r);
  ddot(nrow, r, r, &rtrans);

  normr = sqrt(rtrans);
  if (commIsMaster(&comm)) {
    printf("Initial Residual = %E\n", normr);
  }

  int k;
  timeStart = getTimeStamp();
  for (k = 1; k < itermax && normr > eps; k++) {
    if (k == 1) {
      waxpby(nrow, 1.0, r, 0.0, r, p);
    } else {
      oldrtrans = rtrans;
      ddot(nrow, r, r, &rtrans);
      double beta = rtrans / oldrtrans;
      waxpby(nrow, 1.0, r, beta, p, p);
    }
    normr = sqrt(rtrans);

    if (commIsMaster(&comm) && (k % print_freq == 0 || k + 1 == itermax)) {
      printf("Iteration = %d Residual = %E\n", k, normr);
    }

    commExchange(&comm, &s.A, p);
    spMVM(&s.A, p, Ap);
    // printf("ITERATION = %d ################### Ap\n", k);
    // for (int i = 0; i < nrow; i++) {
    //   printf("%f\n", Ap[i]);
    // }
    double alpha = 0.0;
    ddot(nrow, p, Ap, &alpha);
    alpha = rtrans / alpha;
    waxpby(nrow, 1.0, s.x, alpha, p, s.x);
    waxpby(nrow, 1.0, r, -alpha, Ap, r);
  }
  timeStop = getTimeStamp();

  if (commIsMaster(&comm)) {
    printf("Solution performed %d iterations and took %.2fs\n",
        k,
        timeStop - timeStart);
  }

  solverCheckResidual(&s, &comm);
  commFinalize(&comm);
  return EXIT_SUCCESS;
}
