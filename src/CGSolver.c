/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "comm.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "util.h"

static void initVectors(Matrix *m, CG_FLOAT *x, CG_FLOAT *b, CG_FLOAT *xexact)
{
#ifdef CRS
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

  for (int rowID = 0; rowID < numRows; rowID++) {

    int nnzrow = rowPtr[rowID + 1] - rowPtr[rowID];
    x[rowID]   = 0.0;

    if (xexact != NULL) {
      b[rowID]      = 27.0 - ((CG_FLOAT)(nnzrow - 1));
      xexact[rowID] = 1.0;
    } else {
      b[rowID] = 1.0;
    }
  }
#endif
}

void solverCheckResidual(CommType *c, CG_FLOAT *x, CG_FLOAT *xexact, CG_UINT n)
{
  if (xexact == NULL) {
    return;
  }

  CG_FLOAT residual = 0.0;
  CG_FLOAT *v1      = x;
  CG_FLOAT *v2      = xexact;

  for (int i = 0; i < n; i++) {
    double diff = fabs(v1[i] - v2[i]);
    if (diff > residual)
      residual = diff;
  }

  commReduction(&residual, MAX);

  if (commIsMaster(c)) {
    printf("Difference between computed and exact  = %f\n", residual);
  }
}

int solveCG(CommType *comm, Parameter *param, Matrix *A)
{
  CG_FLOAT eps     = (CG_FLOAT)param->eps;
  int itermax      = param->itermax;

  CG_UINT nrow     = A->nr;
  CG_UINT ncol     = A->nc;
  CG_FLOAT *r      = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT *p      = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, ncol * sizeof(CG_FLOAT));
  CG_FLOAT *Ap     = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT *x      = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT *b      = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT *xexact = NULL;

  if (strcmp(param->filename, "generate") == 0 ||
      strcmp(param->filename, "generate7P") == 0) {
    xexact = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  }
  initVectors(A, x, b, xexact);

  CG_FLOAT normr  = 0.0;
  CG_FLOAT rtrans = 0.0, oldrtrans = 0.0;

  int printFreq = itermax / 10;
  if (printFreq > 50) {
    printFreq = 50;
  }
  if (printFreq < 1) {
    printFreq = 1;
  }
  double timeStart, timeStop, ts;

  PROFILE(WAXPBY, waxpby(nrow, 1.0, x, 0.0, x, p));
  PROFILE(COMM, commExchange(comm, A->nr, p));
  PROFILE(SPMVM, spMVM(A, p, Ap));
  PROFILE(WAXPBY, waxpby(nrow, 1.0, b, -1.0, Ap, r));
  PROFILE(DDOT, ddot(nrow, r, r, &rtrans));

  normr = sqrt(rtrans);
  if (commIsMaster(comm)) {
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

    if (commIsMaster(comm) && (k % printFreq == 0 || k + 1 == itermax)) {
      printf("Iteration = %d Residual = %E\n", k, normr);
    }

    PROFILE(COMM, commExchange(comm, A->nr, p));
    PROFILE(SPMVM, spMVM(A, p, Ap));
    CG_FLOAT alpha = 0.0;
    PROFILE(DDOT, ddot(nrow, p, Ap, &alpha));
    alpha = rtrans / alpha;
    PROFILE(WAXPBY, waxpby(nrow, 1.0, x, alpha, p, x));
    PROFILE(WAXPBY, waxpby(nrow, 1.0, r, -alpha, Ap, r));
  }
  timeStop = getTimeStamp();

  if (commIsMaster(comm)) {
    printf("Solution performed %d iterations and took %.2fs\n", k, timeStop - timeStart);
  }

  solverCheckResidual(comm, x, xexact, A->nr);

  return k;
}
