/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
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
#include "matrix.h"
#include "profiler.h"
#include "solver.h"
#include "util.h"

static void initVectors(Matrix *m, CG_FLOAT *x, CG_FLOAT *b, CG_FLOAT *xexact) {
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

  for (int rowID = 0; rowID < numRows; rowID++) {

    int nnzrow = rowPtr[rowID + 1] - rowPtr[rowID];
    x[rowID] = 0.0;

    if (xexact != NULL) {
      b[rowID] = 27.0 - ((CG_FLOAT)(nnzrow - 1));
      xexact[rowID] = 1.0;
    } else {
      b[rowID] = 1.0;
    }
  }
}

void solverCheckResidual(Solver *s, Comm *c) {
  if (s->xexact == NULL) {
    return;
  }

  CG_FLOAT residual = 0.0;
  CG_FLOAT *v1 = s->x;
  CG_FLOAT *v2 = s->xexact;

  for (int i = 0; i < s->A.nr; i++) {
    double diff = fabs(v1[i] - v2[i]);
    if (diff > residual)
      residual = diff;
  }

  commReduction(&residual, MAX);

  if (commIsMaster(c)) {
    printf("Difference between computed and exact  = %f\n", residual);
  }
}

void initSolver(Solver *s, Comm *c, Parameter *p, char *matrixFormat) {
  s->xexact = NULL;

  if (strcmp(p->filename, "generate") == 0) {
    matrixGenerate(&s->A, p, c->rank, c->size, false);
    s->xexact =
        (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  } else if (strcmp(p->filename, "generate7P") == 0) {
    matrixGenerate(&s->A, p, c->rank, c->size, true);
    s->xexact =
        (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  } else {
    MmMatrix m, mLocal;
    if (commIsMaster(c)) {
      matrixRead(&m, p->filename);
    }
    commDistributeMatrix(c, &m, &mLocal);
    if (IS_EQUAL(matrixFormat, "CRS")) {
      matrixConvertMMtoCRS(&mLocal, &s->A, c->rank, c->size);
    } else if (IS_EQUAL(matrixFormat, "SCS")) {
      // DL: For testing
      s->A.C = (CG_UINT)1;
      s->A.sigma = (CG_UINT)1;
      matrixConvertMMtoSCS(&mLocal, &s->A, c->rank, c->size);
    }
    s->A.matrixFormat = matrixFormat;
  }

  s->x = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  s->b = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  initVectors(&s->A, s->x, s->b, s->xexact);
}

int solveCG(Parameter *param, Matrix *m) {
  CG_FLOAT eps = (CG_FLOAT)param->eps;
  int itermax = param->itermax;

  CG_UINT nrow = s.A.nr;
  CG_UINT ncol = s.A.nc;
  CG_FLOAT *r = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));
  CG_FLOAT *p;
  CG_FLOAT *Ap;
  p = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, ncol * sizeof(CG_FLOAT));
  Ap = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(CG_FLOAT));

  CG_FLOAT normr = 0.0;
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
    printf("Solution performed %d iterations and took %.2fs\n", k,
           timeStop - timeStart);
  }

  return k;
}
