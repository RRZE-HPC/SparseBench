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
#include "matrix.h"
#include "solver.h"
#include "util.h"

static void initVectors(Matrix* m, CG_FLOAT* x, CG_FLOAT* b, CG_FLOAT* xexact)
{
  CG_UINT numRows = m->nr;
  CG_UINT* rowPtr = m->rowPtr;

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
}

void solverCheckResidual(Solver* s, Comm* c)
{
  if (s->xexact == NULL) {
    return;
  }

  CG_FLOAT residual = 0.0;
  CG_FLOAT* v1      = s->x;
  CG_FLOAT* v2      = s->xexact;

  for (int i = 0; i < s->A.nr; i++) {
    double diff = fabs(v1[i] - v2[i]);
    if (diff > residual) residual = diff;
  }

  commReduction(&residual, MAX);

  if (commIsMaster(c)) {
    printf("Difference between computed and exact  = %f\n", residual);
  }
}

void initSolver(Solver* s, Comm* c, Parameter* p)
{
  s->xexact = NULL;

  if (!strcmp(p->filename, "generate")) {
    matrixGenerate(&s->A, p, c->rank, c->size, false);
    s->xexact = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
        s->A.nr * sizeof(CG_FLOAT));
  } else if (!strcmp(p->filename, "generate7P")) {
    matrixGenerate(&s->A, p, c->rank, c->size, true);
    s->xexact = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
        s->A.nr * sizeof(CG_FLOAT));
  } else {
    MmMatrix m;
    matrixRead(&m, p->filename);
    matrixConvertMMtoCRS(&m, &s->A, c->rank, c->size);
  }

  s->x = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  s->b = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, s->A.nr * sizeof(CG_FLOAT));
  initVectors(&s->A, s->x, s->b, s->xexact);
}

void spMVM(Matrix* m, const CG_FLOAT* restrict x, CG_FLOAT* restrict y)
{
  CG_UINT numRows = m->nr;
  CG_UINT* rowPtr = m->rowPtr;
  CG_UINT* colInd = m->colInd;
  CG_FLOAT* val   = m->val;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numRows; i++) {
    CG_FLOAT sum = 0.0;

    // loop over all elements in row
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      sum += val[j] * x[colInd[j]];
    }

    y[i] = sum;
  }
}

void waxpby(const CG_UINT n,
    const CG_FLOAT alpha,
    const CG_FLOAT* restrict x,
    const CG_FLOAT beta,
    const CG_FLOAT* restrict y,
    CG_FLOAT* const w)
{
  if (alpha == 1.0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
}

void ddot(const CG_UINT n,
    const CG_FLOAT* restrict x,
    const CG_FLOAT* restrict y,
    CG_FLOAT* restrict result)
{
  CG_FLOAT sum = 0.0;

  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }

  commReduction(&sum, SUM);
  *result = sum;
}
