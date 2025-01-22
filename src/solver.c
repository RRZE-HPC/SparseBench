/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "solver.h"
#include "util.h"

void initSolver(Solver* s, Comm* c, Parameter* p)
{
  if (!strcmp(p->filename, "generate")) {
    matrixGenerate(&s->A, p, c->rank, c->size, false);
  } else if (!strcmp(p->filename, "generate7P")) {
    matrixGenerate(&s->A, p, c->rank, c->size, true);
  } else {
    MmMatrix m;
    matrixRead(&m, p->filename);
    matrixConvertMMtoCRS(&m, &s->A, c->rank, c->size);
  }
}

void spMVM(Matrix* m, const CG_FLOAT* restrict x, CG_FLOAT* restrict y)
{
  CG_UINT numRows = m->nr;
  CG_UINT* rowPtr = m->rowPtr;
  CG_UINT* colInd = m->colInd;
  CG_FLOAT* val   = m->val;

#pragma omp parallel for
  for (int rowID = 0; rowID < numRows; rowID++) {
    CG_FLOAT tmp = y[rowID];

    // loop over all elements in row
    for (int entry = rowPtr[rowID]; entry < rowPtr[rowID + 1]; entry++) {
      tmp += val[entry] * x[colInd[entry]];
    }

    y[rowID] = tmp;
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
    for (int i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
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
    for (int i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }

  commReduction(&sum, SUM);
  *result = sum;
}
