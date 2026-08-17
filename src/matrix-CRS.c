/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"

/* Allocate every format-specific array of m in one place. Assumes the size
 * scalars (nr, nnz) are already set on m. Pairs with freeMatrix. */
void allocMatrix(Matrix *m)
{
  m->rowPtr = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (m->nr + 1) * sizeof(CG_UINT));
  m->colInd = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  m->val    = (V_ELE *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(V_ELE));
}

/* Free the arrays allocated by allocMatrix. */
void freeMatrix(Matrix *m)
{
  deallocate(m->rowPtr);
  deallocate(m->colInd);
  deallocate(m->val);
}

void convertMatrix(Matrix *sm, GMatrix *m)
{
  sm->startRow = m->startRow;
  sm->stopRow  = m->stopRow;
  sm->totalNr  = m->totalNr;
  sm->totalNnz = m->totalNnz;
  sm->nr       = m->nr;
  sm->nc       = m->nc;
  sm->nnz      = m->nnz;

  allocMatrix(sm);

  Entry *entries  = m->entries;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

  // Convert to CRS format. Parallel row loop with the same schedule the
  // spMVM kernel uses, so val/colInd/rowPtr pages are first-touched on
  // the NUMA node of the thread that will later read them.
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int rowID = 0; rowID < numRows; rowID++) {
    sm->rowPtr[rowID] = m->rowPtr[rowID];

    // loop over all elements in Row
    for (int id = m->rowPtr[rowID]; id < m->rowPtr[rowID + 1]; id++) {
      sm->val[id]    = entries[id].val;
      sm->colInd[id] = (CG_UINT)entries[id].col;
    }
  }

  sm->rowPtr[numRows] = m->rowPtr[numRows];
}

void spMVM(Matrix *m, const V_ELE *restrict x, V_ELE *restrict y)
{
  CG_UINT *colInd = m->colInd;
  V_ELE *val      = m->val;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numRows; i++) {
    V_ELE sum = 0.0;

    // loop over all elements in row
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      sum += val[j] * x[colInd[j]];
    }

    y[i] = sum;
  }
}

void spMMVM(Matrix *m, const DMatrix *x, DMatrix *y)
{
  CG_UINT *colInd = m->colInd;
  V_ELE *val      = m->val;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int row = 0; row < numRows; row++) {
    V_ELE *y_row = &y->entries[row * y->nc];

    /* initialize output row before accumulation */
#pragma omp simd
    for (size_t c = 0; c < y->nc; c++)
      y_row[c] = 0.0;

    /* loop over all elements in row and accumulate the scaled x[col] row */
    for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
      CG_UINT col  = colInd[j];
      V_ELE *x_col = &x->entries[col * x->nc];
      V_ELE a      = val[j];
#pragma omp simd
      for (size_t c = 0; c < x->nc; c++)
        y_row[c] += a * x_col[c];
    }
  }
}

/* Fused y = cA*(m*x) + cP*p + cQ*q, evaluated per row without ever writing
 * m*x out to memory. q may be NULL (with cQ ignored). */
void spMMVMFused(Matrix *m,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y)
{
  CG_UINT *colInd = m->colInd;
  V_ELE *val      = m->val;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;
  CG_UINT nc      = x->nc;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int row = 0; row < numRows; row++) {
    V_ELE acc[nc];
#pragma omp simd
    for (size_t c = 0; c < nc; c++)
      acc[c] = 0.0;

    for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
      CG_UINT col  = colInd[j];
      V_ELE *x_col = &x->entries[col * nc];
      V_ELE a      = val[j];
#pragma omp simd
      for (size_t c = 0; c < nc; c++)
        acc[c] += a * x_col[c];
    }

    V_ELE *y_row = &y->entries[(CG_UINT)row * nc];
    V_ELE *p_row = &p->entries[(CG_UINT)row * nc];
    if (q != NULL) {
      V_ELE *q_row = &q->entries[(CG_UINT)row * nc];
#pragma omp simd
      for (size_t c = 0; c < nc; c++)
        y_row[c] = cA * acc[c] + cP * p_row[c] + cQ * q_row[c];
    } else {
#pragma omp simd
      for (size_t c = 0; c < nc; c++)
        y_row[c] = cA * acc[c] + cP * p_row[c];
    }
  }
}

/* ChebFD recurrence step, fully fused: computes the new filter term
 * y = cA*(m*w) + cP*w + cQ*q (same shape as spMMVMFused with p=w), and in the
 * same row pass accumulates it into the running polynomial sum,
 * x += gc*y. y may alias q (row-local, in-place recurrence update); x is a
 * separate accumulator block. Saves the extra read of y that a follow-up
 * waxpby(x, gc, y, x) would otherwise need, since y is still local here. */
void chebfdOp(Matrix *m,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x)
{
  CG_UINT *colInd = m->colInd;
  V_ELE *val      = m->val;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;
  CG_UINT nc      = w->nc;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int row = 0; row < numRows; row++) {
    V_ELE acc[nc];
#pragma omp simd
    for (size_t c = 0; c < nc; c++)
      acc[c] = 0.0;

    for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
      CG_UINT col  = colInd[j];
      V_ELE *w_col = &w->entries[col * nc];
      V_ELE a      = val[j];
#pragma omp simd
      for (size_t c = 0; c < nc; c++)
        acc[c] += a * w_col[c];
    }

    V_ELE *w_row = &w->entries[(CG_UINT)row * nc];
    V_ELE *y_row = &y->entries[(CG_UINT)row * nc];
    V_ELE *x_row = &x->entries[(CG_UINT)row * nc];
    if (q != NULL) {
      V_ELE *q_row = &q->entries[(CG_UINT)row * nc];
#pragma omp simd
      for (size_t c = 0; c < nc; c++) {
        V_ELE t  = cA * acc[c] + cP * w_row[c] + cQ * q_row[c];
        y_row[c] = t;
        x_row[c] += gc * t;
      }
    } else {
#pragma omp simd
      for (size_t c = 0; c < nc; c++) {
        V_ELE t  = cA * acc[c] + cP * w_row[c];
        y_row[c] = t;
        x_row[c] += gc * t;
      }
    }
  }
}
