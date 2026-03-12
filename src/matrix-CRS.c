/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"

void convertMatrix(Matrix *sm, GMatrix *m)
{
  sm->startRow    = m->startRow;
  sm->stopRow     = m->stopRow;
  sm->totalNr     = m->totalNr;
  sm->totalNnz    = m->totalNnz;
  sm->nr          = m->nr;
  sm->nc          = m->nc;
  sm->nnz         = m->nnz;

  sm->rowPtr      = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (m->nr + 1) * sizeof(CG_UINT));
  sm->colInd      = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  sm->val         = (V_ELE *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(V_ELE));

  Entry *entries  = m->entries;

  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

  // convert to CRS format
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
    for (size_t c = 0; c < y->nc; c++)
      y_row[c] = 0.0;

    /* loop over all elements in row and accumulate the scaled x[col] row */
    for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
      CG_UINT col   = colInd[j];
      V_ELE *x_col  = &x->entries[col * x->nc];
      V_ELE a       = val[j];
      for (size_t c = 0; c < x->nc; c++)
        y_row[c] += a * x_col[c];
    }
  }
}
