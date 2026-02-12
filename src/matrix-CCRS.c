/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "CCRSMatrix.h"
#include "matrix.h"

void convertMatrix(Matrix *sm, GMatrix *m)
{
  sm = (Matrix *)m;
}

void spMVM(Matrix *m, const CG_FLOAT *restrict x, CG_FLOAT *restrict y)
{
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;
  mEntry *entries = m->entries;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numRows; i++) {
    CG_FLOAT sum = 0.0;

    // loop over all elements in row
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      sum += entries[j].val * x[entries[j].col];
    }

    y[i] = sum;
  }
}
