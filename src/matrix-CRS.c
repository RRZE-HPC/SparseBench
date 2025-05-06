/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"

void convertMatrix(Matrix *m, GMatrix *im) {
  m->startRow = mm->startRow;
  m->stopRow = mm->stopRow;
  m->totalNr = mm->totalNr;
  m->totalNnz = mm->totalNnz;
  m->nr = mm->nr;
  m->nc = mm->nr;
  m->nnz = mm->nnz;

  m->rowPtr =
      (CG_UINT *)allocate(ARRAY_ALIGNMENT, (m->nr + 1) * sizeof(CG_UINT));
  m->colInd = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  m->val = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_FLOAT));

  int *valsPerRow = (int *)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(int));

  for (int i = 0; i < m->nr; i++) {
    valsPerRow[i] = 0;
  }

  Entry *entries = mm->entries;
  int startRow = m->startRow;

  for (int i = 0; i < mm->count; i++) {
    valsPerRow[entries[i].row - startRow]++;
  }

  m->rowPtr[0] = 0;

  // convert to CRS format
  for (int rowID = 0; rowID < m->nr; rowID++) {
    m->rowPtr[rowID + 1] = m->rowPtr[rowID] + valsPerRow[rowID];

    // loop over all elements in Row
    for (int id = m->rowPtr[rowID]; id < m->rowPtr[rowID + 1]; id++) {
      m->val[id] = (CG_FLOAT)entries[id].val;
      m->colInd[id] = (CG_UINT)entries[id].col;
    }
  }
}
