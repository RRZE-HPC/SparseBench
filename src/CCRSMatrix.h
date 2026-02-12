/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CCRSMATRIX_H_
#define __CCRSMATRIX_H_
#include "util.h"

typedef struct {
  CG_UINT col;
  CG_FLOAT val;
} mEntry;

typedef struct {
  CG_UINT nr, nc, nnz; // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *rowPtr; // row Pointer
  mEntry *entries;
} Matrix;

#endif // __CCRSMATRIX_H_
