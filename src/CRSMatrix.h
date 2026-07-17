/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CRSMATRIX_H_
#define __CRSMATRIX_H_
#include "util.h"
#include "vtype.h"

typedef struct {
  CG_UINT nr, nc, nnz; // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *rowPtr; // row Pointer
  CG_UINT *rowLocalEnd; // first colInd index of external entries in each row (enables split SpMV)
  CG_UINT *colInd; // column Indices
  V_ELE *val; // matrix entries
} Matrix;

#endif // __CRSMATRIX_H_
