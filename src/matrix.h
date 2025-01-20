/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <stdbool.h>

#include "util.h"

typedef struct {
  int row;
  int col;
  double val;
} Entry;

typedef struct {
  CG_UINT nr, nc, nnz;       // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of rows and non zeros
  CG_UINT startRow, stopRow;
  CG_UINT *colInd, *rowPtr; // colum Indices, row Pointer
  CG_FLOAT* val;
} Matrix;

extern void matrixRead(Matrix* m, char* filename);

#endif // __MATRIX_H_
