/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <stdbool.h>
#include <stddef.h>

#include "parameter.h"
#include "util.h"

typedef struct {
  int row;
  int col;
  double val;
} Entry;

typedef struct {
  size_t count;
  int nr, nnz;
  int totalNr, totalNnz; // number of total rows and non zeros
  int startRow, stopRow; // range of rows owned by current rank
  Entry* entries;
} MmMatrix;

typedef struct {
  CG_UINT nr, nc, nnz;       // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *colInd, *rowPtr;  // colum Indices, row Pointer
  CG_FLOAT* val;             // matrix entries
} Matrix;

extern void matrixRead(MmMatrix* m, char* filename);
extern void matrixConvertMMtoCRS(MmMatrix* mm, Matrix* m, int rank, int size);
extern void matrixGenerate(
    Matrix* m, Parameter* p, int rank, int size, bool use_7pt_stencil);

#endif // __MATRIX_H_
