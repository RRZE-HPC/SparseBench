/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "parameter.h"
#include "util.h"

#define CRS
#ifdef CRS
#include "CRSMatrix.h"
#endif
#ifdef SCS
#include "SCSMatrix.h"
#endif

typedef struct {
  CG_UINT col;
  CG_FLOAT val;
} Entry;

typedef struct {
  CG_UINT nr, nc, nnz;       // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT* rowPtr;           // row Pointer
  Entry* entries;
} GMatrix;

typedef struct {
  int row;
  int col;
  double val;
} MMEntry;

typedef struct {
  size_t count;
  int nr, nnz;
  int totalNr, totalNnz; // number of total rows and non zeros
  int startRow, stopRow; // range of rows owned by current rank
  MMEntry* entries;
} MMMatrix;

extern void dumpMMMatrix(MMMatrix* m);
extern void MMMatrixRead(MMMatrix* m, char* filename);
extern void matrixConvertfromMM(MMMatrix* mm, GMatrix* m);

extern void dumpGMatrix(GMatrix* m);
extern void matrixGenerate(
    GMatrix* m, Parameter* p, int rank, int size, bool use_7pt_stencil);
extern void dumpMatrix(Matrix* m);
extern void convertMatrix(Matrix* m, GMatrix* im);

#endif // __MATRIX_H_
