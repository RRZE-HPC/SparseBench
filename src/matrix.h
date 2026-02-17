/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "parameter.h"
#include "util.h"

#ifdef CRS
#include "CRSMatrix.h"
#endif
#ifdef SCS
#include "SCSMatrix.h"
#endif
#ifdef CCRS
#include "CCRSMatrix.h"
#endif

typedef struct {
  CG_UINT col;
  CG_FLOAT val;
} Entry;

typedef struct {
  CG_UINT nr, nc, nnz; // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *rowPtr; // row Pointer
  Entry *entries;
} GMatrix;

typedef struct {
  CG_UINT nr, nc; // number of rows, columns
  CG_FLOAT *entries;
} DMatrix; // for Block vectors

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
  MMEntry *entries;
} MMMatrix;

extern void MMMatrixRead(MMMatrix *m, char *filename);
extern void matrixConvertfromMM(MMMatrix *mm, GMatrix *m);

extern void matrixGenerate(
    GMatrix *m, Parameter *p, int rank, int size, bool use_7pt_stencil);

extern void convertMatrix(Matrix *m, GMatrix *im);

extern void MMMatrixPrintToFile(MMMatrix *m, char *filename);
extern void MMMatrixPrint(MMMatrix *m);
extern void MMMatrixPrint_impl(MMMatrix *m, FILE *fptr);

extern void dumpMatrixToFile(Matrix *m, char *filename);
extern void dumpMatrix(Matrix *m);
extern void dumpMatrix_impl(Matrix *m, FILE *fptr);

extern void GMatrixPrintTofile(GMatrix *m, char *filename);
extern void GMatrixPrint(GMatrix *m);
extern void GMatrixPrint_impl(GMatrix *m, FILE *fptr);

extern void MatrixPrintTofile(Matrix *m, char *filename);
extern void MatrixPrint(Matrix *m);
extern void MatrixPrint_impl(Matrix *m, FILE *fptr);


#endif // __MATRIX_H_
