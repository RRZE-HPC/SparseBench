/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* restrict is not a keyword in C++; hide it for nvcc/hipcc */
#ifdef __cplusplus
#define restrict __restrict__
#endif

#include "parameter.h"
#include "util.h"
#include "vtype.h"

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
  V_ELE val;
} Entry;

typedef struct {
  CG_UINT nr, nc, nnz;       // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *rowPtr;           // row Pointer
  Entry *entries;
} GMatrix;

typedef struct {
  CG_UINT nr, nc; // number of rows, columns
  V_ELE *entries;
} DMatrix; // for Block vectors

/* Rows a (block) vector compatible with m must have: SCS pads the row count
 * to a multiple of the chunk height C, CRS does not. */
static inline CG_UINT matrixVecRows(const Matrix *m)
{
#ifdef SCS
  return m->nrPadded;
#else
  return m->nr;
#endif
}

typedef struct {
  int row;
  int col;
  double val;
  double val_imag;
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

// sizes of the nnz, nr etc must be already set so matrix specific allocation
// and deallocation can be centralized
extern void allocMatrix(Matrix *m);
extern void freeMatrix(Matrix *m);

// Free the CRS and MMmatrix style
extern void freeGMatrix(GMatrix *m);
extern void freeMMMatrix(MMMatrix *m);

extern void MMMatrixPrintToFile(MMMatrix *m, char *filename);
extern void MMMatrixPrint(MMMatrix *m);
extern void MMMatrixPrint_impl(MMMatrix *m, FILE *fptr);

extern void dumpDMatrixToFile(DMatrix *m, char *filename);
extern void dumpDMatrix(DMatrix *m);
extern void dumpDMatrix_impl(DMatrix *m, FILE *fptr);

extern void dumpMatrixToFile(Matrix *m, char *filename);
extern void dumpMatrix(Matrix *m);
extern void dumpMatrix_impl(Matrix *m, FILE *fptr);

extern void GMatrixPrintTofile(GMatrix *m, char *filename);
extern void GMatrixPrint(GMatrix *m);
extern void GMatrixPrint_impl(GMatrix *m, FILE *fptr);

extern void MatrixPrintTofile(Matrix *m, char *filename);
extern void MatrixPrint(Matrix *m);
extern void MatrixPrint_impl(Matrix *m, FILE *fptr);

extern void dumpVectorPrint(V_ELE *restrict y, CG_UINT numRows);
extern void dumpVectorToFile(V_ELE *restrict y, CG_UINT numRows, FILE *reportedData);

extern void permute_DMatrix(const CG_UINT *perm, const DMatrix *src, DMatrix *dst);

extern void permute_vector(
    const CG_UINT *permute, const V_ELE *vec_src, V_ELE *vec_dst, CG_UINT nr);

#endif // __MATRIX_H_
