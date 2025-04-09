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
  char* matrixFormat; 
  CG_UINT nr, nc, nnz;        // ALL: number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz;  // ALL: number of total rows and non zeros
  CG_UINT startRow, stopRow;  // ALL: range of rows owned by current rank
  CG_UINT *colInd;            // ALL: colum Indices
  CG_FLOAT* val;              // ALL: matrix entries

  CG_UINT *rowPtr;            // CRS: row Pointer

  CG_UINT C, sigma;           // SCS: chunk height and sorting scope
  CG_UINT nrPadded, nChunks;  // SCS: number of rows with SCS padding, number of chunks
  CG_UINT nElems;             // SCS: total number of elements (nnz + padding elements) 
  CG_UINT *chunkPtr;          // SCS: chunk pointers
  CG_UINT *chunkLens;         // SCS: lengths of chunks
  CG_UINT *oldToNewPerm;      // SCS: permutations for rows (and cols)
  CG_UINT *newToOldPerm;      // SCS: inverse permutations for rows (and cols)
} Matrix;

// typedef struct {
//   CG_UINT c, sigma;           // chunk height and sorting scope
//   CG_UINT nr, nc, nnz;        // number of rows, columns and non zeros
//   // CG_UINT nrPadded, nChunks;  // number of rows with SCS padding, number of chunks
//   // CG_UINT nElems;             // number of SCS elements (nnz + padding elements) 
//   // CG_UINT totalNr, totalNnz;  // number of total rows and non zeros
//   // CG_UINT startRow, stopRow;  // range of rows owned by current rank
//   // CG_UINT *colInd, *chunkPtr; // colum Indices and chunk Pointers
//   // CG_UINT *chunkLens;         // lengths of chunks
//   // CG_UINT *oldToNewPerm;      // permutations for rows (and cols)
//   // CG_UINT *newToOldPerm;      // inverse permutations for rows (and cols)
//   // CG_FLOAT* val;              // value of matrix entries
// } SellCSigmaMatrix;

typedef struct {
  int index;
  int count;
} SellCSigmaPair;

extern void dumpMMMatrix(MmMatrix* m);
extern void dumpSCSMatrix(Matrix* m);
extern void dumpSCSMatrixToFile(Matrix* m, FILE* file);
extern void dumpVectorToFile(CG_FLOAT* y, int size, FILE* file);
extern void matrixRead(MmMatrix* m, char* filename);
extern void matrixConvertMMtoCRS(MmMatrix* mm, Matrix* m, int rank, int size);
extern void matrixConvertMMtoSCS(MmMatrix* mm, Matrix* m, int rank, int size);

extern void matrixGenerate(
    Matrix* m, Parameter* p, int rank, int size, bool use_7pt_stencil);

#endif // __MATRIX_H_
