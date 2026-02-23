/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __SCSMATRIX_H_
#define __SCSMATRIX_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "util.h"

typedef struct {
  CG_UINT nr, nc, nnz; // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *colInd; // colum Indices
  CG_FLOAT *val; // matrix entries
  CG_UINT C, sigma; // chunk height and sorting scope
  CG_UINT nrPadded,
      nChunks; // number of rows with SCS padding, number of chunks
  CG_UINT nElems; // total number of elements (nnz + padding elements)
  CG_UINT *chunkPtr; // chunk pointers
  CG_UINT *chunkLens; // lengths of chunks
  CG_UINT *oldToNewPerm; // permutations for rows (and cols)
  CG_UINT *newToOldPerm; // inverse permutations for rows (and cols)
} Matrix;

typedef struct {
  int index;
  int count;
} SellCSigmaPair;

#endif // __SCSMATRIX_H_
