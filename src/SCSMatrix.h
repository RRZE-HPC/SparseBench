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
#include "vtype.h"

typedef struct {
  CG_UINT nr, nc, nnz;       // number of rows, columns and non zeros
  CG_UINT totalNr, totalNnz; // number of total rows and non zeros
  CG_UINT startRow, stopRow; // range of rows owned by current rank
  CG_UINT *colInd;           // column Indices
  V_ELE *val;                // matrix entries
  CG_UINT C, sigma;          // chunk height and sorting scope
  CG_UINT nrPadded;          // number of rows with SCS padding,
  CG_UINT nChunks;           // number of chunks
  CG_UINT nElems;            // total number of elements (nnz + padding elements)
  CG_UINT *chunkPtr;         // chunk pointers
  CG_UINT *chunkLens;        // lengths of chunks
  CG_UINT *oldToNewPerm;     // permutations for rows (and cols)
  CG_UINT *newToOldPerm;     // inverse permutations for rows (and cols)
  double beta;
} Matrix;

typedef struct {
  int index;
  int count;
} SellCSigmaPair;

/* spMMVM (matrix-SCS.c) puts a per-thread `V_ELE tmp[C * blockwidth]` VLA on the
 * OpenMP worker stack, so callers must check the block width fits beforehand. */
#ifndef SCS_MAX_SPMMVM_VLA_BYTES
#define SCS_MAX_SPMMVM_VLA_BYTES (2u * 1024u * 1024u)
#endif

static inline int spMMVMBlockWidthOk(CG_UINT C, int blockwidth)
{
  if (blockwidth < 1) {
    return 0; /* a zero/negative width is a zero-length VLA (UB) or a huge nc */
  }
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  /* The device kernels accumulate in registers and tile the block over the
   * grid, so there is no per-thread scratch to overflow — the limit is purely
   * a host-stack artifact. Every call site guarded by this check dispatches to
   * the device kernels in a GPU build (see kernel_dispatch.h). */
  (void)C;
  return 1;
#else
  return (size_t)C * (size_t)blockwidth * sizeof(V_ELE) <= SCS_MAX_SPMMVM_VLA_BYTES;
#endif
}

#endif // __SCSMATRIX_H_
