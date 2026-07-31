/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "comm.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "vtype.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "cuda/cuda_kernels.h"
#endif

static void initVectors(Matrix *m, V_ELE *x, V_ELE *b, V_ELE *xexact)
{
#ifdef CRS
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;

  for (int rowID = 0; rowID < numRows; rowID++) {

    int nnzrow = rowPtr[rowID + 1] - rowPtr[rowID];
    x[rowID]   = 0.0;

    if (xexact != NULL) {
      b[rowID]      = 27.0 - ((CG_FLOAT)(nnzrow - 1));
      xexact[rowID] = 1.0;
    } else {
      b[rowID] = 1.0;
    }
  }
#elif SCS
  CG_UINT numRows       = m->nr;
  CG_UINT c             = m->C;
  CG_UINT *chunkPtr     = m->chunkPtr;
  CG_UINT *chunkLens    = m->chunkLens;
  CG_UINT *colInd       = m->colInd;
  V_ELE *val            = m->val;
  CG_UINT *oldToNewPerm = m->oldToNewPerm;

  for (int rowID = 0; rowID < numRows; rowID++) {
    x[rowID] = 0.0;

    // Map original row to new row position in SCS format
    CG_UINT newRow     = oldToNewPerm[rowID];
    CG_UINT chunkIdx   = newRow / c;
    CG_UINT chunkRow   = newRow % c;
    CG_UINT chunkStart = chunkPtr[chunkIdx];
    CG_UINT rowLen     = chunkLens[chunkIdx];

    // Count actual non-zero values in this row
    int nnzrow = 0;
    for (CG_UINT j = 0; j < rowLen; ++j) {
      CG_UINT idx = chunkStart + j * c + chunkRow;
#ifdef USE_COMPLEX
      if (VREAL(val[idx]) != 0.0 || VIMAG(val[idx]) != 0.0) {
#else
      if (val[idx] != 0.0) {
#endif
        nnzrow++;
      }
    }

    if (xexact != NULL) {
      b[rowID]      = 27.0 - ((CG_FLOAT)(nnzrow - 1));
      xexact[rowID] = 1.0;
    } else {
      b[rowID] = 1.0;
    }
  }
#endif
}

/* End-to-end sanity check of the solution. Cheap, and the only thing that
 * catches a halo contribution that silently went missing - which is exactly the
 * failure mode of a broken communication/computation overlap. */
static void solverCheckResidual(CommType *c, V_ELE *x, V_ELE *xexact, CG_UINT n)
{
  if (xexact == NULL) {
    return;
  }

  CG_FLOAT residual = 0.0;
  V_ELE *v1         = x;
  V_ELE *v2         = xexact;

  for (CG_UINT i = 0; i < n; i++) {
#ifdef USE_COMPLEX
    double diff = VABS(v1[i] - v2[i]);
#else
    double diff = fabs(v1[i] - v2[i]);
#endif
    if (diff > residual) {
      residual = diff;
    }
  }

  commReduction(&residual, MAX);

  if (commIsMaster(c)) {
    printf("Difference between computed and exact  = %E\n", residual);
  }
}

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#define WAXBYFUNC gpu_waxpby_sync
#define SPMVMFUNC gpu_spMVM
#define DDOTFUNC gpu_ddot_sync
#else
#define WAXBYFUNC waxpby
#define SPMVMFUNC spMVM
#define DDOTFUNC ddot
#endif

#ifdef USE_COMPLEX
#define CAST(v) VREAL((v))
#else
#define CAST(v) v
#endif

/* USE_OVERLAP_SPMVM and OVERLAP_NUDGE_CHUNKS come from solver.h so that the
 * profiler configuration in main.c always matches the path taken here. */

#ifdef USE_OVERLAP_SPMVM
/**
 * @brief ap = A * p with the halo exchange hidden behind the local SpMV.
 *
 * Posts the exchange, then computes the part of the SpMV that only reads local
 * vector entries. That local phase is walked in OVERLAP_NUDGE_CHUNKS row chunks
 * with an MPI_Test in between: most MPI implementations make no progress on a
 * non-blocking collective unless the application re-enters the library, so
 * without those calls the whole transfer would happen inside commExchangeEnd
 * and nothing would actually be overlapped.
 *
 * COMM_WAIT therefore measures the communication that could NOT be hidden.
 */
static void applyA(CommType *comm, Matrix *A, V_ELE *p, V_ELE *ap)
{
  double ts;
  MPI_Request req;
  const CG_UINT numRows = A->nr;

  PROFILE(COMM, commExchangeBegin(comm, numRows, p, &req));

  const CG_UINT chunks = (OVERLAP_NUDGE_CHUNKS > 1) ? (CG_UINT)OVERLAP_NUDGE_CHUNKS : 1;
  const CG_UINT chunkSize = (numRows + chunks - 1) / chunks;

  for (CG_UINT rowStart = 0; rowStart < numRows; rowStart += chunkSize) {
    CG_UINT rowEnd = rowStart + chunkSize;
    if (rowEnd > numRows) {
      rowEnd = numRows;
    }

    PROFILE(SPMVM_LOCAL, spMVM_local_range(A, p, ap, rowStart, rowEnd));
    PROFILE(COMM, commExchangeTest(&req));
  }

  PROFILE(COMM_WAIT, commExchangeEnd(&req));
  PROFILE(SPMVM_EXT, spMVM_external(A, p, ap));
}
#else
/**
 * @brief ap = A * p with a blocking halo exchange followed by a full SpMV.
 */
static void applyA(CommType *comm, Matrix *A, V_ELE *p, V_ELE *ap)
{
  double ts;

  PROFILE(COMM, commExchange(comm, A->nr, p));
  PROFILE(SPMVM, SPMVMFUNC(A, p, ap));
}
#endif

int solveCG(CommType *comm, Parameter *param, Matrix *A)
{
  CG_FLOAT eps = (CG_FLOAT)param->eps;
  int itermax  = param->itermax;

  CG_UINT nrow = A->nr;
  CG_UINT ncol = A->nc;
  V_ELE *r     = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
  V_ELE *p     = (V_ELE *)allocate(ARRAY_ALIGNMENT, ncol * sizeof(V_ELE));
#ifdef SCS
  V_ELE *ap = (V_ELE *)allocate(ARRAY_ALIGNMENT, A->nrPadded * sizeof(V_ELE));
#else
  V_ELE *ap = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
#endif
  V_ELE *x      = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
  V_ELE *b      = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
  V_ELE *xexact = NULL;

  if (strcmp(param->filename, "generate") == 0 ||
      strcmp(param->filename, "generate7P") == 0) {
    xexact = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
  }

  initVectors(A, x, b, xexact);

  // Permute colInd and vectors to SCS ordering so no per-iteration
  // permute_vector is needed inside the CG loop.
#ifdef SCS
  CG_UINT *oldToNewPerm = A->oldToNewPerm;
  CG_UINT *newToOldPerm = A->newToOldPerm;
  CG_UINT *colIndScs    = A->colInd;
  CG_UINT nElemsScs     = A->nElems;

  // Permute b, x (and xexact) from original to SCS ordering
  V_ELE *permTmp = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));

  permute_vector(oldToNewPerm, b, permTmp, nrow);
  memcpy(b, permTmp, nrow * sizeof(V_ELE));

  permute_vector(oldToNewPerm, x, permTmp, nrow);
  memcpy(x, permTmp, nrow * sizeof(V_ELE));

  if (xexact != NULL) {
    permute_vector(oldToNewPerm, xexact, permTmp, nrow);
    memcpy(xexact, permTmp, nrow * sizeof(V_ELE));
  }
#endif

  CG_FLOAT normr  = 0.0;
  V_ELE rtrans    = 0.0;
  V_ELE oldrtrans = 0.0;

  int printFreq   = itermax / 10;
  if (printFreq > 50) {
    printFreq = 50;
  }
  if (printFreq < 1) {
    printFreq = 1;
  }
  double timeStart, timeStop, ts;

  PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, x, 0.0, x, p));
  applyA(comm, A, p, ap);
  PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, b, -1.0, ap, r));
  PROFILE(DDOT, DDOTFUNC(nrow, r, r, &rtrans));

  normr = sqrt(CAST(rtrans));
  if (commIsMaster(comm)) {
    printf("Initial Residual = %E\n", normr);
  }

  int k;
  timeStart = getTimeStamp();
  for (k = 1; k < itermax && normr > eps; k++) {
    if (k == 1) {
      PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, r, 0.0, r, p));
    } else {
      oldrtrans = rtrans;
      PROFILE(DDOT, DDOTFUNC(nrow, r, r, &rtrans));
      V_ELE beta = rtrans / oldrtrans;
      PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, r, beta, p, p));
    }
    normr = sqrt(CAST(rtrans));

    if (commIsMaster(comm) && (k % printFreq == 0 || k + 1 == itermax)) {
      printf("Iteration = %d Residual = %E\n", k, normr);
    }

    applyA(comm, A, p, ap);

    V_ELE alpha = 0.0;
    PROFILE(DDOT, DDOTFUNC(nrow, p, ap, &alpha));
    alpha = rtrans / alpha;
    PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, x, alpha, p, x));
    PROFILE(WAXPBY, WAXBYFUNC(nrow, 1.0, r, -alpha, ap, r));
  }
  timeStop = getTimeStamp();

  if (commIsMaster(comm)) {
    printf("Solution performed %d iterations and took %.2fs\n", k, timeStop - timeStart);
  }

#ifdef SCS
  permute_vector(newToOldPerm, x, permTmp, nrow);
  memcpy(x, permTmp, nrow * sizeof(V_ELE));

  if (xexact != NULL) {
    permute_vector(newToOldPerm, xexact, permTmp, nrow);
    memcpy(xexact, permTmp, nrow * sizeof(V_ELE));
  }
  deallocate(permTmp);
#endif

  solverCheckResidual(comm, x, xexact, nrow);

  return k;
}
