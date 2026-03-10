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
#include "util.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "cuda/cuda_kernels.h"
#endif

static void initVectors(Matrix *m, CG_FLOAT *x, CG_FLOAT *b, CG_FLOAT *xexact)
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
  CG_UINT C             = m->C;
  CG_UINT *chunkPtr     = m->chunkPtr;
  CG_UINT *chunkLens    = m->chunkLens;
  CG_UINT *colInd       = m->colInd;
  CG_FLOAT *val         = m->val;
  CG_UINT *oldToNewPerm = m->oldToNewPerm;

  for (int rowID = 0; rowID < numRows; rowID++) {
    x[rowID] = 0.0;

    // Map original row to new row position in SCS format
    CG_UINT newRow     = oldToNewPerm[rowID];
    CG_UINT chunkIdx   = newRow / C;
    CG_UINT chunkRow   = newRow % C;
    CG_UINT chunkStart = chunkPtr[chunkIdx];
    CG_UINT rowLen     = chunkLens[chunkIdx];

    // Count actual non-zero values in this row
    int nnzrow = 0;
    for (CG_UINT j = 0; j < rowLen; ++j) {
      CG_UINT idx = chunkStart + j * C + chunkRow;
      if (val[idx] != 0.0) {
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

void solverCheckResidual(CommType *c, CG_FLOAT *x, CG_FLOAT *xexact, CG_UINT n)
{
  if (xexact == NULL) {
    return;
  }

  CG_FLOAT residual = 0.0;
  CG_FLOAT *v1      = x;
  CG_FLOAT *v2      = xexact;

  for (int i = 0; i < n; i++) {
    double diff = fabs(v1[i] - v2[i]);
    if (diff > residual)
      residual = diff;
  }

  commReduction(&residual, MAX);

  if (commIsMaster(c)) {
    printf("Difference between computed and exact  = %f\n", residual);
  }
}

int solveCG(CommType *comm, Parameter *param, Matrix *A)
{
  CG_FLOAT eps      = (CG_FLOAT)param->eps;
  int itermax       = param->itermax;

  CG_UINT nrow_base = A->nr;
  CG_UINT ncol_base = A->nc;

  CG_FLOAT *r_base  = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));
  CG_FLOAT *p_base  = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, ncol_base * sizeof(CG_FLOAT));
#ifdef SCS
  CG_FLOAT *Ap_base =
      (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, A->nrPadded * sizeof(CG_FLOAT));
#else
  CG_FLOAT *Ap_base = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));
#endif
  CG_FLOAT *x_base = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));
  CG_FLOAT *b_base = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));
  CG_FLOAT *xexact_base = NULL;

  if (strcmp(param->filename, "generate") == 0 ||
      strcmp(param->filename, "generate7P") == 0) {
    xexact_base = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));
  }
  
  initVectors(A, x_base, b_base, xexact_base);

  // Permute colInd and vectors to SCS ordering so no per-iteration
  // permute_vector is needed inside the CG loop.
#ifdef SCS
  CG_UINT *oldToNewPerm = A->oldToNewPerm;
  CG_UINT *newToOldPerm = A->newToOldPerm;
  CG_UINT *colInd_scs   = A->colInd;
  CG_UINT nElems_scs    = A->nElems;

  // Permute b, x (and xexact) from original to SCS ordering
  CG_FLOAT *perm_tmp =
      (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, nrow_base * sizeof(CG_FLOAT));

  permute_vector(oldToNewPerm, b_base, perm_tmp, nrow_base);
  memcpy(b_base, perm_tmp, nrow_base * sizeof(CG_FLOAT));

  permute_vector(oldToNewPerm, x_base, perm_tmp, nrow_base);
  memcpy(x_base, perm_tmp, nrow_base * sizeof(CG_FLOAT));

  if (xexact_base != NULL) {
    permute_vector(oldToNewPerm, xexact_base, perm_tmp, nrow_base);
    memcpy(xexact_base, perm_tmp, nrow_base * sizeof(CG_FLOAT));
  }
#endif

  CG_FLOAT normr  = 0.0;
  CG_FLOAT rtrans = 0.0, oldrtrans = 0.0;

  int printFreq = itermax / 10;
  if (printFreq > 50) {
    printFreq = 50;
  }
  if (printFreq < 1) {
    printFreq = 1;
  }
  double timeStart, timeStop, ts;

  CG_UINT nrow     = nrow_base;
  CG_FLOAT *r      = r_base;
  CG_FLOAT *p      = p_base;
  CG_FLOAT *Ap     = Ap_base;
  CG_FLOAT *x      = x_base;
  CG_FLOAT *b      = b_base;
  CG_FLOAT *xexact = xexact_base;

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, x, 0.0, x, p));
  PROFILE(COMM, commExchange(comm, A->nr, p));

  PROFILE(SPMVM, gpu_spMVM(A, p, Ap));
  PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, b, -1.0, Ap, r));
  PROFILE(DDOT, gpu_ddot_sync(nrow, r, r, &rtrans));
#else
  PROFILE(WAXPBY, waxpby(nrow, 1.0, x, 0.0, x, p));
  PROFILE(COMM, commExchange(comm, A->nr, p));

  PROFILE(SPMVM, spMVM(A, p, Ap));
  PROFILE(WAXPBY, waxpby(nrow, 1.0, b, -1.0, Ap, r));
  PROFILE(DDOT, ddot(nrow, r, r, &rtrans));
#endif

  normr = sqrt(rtrans);
  if (commIsMaster(comm)) {
    printf("Initial Residual = %E\n", normr);
  }

  int k;
  timeStart = getTimeStamp();
  for (k = 1; k < itermax && normr > eps; k++) {
    if (k == 1) {
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
      PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, r, 0.0, r, p));
#else
      PROFILE(WAXPBY, waxpby(nrow, 1.0, r, 0.0, r, p));
#endif
    } else {
      oldrtrans = rtrans;
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
      PROFILE(DDOT, gpu_ddot_sync(nrow, r, r, &rtrans));
      double beta = rtrans / oldrtrans;
      PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, r, beta, p, p));
#else
      PROFILE(DDOT, ddot(nrow, r, r, &rtrans));
      double beta = rtrans / oldrtrans;
      PROFILE(WAXPBY, waxpby(nrow, 1.0, r, beta, p, p));
#endif
    }
    normr = sqrt(rtrans);

    if (commIsMaster(comm) && (k % printFreq == 0 || k + 1 == itermax)) {
      printf("Iteration = %d Residual = %E\n", k, normr);
    }

    PROFILE(COMM, commExchange(comm, A->nr, p));

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
    PROFILE(SPMVM, gpu_spMVM(A, p, Ap));

    CG_FLOAT alpha = 0.0;
    PROFILE(DDOT, gpu_ddot_sync(nrow, p, Ap, &alpha));
    alpha = rtrans / alpha;
    PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, x, alpha, p, x));
    PROFILE(WAXPBY, gpu_waxpby_sync(nrow, 1.0, r, -alpha, Ap, r));
#else
    PROFILE(SPMVM, spMVM(A, p, Ap));

    CG_FLOAT alpha = 0.0;
    PROFILE(DDOT, ddot(nrow, p, Ap, &alpha));
    alpha = rtrans / alpha;
    PROFILE(WAXPBY, waxpby(nrow, 1.0, x, alpha, p, x));
    PROFILE(WAXPBY, waxpby(nrow, 1.0, r, -alpha, Ap, r));
#endif
  }
  timeStop = getTimeStamp();

  if (commIsMaster(comm)) {
    printf("Solution performed %d iterations and took %.2fs\n", k, timeStop - timeStart);
  }

#ifdef SCS

  permute_vector(newToOldPerm, x, perm_tmp, nrow);
  memcpy(x, perm_tmp, nrow * sizeof(CG_FLOAT));

  if (xexact != NULL) {
    permute_vector(newToOldPerm, xexact, perm_tmp, nrow);
    memcpy(xexact, perm_tmp, nrow * sizeof(CG_FLOAT));
  }
  deallocate(perm_tmp);
#endif

  return k;
}
