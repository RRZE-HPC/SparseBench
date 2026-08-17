/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __GPU_KERNELS_H_
#define __GPU_KERNELS_H_

#include "matrix.h"
#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Low-level kernel launchers (async — no sync, no managed memory) */
void gpu_waxpby(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w);

void gpu_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result);

void gpu_spmv_scs(CG_UINT nChunks,
    CG_UINT C,
    const CG_UINT *chunkPtr,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y);

void gpu_spmv_crs(CG_UINT numRows,
    const CG_UINT *rowPtr,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y);

/*
 * High-level wrappers — mirror the CPU interface from solver.h.
 * These use managed memory for internal allocations and call
 * DeviceSynchronize after each kernel so they behave synchronously
 * like the CPU versions.
 */
void gpu_spMVM(Matrix *m, const V_ELE *x, V_ELE *y);
void gpu_spMMVM(Matrix *m, const DMatrix *x, DMatrix *y);

/* Fused block kernels backing the ChebFD recurrence. Both are served by one
 * device kernel per format: spMMVMFused is the accumulator-free form, and
 * chebfdOp additionally folds x += gc*y into the same pass. */
void gpu_spMMVMFused(Matrix *m,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y);

void gpu_chebfdOp(Matrix *m,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x);

void gpu_waxpby_sync(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w);

void gpu_waxpby3(CG_UINT n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w);

void gpu_waxpby3_sync(CG_UINT n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w);

void gpu_ddot_sync(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result);

/* Managed-memory allocation helpers */
void *gpu_allocate_managed(size_t bytes);
void gpu_free_managed(void *ptr);

/* Device init / finalize */
void gpu_init(int device);
void gpu_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* __GPU_KERNELS_H_ */
