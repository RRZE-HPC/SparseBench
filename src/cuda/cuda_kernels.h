/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __GPU_KERNELS_H_
#define __GPU_KERNELS_H_

#include "../matrix.h"
#include "../util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Low-level kernel launchers (async — no sync, no managed memory) */
void gpu_waxpby(CG_UINT n,
    CG_FLOAT alpha,
    const CG_FLOAT *x,
    CG_FLOAT beta,
    const CG_FLOAT *y,
    CG_FLOAT *w);

void gpu_ddot(CG_UINT n, const CG_FLOAT *x, const CG_FLOAT *y, CG_FLOAT *result);

void gpu_spmv_scs(CG_UINT nChunks,
    CG_UINT C,
    const CG_UINT *chunkPtr,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const CG_FLOAT *val,
    const CG_FLOAT *x,
    CG_FLOAT *y);

void gpu_spmv_crs(CG_UINT numRows,
    const CG_UINT *rowPtr,
    const CG_UINT *colInd,
    const CG_FLOAT *val,
    const CG_FLOAT *x,
    CG_FLOAT *y);

/*
 * High-level wrappers — mirror the CPU interface from solver.h.
 * These use managed memory for internal allocations and call
 * DeviceSynchronize after each kernel so they behave synchronously
 * like the CPU versions.
 */
void gpu_spMVM(Matrix *m, const CG_FLOAT *x, CG_FLOAT *y);
void gpu_spMMVM(Matrix *m, const DMatrix *x, DMatrix *y);

void gpu_waxpby_sync(CG_UINT n,
    CG_FLOAT alpha,
    const CG_FLOAT *x,
    CG_FLOAT beta,
    const CG_FLOAT *y,
    CG_FLOAT *w);

void gpu_ddot_sync(CG_UINT n, const CG_FLOAT *x, const CG_FLOAT *y, CG_FLOAT *result);

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
