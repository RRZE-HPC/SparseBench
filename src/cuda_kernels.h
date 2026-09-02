/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __GPU_KERNELS_H_
#define __GPU_KERNELS_H_

#include "allocate.h"
#include "matrix.h"
#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Low-level kernel launchers (async — no sync, no managed memory) */
void gpu_waxpby(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w);

void gpu_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result);

// Whole dot product on the GPU
void gpu_ddot_device(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result_d);

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

/* Host-resident matrix streaming for ChebFD: the matrix stays where
 * allocate() put it and is streamed to the device in double-buffered parts
 * (machinery in cuda_matrix_stream.cu; the sweeps live next to the kernels
 * in cuda_spmv_{scs,crs}.cu). partBytes is a byte count — callers convert
 * the gpu_stream_mb knob. nb is the column-subblock width, 0 or >= width
 * meaning one full-width launch. This header stays compilable by plain gcc,
 * hence the opaque struct. */
typedef struct GpuMatrixStream GpuMatrixStream;

GpuMatrixStream *gpu_matrix_stream_init(const Matrix *m, size_t partBytes, int verbose);

void gpu_matrix_stream_free(GpuMatrixStream *s);

/* cudaMemPrefetchAsync + sync: one-time pull of a managed block to the
 * device (e.g. the ChebFD vector blocks before the first sweep). */
void gpu_matrix_stream_prefetch(const V_ELE *p, size_t bytes);

void gpu_matrix_stream_stats(const GpuMatrixStream *s,
    size_t *h2dBytes,
    int *nParts,
    unsigned long long *nSweeps,
    double *copyMs);

/* Sweeps over a streamed matrix (one pass of the polynomial recurrence). */
void gpu_stream_spMMVM(GpuMatrixStream *s, const DMatrix *x, DMatrix *y, int nb);

void gpu_stream_spMMVMFused(GpuMatrixStream *s,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    int nb);

void gpu_stream_chebfdOp(GpuMatrixStream *s,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x,
    int nb);

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

/* Dense ChebFD block steps (cuda_chebfd_dense.cu). Signatures mirror their
 * host counterparts in chebFDSolver.h. */
void gpu_gramYtAY(CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H);

void gpu_computeRitzResidual(DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double evalk,
    double *evec,
    int k,
    double *evk,
    V_ELE *avbuf);

int gpu_orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol);

/* Release the persistent scratch owned by cuda_chebfd_dense.cu. */
void gpu_chebfd_scratch_free(void);

/* Mode-selectable buffer allocation (AllocType from allocate.h). One mode
 * per run: set via gpu_set_alloc_type before the first gpu_allocate, and
 * the free functions dispatch on that same mode. Under ALLOC_EXPLICIT,
 * gpu_allocate places host-side buffers (cudaMallocHost) while
 * gpu_allocate_device places kernel-only buffers (cudaMalloc); other modes
 * hand out one backing store from both. Pair each allocation with its
 * matching free (gpu_free / gpu_free_device). */
void gpu_set_alloc_type(AllocType type);
void *gpu_allocate(size_t bytes);
void *gpu_allocate_device(size_t bytes);
void gpu_free(void *ptr);
void gpu_free_device(void *ptr);

/* Device-side ChebFD block initialization, so the vector blocks can be
 * kernel-only buffers (allocateDevice) under ALLOC_EXPLICIT.
 * gpu_randomInitBlock mirrors randomInitBlock in chebFDSolver.c (same
 * splitmix64 key layout; pass comm->rank * 0xD1B54A32D192ED03ull as
 * rankKey). gpu_zeroPadRows zeroes rows [startRow, startRow + numRows) of
 * a row-major block — the post-ortho SCS re-padding. */
void gpu_randomInitBlock(
    unsigned long long rankKey, CG_UINT nr, CG_UINT vecRows, V_ELE *e, int nv);

void gpu_zeroPadRows(V_ELE *e, CG_UINT startRow, CG_UINT numRows, CG_UINT stride);

/* Device init / finalize */
void gpu_init(int device);
void gpu_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* __GPU_KERNELS_H_ */
