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

/* cudaMemPrefetchAsync + sync of the whole matrix (val, colInd, ptr and,
 * for SCS, chunkLens) to the current device, so a managed matrix is
 * device-resident from the first pass instead of migrating page by page on
 * first touch inside the timed region. No-op for non-managed buffers. */
void gpu_matrix_prefetch(const Matrix *m);

/* Search-space streaming for ChebFD (cuda_vector_stream.cu): the matrix is
 * device-resident, the dense blocks Y / AY are pinned host memory
 * (gpu_allocate_host) and are streamed through the device — column
 * sub-blocks of nb columns for the matrix passes, row chunks of ~chunkBytes
 * (0 = default 128 MiB) for the dense passes — double-buffered against
 * compute. All calls return with the device drained. This header stays
 * compilable by plain gcc, hence the opaque struct. */
typedef struct GpuVectorStream GpuVectorStream;

GpuVectorStream *gpu_vstream_init(
    const Matrix *A, int NS, int nb, size_t chunkBytes, int verbose);

void gpu_vstream_free(GpuVectorStream *s);

/* Y (vecRows x nc, row-major, pinned) <- p(A) Y: the full Chebyshev
 * recurrence of degree Np with affine map (alpha, beta) and coefficients
 * gc[0..Np], applied block-outer (all degrees on one column sub-block). */
void gpu_vstream_filter(GpuVectorStream *s,
    V_ELE *Yh,
    int nc,
    double alpha,
    double beta,
    const double *gc,
    int Np);

/* AY = A * Y over nc columns. */
void gpu_vstream_spmmv(GpuVectorStream *s, const V_ELE *Yh, V_ELE *AYh, int nc);

/* G (m x m, host) = A^T B for two vecRows x m blocks (pass Bh == Ah or
 * NULL for A^T A). Exactly symmetric. */
void gpu_vstream_gram(GpuVectorStream *s, const V_ELE *Ah, const V_ELE *Bh, int m, double *Gh);

/* Y (stride m) <- Y * B with B m x mOut row-major (host); the result is
 * written back in place at stride mOut (<= m). */
void gpu_vstream_update(GpuVectorStream *s, V_ELE *Yh, int m, const double *Bh, int mOut);

/* res2[t] = || AY e_k - eval[k] Y e_k ||^2 for k = sel[t], t < nsel, with
 * e_k = evec[:, k] (evec m x m row-major as produced by jacobiEigen). */
void gpu_vstream_ritzResiduals(GpuVectorStream *s,
    const V_ELE *Yh,
    const V_ELE *AYh,
    int m,
    const double *eval,
    const double *evec,
    const int *sel,
    int nsel,
    double *res2);

void gpu_vstream_stats(const GpuVectorStream *s,
    size_t *h2dBytes,
    size_t *d2hBytes,
    unsigned long long *colPasses,
    unsigned long long *rowPasses);

/* Pinned host memory (cudaMallocHost) for the streamed blocks. */
void *gpu_allocate_host(size_t bytes);
void gpu_free_host(void *p);

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

/* Device init / finalize */
void gpu_init(int device);
void gpu_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* __GPU_KERNELS_H_ */
