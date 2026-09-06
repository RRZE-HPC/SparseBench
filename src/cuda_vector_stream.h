/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CUDA_VECTOR_STREAM_H_
#define __CUDA_VECTOR_STREAM_H_

/* Internal header for the ChebFD search-space streaming (cuda_vector_stream.cu
 * and the per-format launchers in cuda_spmv_{scs,crs}.cu). GPU builds only:
 * includes gpu_backend.h, so it must never be included from a gcc-compiled
 * translation unit (use the opaque declarations in cuda_kernels.h there).
 *
 * Placement: the matrix is device-resident (managed + prefetched, see
 * gpu_matrix_prefetch). The dense search-space blocks Y and AY are
 * host-resident in pinned memory and are streamed through the device in
 * two shapes, both double-buffered on a copy stream against a compute
 * stream:
 *
 *   column sub-blocks (nb columns, all rows)  — the matrix passes: filter
 *       recurrence (block-outer: every degree on one sub-block before the
 *       next sub-block is fetched, so Y crosses the link once in and once
 *       out per filter) and AY = A*Y.
 *   row chunks (all columns, ~chunkBytes rows) — the dense passes: Gram
 *       matrices Y^T Y / Y^T AY, the block update Y <- Y B and the Ritz
 *       residuals; row-major blocks make these contiguous copies. */

#include "gpu_backend.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Identity view of the whole resident matrix, handed to the per-format
 * kernel launchers. ptr/lens hold absolute element ids. */
typedef struct {
  const V_ELE *val;
  const CG_UINT *colInd;
  const CG_UINT *lens; /* SCS only, NULL for CRS */
  const CG_UINT *ptr;
  CG_UINT elemBase; /* 0 for the whole matrix */
  CG_UINT rowBase;  /* 0 for the whole matrix */
  CG_UINT count;    /* nChunks (SCS) / nr (CRS) */
} GpuPartView;

/* Per-format launchers (cuda_spmv_{scs,crs}.cu): one fused-ChebFD / SpMMV
 * kernel over the whole matrix on `stream`, over `width` columns of
 * row-major blocks with leading dimension `ld`. Async; no sync.
 *   y = cA*(A x) + cP*p + cQ*q + cR*r  (q, r optional), acc += gc*y (optional) */
void gpu_launch_chebfd(const Matrix *m,
    gpuStream_t stream,
    const V_ELE *x,
    V_ELE cA,
    const V_ELE *p,
    V_ELE cP,
    const V_ELE *q,
    V_ELE cQ,
    V_ELE *y,
    V_ELE gc,
    V_ELE *acc,
    const V_ELE *r,
    V_ELE cR,
    CG_UINT width,
    CG_UINT ld);

void gpu_launch_spmmv(const Matrix *m,
    gpuStream_t stream,
    const V_ELE *x,
    V_ELE *y,
    CG_UINT width,
    CG_UINT ld);

#define GPU_VSTREAM_NSLOT 2

typedef struct GpuVectorStream {
  const Matrix *A;
  CG_UINT nr;        /* matrix rows */
  CG_UINT vecRows;   /* block rows incl. SCS padding */
  int NS;            /* allocation width of the host blocks */
  int nb;            /* columns per streamed sub-block */
  CG_UINT chunkRows; /* rows per streamed row chunk (multiple of 16) */

  /* Column sub-block buffers, all vecRows x nb with ld = nb. X is
   * double-buffered (one slot computes while the other is copied). U / W
   * are the recurrence scratch (T_{n-2}, T_{n-1}) of the sub-block being
   * computed; compute is serialized on computeStream, so one pair serves
   * both slots. The A*Y pass uses U / W alternately as its output slots. */
  V_ELE *X[GPU_VSTREAM_NSLOT], *U, *W;

  /* Row-chunk slots: two input chunks (chunkRows x NS) and one output. */
  V_ELE *rA[GPU_VSTREAM_NSLOT], *rB[GPU_VSTREAM_NSLOT], *rO[GPU_VSTREAM_NSLOT];

  /* Dense scratch (device). */
  double *partial;   /* Gram / residual partial sums */
  size_t partialCap; /* elements */
  double *G;         /* NS x NS accumulator (Gram results) */
  double *B;         /* NS x NS (update matrix / eigenvectors) */
  double *eval;      /* NS */
  int *sel;          /* NS selected Ritz pairs */
  double *res2;      /* NS squared residual norms */

  gpuStream_t computeStream, copyStream;
  gpuEvent_t computeDone[GPU_VSTREAM_NSLOT], copyDone[GPU_VSTREAM_NSLOT];

  /* Statistics over the lifetime of the context. */
  size_t h2dBytes, d2hBytes;
  unsigned long long colPasses, rowPasses;
} GpuVectorStream;

#ifdef __cplusplus
}
#endif

#endif /* __CUDA_VECTOR_STREAM_H_ */
