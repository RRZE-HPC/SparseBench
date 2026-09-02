/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * GPU SpMV / SpMMV / fused-ChebFD kernels for the Sell-C-sigma (SCS) format.
 *
 * Launch shape for the block-vector (SpMMV) kernels
 * -------------------------------------------------
 * The block vectors are row-major with stride numVecs, so the *vector* index
 * is the contiguous one. Hence threadIdx.x indexes vectors, not rows:
 *
 *   grid  = (nChunks, ceil(numVecs / VEC_TILE))
 *   block = (VEC_TILE, ROW_TILE)          VEC_TILE = 32 (one warp)
 *
 * A warp then covers 32 consecutive vector components of one row, so the
 * gather x[col * numVecs + vec] is a single coalesced transaction, while
 * colInd/val are read once per warp and broadcast. Rows within a chunk are
 * walked by a strided loop over threadIdx.y, which decouples the block size
 * from C entirely.
 *
 * The obvious alternative — threadIdx.x = row-in-chunk, threadIdx.y = vec, as
 * this file used to do — caps numVecs at 1024/C (16 for the default C=64) and
 * makes every gather a scattered 8-byte access. ChebFD runs with numVecs in
 * the hundreds, so that shape both fails to launch and mis-uses the bus.
 *
 * Part and subblock views (matrix streaming / cheb_nb)
 * ----------------------------------------------------
 * Every kernel takes its matrix through a *part* view: nPartChunks
 * consecutive chunks described by sliced chunkPtr/chunkLens/colInd/val
 * arrays plus the scalars elemBase (= chunkPtr[c0], absolute) and rowBase
 * (= c0*C, global first row). chunkPtr keeps ABSOLUTE element ids, so the
 * kernels rebase with `chunkPtr[pc] - elemBase` to index the slices, while
 * colInd values stay global — the vector gather xin[colInd * ld + vec]
 * reaches any row, which is why a streamed column sub-block must always
 * carry all rows (cuda_vector_stream.cu).
 *
 * The block vectors are addressed with a separate leading dimension ld >=
 * numVecs, so a launch may cover a width-numVecs column slice at offset v0
 * of a wider row-major block (base pointer + v0), or a device sub-block
 * buffer with ld = nb. The plain wrappers pass the identity view (ld =
 * numVecs, bases 0); the _nb wrappers tile the columns; gpu_launch_* is
 * the stream-aware entry the vector streaming uses.
 */
#include <stdint.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "gpu_backend.h"

#include "cuda_vector_stream.h"
#include "nvtx_marker.h"

#define VEC_TILE 32
#define ROW_TILE 8


/* ------------------------------------------------------------------ */
/*  SCS SpMV:  y = A * x                                             */
/*                                                                    */
/*  Flat thread -> row mapping; chunk/lane are derived. Block size is */
/*  independent of C, unlike a one-block-per-chunk mapping which for  */
/*  the default C = 64 would run 2-warp blocks.                       */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmv_scs(CG_UINT nChunks,
    CG_UINT C,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *chunkPtr,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  CG_UINT row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nChunks * C)
    return;

  CG_UINT chunk  = row / C;
  CG_UINT lane   = row % C;
  CG_UINT offset = chunkPtr[chunk] - elemBase;
  CG_UINT len    = chunkLens[chunk];

  V_ELE tmp      = VCONST(0, 0);
  for (CG_UINT j = 0; j < len; j++) {
    CG_UINT idx = offset + j * C + lane;
    tmp += val[idx] * x[colInd[idx]];
  }

  y[rowBase + row] = tmp;
}

extern "C" void gpu_spmv_scs(CG_UINT nChunks,
    CG_UINT C,
    const CG_UINT *chunkPtr,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  int threads = 256;
  int blocks  = (int)((nChunks * C + threads - 1) / threads);
  kernel_spmv_scs<<<blocks, threads>>>(
      nChunks, C, 0, 0, chunkPtr, chunkLens, colInd, val, x, y);
}

/* ------------------------------------------------------------------ */
/*  SCS SpMMV:  Y = A * X   (block-vector, numVecs columns)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_scs(CG_UINT nPartChunks,
    CG_UINT C,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *chunkPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  CG_UINT vec = blockIdx.y * blockDim.x + threadIdx.x;
  /* Early return is safe: no shared memory and no __syncthreads below, so a
   * partially-populated block needs no participation from the idle lanes. */
  if (vec >= numVecs)
    return;

  CG_UINT pc     = blockIdx.x;
  CG_UINT offset = chunkPtr[pc] - elemBase;
  CG_UINT len    = chunkLens[pc];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE tmp = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      tmp += val[idx] * x[(size_t)colInd[idx] * ld + vec];
    }
    y[(size_t)(rowBase + pc * C + lane) * ld + vec] = tmp;
  }
}

/* ------------------------------------------------------------------ */
/*  Fused ChebFD kernel                                               */
/*                                                                    */
/*    y   = cA*(A*xin) + cP*p + cQ*q + cR*r   (q, r optional)         */
/*    acc = acc + gc*y                        (acc optional)          */
/*                                                                    */
/*  Covers every host entry point with one kernel:                    */
/*    spMMVMFused -> acc = NULL (no accumulate)                       */
/*    chebfdOp    -> p = xin, acc = the polynomial accumulator        */
/*    Clenshaw    -> r = the constant input x (cuda_vector_stream.cu) */
/*  All terms are row-local, so y may alias q or r, p may alias xin.  */
/* ------------------------------------------------------------------ */
__global__ void kernel_chebfd_scs(CG_UINT nPartChunks,
    CG_UINT C,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *chunkPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *xin,
    V_ELE cA,
    const V_ELE *p,
    V_ELE cP,
    const V_ELE *q,
    V_ELE cQ,
    V_ELE *y,
    V_ELE gc,
    V_ELE *acc,
    const V_ELE *r,
    V_ELE cR)
{
  CG_UINT vec = blockIdx.y * blockDim.x + threadIdx.x;
  if (vec >= numVecs)
    return;

  CG_UINT pc     = blockIdx.x;
  CG_UINT offset = chunkPtr[pc] - elemBase;
  CG_UINT len    = chunkLens[pc];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE tmp = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      tmp += val[idx] * xin[(size_t)colInd[idx] * ld + vec];
    }

    size_t e = (size_t)(rowBase + pc * C + lane) * ld + vec;
    V_ELE t  = cA * tmp + cP * p[e];
    if (q != NULL) {
      t += cQ * q[e];
    }
    if (r != NULL) {
      t += cR * r[e];
    }
    y[e] = t;
    if (acc != NULL) {
      acc[e] += gc * t;
    }
  }
}


/* Two adjacent block columns per thread (V_ELE2 = double2 / float2): same
 * bytes as kernel_chebfd_scs, half the L1/L2 requests. ncu on the scalar
 * kernel: L2 throughput 70 % > DRAM 62 %, gathers are 60 % of the L2
 * traffic — the request count, not the byte count, is the limiter.
 * Requires even numVecs / ld and 16-byte aligned column offsets (the
 * launcher checks); the per-element arithmetic order is unchanged. */
#ifndef USE_COMPLEX
#if PRECISION == 1
typedef float2 V_ELE2;
#else
typedef double2 V_ELE2;
#endif
__global__ void kernel_chebfd_scs_v2(CG_UINT nPartChunks,
    CG_UINT C,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *chunkPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *xin,
    V_ELE cA,
    const V_ELE *p,
    V_ELE cP,
    const V_ELE *q,
    V_ELE cQ,
    V_ELE *y,
    V_ELE gc,
    V_ELE *acc,
    const V_ELE *r,
    V_ELE cR)
{
  CG_UINT vec = 2u * (blockIdx.y * blockDim.x + threadIdx.x);
  if (vec >= numVecs)
    return;

  CG_UINT pc     = blockIdx.x;
  CG_UINT offset = chunkPtr[pc] - elemBase;
  CG_UINT len    = chunkLens[pc];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE t0 = VCONST(0, 0), t1 = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      V_ELE v     = val[idx];
      V_ELE2 xv   = *reinterpret_cast<const V_ELE2 *>(&xin[(size_t)colInd[idx] * ld + vec]);
      t0 += v * xv.x;
      t1 += v * xv.y;
    }

    size_t e  = (size_t)(rowBase + pc * C + lane) * ld + vec;
    V_ELE2 pv = *reinterpret_cast<const V_ELE2 *>(&p[e]);
    t0        = cA * t0 + cP * pv.x;
    t1        = cA * t1 + cP * pv.y;
    if (q != NULL) {
      V_ELE2 qv = *reinterpret_cast<const V_ELE2 *>(&q[e]);
      t0 += cQ * qv.x;
      t1 += cQ * qv.y;
    }
    if (r != NULL) {
      V_ELE2 rv = *reinterpret_cast<const V_ELE2 *>(&r[e]);
      t0 += cR * rv.x;
      t1 += cR * rv.y;
    }
    V_ELE2 yv;
    yv.x = t0;
    yv.y = t1;
    *reinterpret_cast<V_ELE2 *>(&y[e]) = yv;
    if (acc != NULL) {
      V_ELE2 av = *reinterpret_cast<V_ELE2 *>(&acc[e]);
      av.x += gc * t0;
      av.y += gc * t1;
      *reinterpret_cast<V_ELE2 *>(&acc[e]) = av;
    }
  }
}
#endif /* !USE_COMPLEX */


/* Four adjacent block columns per thread (V_ELE4 = double4 / float4; two
 * 16 B loads per operand). ~3 % faster than the column-pair kernel at the
 * same ~90 % L2 throughput; needs 32 B aligned operands and a launch shape
 * down to 4 vector lanes (vecTile) for 16-column sub-blocks. */
#ifndef USE_COMPLEX
#if PRECISION == 1
typedef float4 V_ELE4;
#else
typedef double4 V_ELE4;
#endif
__global__ void kernel_chebfd_scs_v4(CG_UINT nPartChunks,
    CG_UINT C,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *chunkPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *chunkLens,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *xin,
    V_ELE cA,
    const V_ELE *p,
    V_ELE cP,
    const V_ELE *q,
    V_ELE cQ,
    V_ELE *y,
    V_ELE gc,
    V_ELE *acc,
    const V_ELE *r,
    V_ELE cR)
{
  CG_UINT vec = 4u * (blockIdx.y * blockDim.x + threadIdx.x);
  if (vec >= numVecs)
    return;

  CG_UINT pc     = blockIdx.x;
  CG_UINT offset = chunkPtr[pc] - elemBase;
  CG_UINT len    = chunkLens[pc];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE t0 = VCONST(0, 0), t1 = VCONST(0, 0), t2 = VCONST(0, 0), t3 = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      V_ELE v     = val[idx];
      V_ELE4 xv   = *reinterpret_cast<const V_ELE4 *>(&xin[(size_t)colInd[idx] * ld + vec]);
      t0 += v * xv.x;
      t1 += v * xv.y;
      t2 += v * xv.z;
      t3 += v * xv.w;
    }

    size_t e  = (size_t)(rowBase + pc * C + lane) * ld + vec;
    V_ELE4 pv = *reinterpret_cast<const V_ELE4 *>(&p[e]);
    t0        = cA * t0 + cP * pv.x;
    t1        = cA * t1 + cP * pv.y;
    t2        = cA * t2 + cP * pv.z;
    t3        = cA * t3 + cP * pv.w;
    if (q != NULL) {
      V_ELE4 qv = *reinterpret_cast<const V_ELE4 *>(&q[e]);
      t0 += cQ * qv.x;
      t1 += cQ * qv.y;
      t2 += cQ * qv.z;
      t3 += cQ * qv.w;
    }
    if (r != NULL) {
      V_ELE4 rv = *reinterpret_cast<const V_ELE4 *>(&r[e]);
      t0 += cR * rv.x;
      t1 += cR * rv.y;
      t2 += cR * rv.z;
      t3 += cR * rv.w;
    }
    V_ELE4 yv;
    yv.x = t0;
    yv.y = t1;
    yv.z = t2;
    yv.w = t3;
    *reinterpret_cast<V_ELE4 *>(&y[e]) = yv;
    if (acc != NULL) {
      V_ELE4 av = *reinterpret_cast<V_ELE4 *>(&acc[e]);
      av.x += gc * t0;
      av.y += gc * t1;
      av.z += gc * t2;
      av.w += gc * t3;
      *reinterpret_cast<V_ELE4 *>(&acc[e]) = av;
    }
  }
}
#endif /* !USE_COMPLEX */

/* ------------------------------------------------------------------ */
/*  High-level wrappers matching the CPU interface in solver.h.        */
/*  Expect managed/device pointers (see allocate.c).                   */
/* ------------------------------------------------------------------ */

#ifdef SCS
static inline dim3 blockGrid(const Matrix *m, CG_UINT numVecs)
{
  return dim3((unsigned)m->nChunks, (unsigned)((numVecs + VEC_TILE - 1) / VEC_TILE));
}

/* Column-slice bases of one fused launch: +v0 offsets into the row-major
 * block, NULL passes through untouched (device-side NULL test). */
typedef struct {
  const V_ELE *x, *p, *q, *r;
  V_ELE *y, *acc;
  V_ELE cA, cP, cQ, gc, cR;
  CG_UINT width, ld, nb;
  CG_UINT C;
} ChebfdArgs;

/* Effective subblock width: 0 or >= width means "one full-width launch". */
static inline CG_UINT effNb(CG_UINT nb, CG_UINT width)
{
  return (nb == 0 || nb >= width) ? width : nb;
}

/* Vector lanes per block row for a launch over w columns: a full warp when
 * the slice is wide enough, otherwise 16 or 8 lanes so a warp spans 2 or 4
 * rows instead of idling half its threads (narrow streamed sub-blocks).
 * The block keeps LAUNCH_THREADS threads; the kernels derive everything
 * from blockDim, so only the launch shape changes. */
#define LAUNCH_THREADS (VEC_TILE * ROW_TILE)
static inline unsigned vecTile(CG_UINT w)
{
  return (w <= 4) ? 4u : (w <= 8) ? 8u : (w <= 16) ? 16u : (unsigned)VEC_TILE;
}

/* Fused kernel on one part (or an identity view of the whole matrix),
 * tiled into width-nb column slices. stream 0 = legacy default stream. */

/* Column-vector width of the fused kernel: 4 (default, kernel_chebfd_scs_v4),
 * 2 (kernel_chebfd_scs_v2) or 1 (scalar); the launcher falls back to the
 * next narrower kernel when the alignment conditions fail. Measured on
 * GH200 at 256^3 / nb 16: 5.65 ms (scalar) / 3.54 ms (2) / 3.44 ms (4) per
 * launch — both vector kernels sit at ~90 % L2 throughput, so the width is
 * a request-count lever with a small remaining margin. Env CHEBFD_VEC
 * overrides for experiments. */
static int g_chebfdVec = -1;
static void initChebfdVec(void)
{
  if (g_chebfdVec < 0) {
    const char *e = getenv("CHEBFD_VEC");
    g_chebfdVec   = (e != NULL) ? atoi(e) : 4;
  }
}
static void launchChebfdPart(const GpuPartView *v, gpuStream_t stream, void *ua)
{
  initChebfdVec();
  ChebfdArgs *a = (ChebfdArgs *)ua;
  CG_UINT nb    = effNb(a->nb, a->width);
  for (CG_UINT v0 = 0; v0 < a->width; v0 += nb) {
    CG_UINT w   = (nb < a->width - v0) ? nb : a->width - v0;
#ifndef USE_COMPLEX
    /* Column-pair kernel when every operand is 16-byte aligned: even width
     * and ld, even slice offset, and 16 B aligned base pointers. */
    int pairOk = (w % 2 == 0) && (a->ld % 2 == 0) && (v0 % 2 == 0) &&
                 (((uintptr_t)a->x | (uintptr_t)a->p | (uintptr_t)a->y |
                      (uintptr_t)(a->q ? a->q : a->x) | (uintptr_t)(a->r ? a->r : a->x) |
                      (uintptr_t)(a->acc ? a->acc : a->x)) %
                         16 ==
                     0);
    int quadOk = pairOk && (w % 4 == 0) && (a->ld % 4 == 0) && (v0 % 4 == 0) &&
                 (((uintptr_t)a->x | (uintptr_t)a->p | (uintptr_t)a->y |
                      (uintptr_t)(a->q ? a->q : a->x) | (uintptr_t)(a->r ? a->r : a->x) |
                      (uintptr_t)(a->acc ? a->acc : a->x)) %
                         32 ==
                     0);
    if (quadOk && g_chebfdVec == 4) {
      CG_UINT wq  = w / 4;
      unsigned vt = vecTile(wq);
      dim3 grid((unsigned)v->count, (unsigned)((wq + vt - 1) / vt));
      kernel_chebfd_scs_v4<<<grid, dim3(vt, LAUNCH_THREADS / vt), 0, stream>>>(v->count,
          a->C,
          w,
          a->ld,
          v->ptr,
          v->elemBase,
          v->rowBase,
          v->lens,
          v->colInd,
          v->val,
          a->x + v0,
          a->cA,
          a->p + v0,
          a->cP,
          (a->q != NULL) ? a->q + v0 : NULL,
          a->cQ,
          a->y + v0,
          a->gc,
          (a->acc != NULL) ? a->acc + v0 : NULL,
          (a->r != NULL) ? a->r + v0 : NULL,
          a->cR);
      continue;
    }
    if (pairOk && g_chebfdVec >= 2) {
      CG_UINT wp  = w / 2;
      unsigned vt = vecTile(wp);
      dim3 grid((unsigned)v->count, (unsigned)((wp + vt - 1) / vt));
      kernel_chebfd_scs_v2<<<grid, dim3(vt, LAUNCH_THREADS / vt), 0, stream>>>(v->count,
          a->C,
          w,
          a->ld,
          v->ptr,
          v->elemBase,
          v->rowBase,
          v->lens,
          v->colInd,
          v->val,
          a->x + v0,
          a->cA,
          a->p + v0,
          a->cP,
          (a->q != NULL) ? a->q + v0 : NULL,
          a->cQ,
          a->y + v0,
          a->gc,
          (a->acc != NULL) ? a->acc + v0 : NULL,
          (a->r != NULL) ? a->r + v0 : NULL,
          a->cR);
      continue;
    }
#endif
    unsigned vt = vecTile(w);
    dim3 grid((unsigned)v->count, (unsigned)((w + vt - 1) / vt));
    kernel_chebfd_scs<<<grid, dim3(vt, LAUNCH_THREADS / vt), 0, stream>>>(v->count,
        a->C,
        w,
        a->ld,
        v->ptr,
        v->elemBase,
        v->rowBase,
        v->lens,
        v->colInd,
        v->val,
        a->x + v0,
        a->cA,
        a->p + v0,
        a->cP,
        (a->q != NULL) ? a->q + v0 : NULL,
        a->cQ,
        a->y + v0,
        a->gc,
        (a->acc != NULL) ? a->acc + v0 : NULL,
        (a->r != NULL) ? a->r + v0 : NULL,
        a->cR);
  }
}

/* Plain SpMMV on one part, same column tiling. */
typedef struct {
  const V_ELE *x;
  V_ELE *y;
  CG_UINT width, ld, nb;
  CG_UINT C;
} SpmmvArgs;

static void launchSpmmvPart(const GpuPartView *v, gpuStream_t stream, void *ua)
{
  SpmmvArgs *a = (SpmmvArgs *)ua;
  CG_UINT nb   = effNb(a->nb, a->width);
  for (CG_UINT v0 = 0; v0 < a->width; v0 += nb) {
    CG_UINT w   = (nb < a->width - v0) ? nb : a->width - v0;
    unsigned vt = vecTile(w);
    dim3 grid((unsigned)v->count, (unsigned)((w + vt - 1) / vt));
    kernel_spmmv_scs<<<grid, dim3(vt, LAUNCH_THREADS / vt), 0, stream>>>(v->count,
        a->C,
        w,
        a->ld,
        v->ptr,
        v->elemBase,
        v->rowBase,
        v->lens,
        v->colInd,
        v->val,
        a->x + v0,
        a->y + v0);
  }
}

/* Identity part view of the whole matrix (resident mode: the kernels read
 * the matrix through the pointers allocate() handed out). */
static void wholeMatrixView(const Matrix *m, GpuPartView *v)
{
  v->val      = m->val;
  v->colInd   = m->colInd;
  v->lens     = m->chunkLens;
  v->ptr      = m->chunkPtr;
  v->elemBase = 0;
  v->rowBase  = 0;
  v->count    = m->nChunks;
}

extern "C" void gpu_spMVM_nosync(Matrix *m, const V_ELE *x, V_ELE *y)
{
  gpu_spmv_scs(m->nChunks, m->C, m->chunkPtr, m->chunkLens, m->colInd, m->val, x, y);
}

extern "C" void gpu_spMMVM_nosync(Matrix *m, const DMatrix *x, DMatrix *y)
{
  kernel_spmmv_scs<<<blockGrid(m, x->nc), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      x->nc,
      x->nc,
      m->chunkPtr,
      0,
      0,
      m->chunkLens,
      m->colInd,
      m->val,
      x->entries,
      y->entries);
}

extern "C" void gpu_spMMVMFused_nosync(Matrix *m,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y)
{
  kernel_chebfd_scs<<<blockGrid(m, x->nc), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      x->nc,
      x->nc,
      m->chunkPtr,
      0,
      0,
      m->chunkLens,
      m->colInd,
      m->val,
      x->entries,
      cA,
      p->entries,
      cP,
      (q != NULL) ? q->entries : NULL,
      cQ,
      y->entries,
      VCONST(0, 0),
      NULL,
      NULL,
      VCONST(0, 0));
}

extern "C" void gpu_chebfdOp_nosync(Matrix *m,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x)
{
  kernel_chebfd_scs<<<blockGrid(m, w->nc), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      w->nc,
      w->nc,
      m->chunkPtr,
      0,
      0,
      m->chunkLens,
      m->colInd,
      m->val,
      w->entries,
      cA,
      w->entries, /* chebfdOp's cP term is the matvec operand itself */
      cP,
      (q != NULL) ? q->entries : NULL,
      cQ,
      y->entries,
      gc,
      x->entries,
      NULL,
      VCONST(0, 0));
}

extern "C" void gpu_spMVM(Matrix *m, const V_ELE *x, V_ELE *y)
{
  NVTX_RANGE_PUSH_C("gpu.spMVM", NVTX_C_MATVEC);
  gpu_spMVM_nosync(m, x, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_spMMVM(Matrix *m, const DMatrix *x, DMatrix *y)
{
  NVTX_RANGE_PUSH_C("gpu.spMMVM", NVTX_C_MATVEC);
  gpu_spMMVM_nosync(m, x, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_spMMVMFused(Matrix *m,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y)
{
  NVTX_RANGE_PUSH_C("gpu.spMMVMFused", NVTX_C_MATVEC);
  gpu_spMMVMFused_nosync(m, x, cA, p, cP, q, cQ, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_chebfdOp(Matrix *m,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x)
{
  gpu_chebfdOp_nosync(m, w, cA, cP, q, cQ, y, gc, x);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

/* ------------------------------------------------------------------ */
/*  Subblock-width variants (cheb_nb). Same kernels and semantics as   */
/*  the wrappers above, but the columns are tiled into width-nb       */
/*  slices — the "subspace blocking" knob from the ChebFD papers.     */
/* ------------------------------------------------------------------ */
extern "C" void gpu_spMMVM_nb(Matrix *m, const DMatrix *x, DMatrix *y, int nb)
{
  NVTX_RANGE_PUSH_C("gpu.spMMVM_nb", NVTX_C_MATVEC);
  GpuPartView v;
  wholeMatrixView(m, &v);
  SpmmvArgs a;
  a.x     = x->entries;
  a.y     = y->entries;
  a.width = a.ld = x->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = m->C;
  launchSpmmvPart(&v, 0, &a);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_spMMVMFused_nb(Matrix *m,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    int nb)
{
  NVTX_RANGE_PUSH_C("gpu.spMMVMFused_nb", NVTX_C_MATVEC);
  GpuPartView v;
  wholeMatrixView(m, &v);
  ChebfdArgs a;
  a.x     = x->entries;
  a.p     = p->entries;
  a.q     = (q != NULL) ? q->entries : NULL;
  a.y     = y->entries;
  a.acc   = NULL;
  a.r     = NULL;
  a.cR    = VCONST(0, 0);
  a.cA    = cA;
  a.cP    = cP;
  a.cQ    = cQ;
  a.gc    = VCONST(0, 0);
  a.width = a.ld = x->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = m->C;
  launchChebfdPart(&v, 0, &a);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_chebfdOp_nb(Matrix *m,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x,
    int nb)
{
  GpuPartView v;
  wholeMatrixView(m, &v);
  ChebfdArgs a;
  a.x     = w->entries;
  a.p     = w->entries; /* chebfdOp's cP term is the matvec operand itself */
  a.q     = (q != NULL) ? q->entries : NULL;
  a.y     = y->entries;
  a.acc   = x->entries;
  a.r     = NULL;
  a.cR    = VCONST(0, 0);
  a.cA    = cA;
  a.cP    = cP;
  a.cQ    = cQ;
  a.gc    = gc;
  a.width = a.ld = w->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = m->C;
  launchChebfdPart(&v, 0, &a);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

/* ------------------------------------------------------------------ */
/*  Stream-aware launchers for the search-space streaming              */
/*  (cuda_vector_stream.cu): whole resident matrix, `width` columns of  */
/*  blocks with leading dimension `ld`, no sync.                        */
/* ------------------------------------------------------------------ */
extern "C" void gpu_launch_chebfd(const Matrix *m,
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
    CG_UINT ld)
{
  GpuPartView v;
  wholeMatrixView(m, &v);
  ChebfdArgs a;
  a.x     = x;
  a.p     = p;
  a.q     = q;
  a.r     = r;
  a.y     = y;
  a.acc   = acc;
  a.cR    = cR;
  a.cA    = cA;
  a.cP    = cP;
  a.cQ    = cQ;
  a.gc    = gc;
  a.width = width;
  a.ld    = ld;
  a.nb    = 0;
  a.C     = m->C;
  launchChebfdPart(&v, stream, &a);
}

extern "C" void gpu_launch_spmmv(const Matrix *m,
    gpuStream_t stream,
    const V_ELE *x,
    V_ELE *y,
    CG_UINT width,
    CG_UINT ld)
{
  GpuPartView v;
  wholeMatrixView(m, &v);
  SpmmvArgs a;
  a.x     = x;
  a.y     = y;
  a.width = width;
  a.ld    = ld;
  a.nb    = 0;
  a.C     = m->C;
  launchSpmmvPart(&v, stream, &a);
}
#endif /* SCS */
