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
 * reaches any row, which is why the block vectors must stay fully
 * device-resident when the matrix is streamed (cuda_matrix_stream.cu).
 *
 * The block vectors are addressed with a separate leading dimension ld >=
 * numVecs, so a launch may cover a width-numVecs column slice at offset v0
 * of a wider row-major block (base pointer + v0). The plain wrappers pass
 * the identity view (ld = numVecs, bases 0, unsliced arrays); the _nb
 * wrappers tile the columns; the gpu_stream_* sweeps combine both.
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"

#include "cuda_matrix_stream.h"
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
/*    y   = cA*(A*xin) + cP*p + cQ*q          (q optional)            */
/*    acc = acc + gc*y                        (acc optional)          */
/*                                                                    */
/*  Covers both host entry points with one kernel:                    */
/*    spMMVMFused -> acc = NULL (no accumulate)                       */
/*    chebfdOp    -> p = xin, acc = the polynomial accumulator        */
/*  Both are row-local, so y may alias q and p may alias xin.         */
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
    V_ELE *acc)
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
    y[e] = t;
    if (acc != NULL) {
      acc[e] += gc * t;
    }
  }
}

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
  const V_ELE *x, *p, *q;
  V_ELE *y, *acc;
  V_ELE cA, cP, cQ, gc;
  CG_UINT width, ld, nb;
  CG_UINT C;
} ChebfdArgs;

/* Effective subblock width: 0 or >= width means "one full-width launch". */
static inline CG_UINT effNb(CG_UINT nb, CG_UINT width)
{
  return (nb == 0 || nb >= width) ? width : nb;
}

/* Fused kernel on one part (or an identity view of the whole matrix),
 * tiled into width-nb column slices. stream 0 = legacy default stream. */
static void launchChebfdPart(const GpuPartView *v, gpuStream_t stream, void *ua)
{
  ChebfdArgs *a = (ChebfdArgs *)ua;
  CG_UINT nb    = effNb(a->nb, a->width);
  for (CG_UINT v0 = 0; v0 < a->width; v0 += nb) {
    CG_UINT w   = (nb < a->width - v0) ? nb : a->width - v0;
    dim3 grid((unsigned)v->count, (unsigned)((w + VEC_TILE - 1) / VEC_TILE));
    kernel_chebfd_scs<<<grid, dim3(VEC_TILE, ROW_TILE), 0, stream>>>(v->count,
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
        (a->acc != NULL) ? a->acc + v0 : NULL);
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
    dim3 grid((unsigned)v->count, (unsigned)((w + VEC_TILE - 1) / VEC_TILE));
    kernel_spmmv_scs<<<grid, dim3(VEC_TILE, ROW_TILE), 0, stream>>>(v->count,
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
      NULL);
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
      x->entries);
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
/*  Streaming sweeps (host-resident matrix; see cuda_matrix_stream).   */
/*  Each streamed part is applied to all columns (optionally tiled     */
/*  by nb) before the pipeline moves on to the next part.              */
/* ------------------------------------------------------------------ */
extern "C" void gpu_stream_spMMVM(GpuMatrixStream *s,
    const DMatrix *x,
    DMatrix *y,
    int nb)
{
  NVTX_RANGE_PUSH_C("gpu.stream.spMMVM", NVTX_C_STREAM);
  SpmmvArgs a;
  a.x     = x->entries;
  a.y     = y->entries;
  a.width = a.ld = x->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = s->C;
  gpu_matrix_stream_sweep(s, launchSpmmvPart, &a);
  NVTX_RANGE_POP();
}

extern "C" void gpu_stream_spMMVMFused(GpuMatrixStream *s,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    int nb)
{
  NVTX_RANGE_PUSH_C("gpu.stream.spMMVMFused", NVTX_C_STREAM);
  ChebfdArgs a;
  a.x     = x->entries;
  a.p     = p->entries;
  a.q     = (q != NULL) ? q->entries : NULL;
  a.y     = y->entries;
  a.acc   = NULL;
  a.cA    = cA;
  a.cP    = cP;
  a.cQ    = cQ;
  a.gc    = VCONST(0, 0);
  a.width = a.ld = x->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = s->C;
  gpu_matrix_stream_sweep(s, launchChebfdPart, &a);
  NVTX_RANGE_POP();
}

extern "C" void gpu_stream_chebfdOp(GpuMatrixStream *s,
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
  ChebfdArgs a;
  a.x     = w->entries;
  a.p     = w->entries; /* chebfdOp's cP term is the matvec operand itself */
  a.q     = (q != NULL) ? q->entries : NULL;
  a.y     = y->entries;
  a.acc   = x->entries;
  a.cA    = cA;
  a.cP    = cP;
  a.cQ    = cQ;
  a.gc    = gc;
  a.width = a.ld = w->nc;
  a.nb    = (CG_UINT)nb;
  a.C     = s->C;
  gpu_matrix_stream_sweep(s, launchChebfdPart, &a);
}
#endif /* SCS */
