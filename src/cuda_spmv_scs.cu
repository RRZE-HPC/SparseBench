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
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"

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

  CG_UINT offset = chunkPtr[chunk];
  CG_UINT len    = chunkLens[chunk];

  V_ELE tmp      = VCONST(0, 0);
  for (CG_UINT j = 0; j < len; j++) {
    CG_UINT idx = offset + j * C + lane;
    tmp += val[idx] * x[colInd[idx]];
  }

  y[row] = tmp;
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
      nChunks, C, chunkPtr, chunkLens, colInd, val, x, y);
}

/* ------------------------------------------------------------------ */
/*  SCS SpMMV:  Y = A * X   (block-vector, numVecs columns)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_scs(CG_UINT nChunks,
    CG_UINT C,
    CG_UINT numVecs,
    const CG_UINT *chunkPtr,
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

  CG_UINT chunk  = blockIdx.x;
  CG_UINT offset = chunkPtr[chunk];
  CG_UINT len    = chunkLens[chunk];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE tmp = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      tmp += val[idx] * x[colInd[idx] * numVecs + vec];
    }
    y[(chunk * C + lane) * numVecs + vec] = tmp;
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
__global__ void kernel_chebfd_scs(CG_UINT nChunks,
    CG_UINT C,
    CG_UINT numVecs,
    const CG_UINT *chunkPtr,
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

  CG_UINT chunk  = blockIdx.x;
  CG_UINT offset = chunkPtr[chunk];
  CG_UINT len    = chunkLens[chunk];

  for (CG_UINT lane = threadIdx.y; lane < C; lane += blockDim.y) {
    V_ELE tmp = VCONST(0, 0);
    for (CG_UINT j = 0; j < len; j++) {
      CG_UINT idx = offset + j * C + lane;
      tmp += val[idx] * xin[colInd[idx] * numVecs + vec];
    }

    CG_UINT e = (chunk * C + lane) * numVecs + vec;
    V_ELE t   = cA * tmp + cP * p[e];
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

extern "C" void gpu_spMVM_nosync(Matrix *m, const V_ELE *x, V_ELE *y)
{
  gpu_spmv_scs(m->nChunks, m->C, m->chunkPtr, m->chunkLens, m->colInd, m->val, x, y);
}

extern "C" void gpu_spMMVM_nosync(Matrix *m, const DMatrix *x, DMatrix *y)
{
  CG_UINT numVecs = x->nc;
  kernel_spmmv_scs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      numVecs,
      m->chunkPtr,
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
  CG_UINT numVecs = x->nc;
  kernel_chebfd_scs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      numVecs,
      m->chunkPtr,
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
  CG_UINT numVecs = w->nc;
  kernel_chebfd_scs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(m->nChunks,
      m->C,
      numVecs,
      m->chunkPtr,
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
  gpu_spMVM_nosync(m, x, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C" void gpu_spMMVM(Matrix *m, const DMatrix *x, DMatrix *y)
{
  gpu_spMMVM_nosync(m, x, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
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
  gpu_spMMVMFused_nosync(m, x, cA, p, cP, q, cQ, y);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
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
#endif /* SCS */
