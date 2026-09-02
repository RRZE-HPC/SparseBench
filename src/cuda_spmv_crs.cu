/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * GPU SpMV / SpMMV / fused-ChebFD kernels for the CRS (CSR) matrix format.
 * SpMV is one thread per row. The block-vector kernels use the same
 * vector-contiguous launch shape as the SCS file — see the header comment in
 * cuda_spmv_scs.cu for why threadIdx.x indexes vectors rather than rows,
 * and for the part-view / leading-dimension conventions: rowPtr keeps
 * ABSOLUTE nnz ids (kernels rebase by -elemBase), colInd stays global, and
 * ld decouples the column-slice width from the block stride.
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"

#include "cuda_vector_stream.h"
#include "nvtx_marker.h"

#define VEC_TILE 32
#define ROW_TILE 8


/* ------------------------------------------------------------------ */
/*  CRS SpMV:  y = A * x                                             */
/*  Each thread computes one row.                                     */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmv_crs(CG_UINT numRows,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *rowPtr,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  CG_UINT row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= numRows)
    return;

  V_ELE sum = VCONST(0, 0);
  CG_UINT j0 = rowPtr[row] - elemBase;
  CG_UINT j1 = rowPtr[row + 1] - elemBase;
  for (CG_UINT j = j0; j < j1; j++) {
    sum += val[j] * x[colInd[j]];
  }

  y[rowBase + row] = sum;
}

extern "C" void gpu_spmv_crs(CG_UINT numRows,
    const CG_UINT *rowPtr,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  int threads = 256;
  int blocks  = (numRows + threads - 1) / threads;
  kernel_spmv_crs<<<blocks, threads>>>(numRows, 0, 0, rowPtr, colInd, val, x, y);
}

/* ------------------------------------------------------------------ */
/*  CRS SpMMV:  Y = A * X   (block-vector, numVecs columns)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_crs(CG_UINT nPartRows,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *rowPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  CG_UINT vec = blockIdx.y * blockDim.x + threadIdx.x;
  /* Safe early return: no shared memory and no __syncthreads below. */
  if (vec >= numVecs)
    return;

  CG_UINT row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= nPartRows)
    return;

  V_ELE sum   = VCONST(0, 0);
  CG_UINT j0  = rowPtr[row] - elemBase;
  CG_UINT end = rowPtr[row + 1] - elemBase;
  for (CG_UINT j = j0; j < end; j++) {
    sum += val[j] * x[(size_t)colInd[j] * ld + vec];
  }

  y[(size_t)(rowBase + row) * ld + vec] = sum;
}

/* ------------------------------------------------------------------ */
/*  Fused ChebFD kernel — see cuda_spmv_scs.cu for the contract.      */
/*    y   = cA*(A*xin) + cP*p + cQ*q + cR*r   (q, r optional)         */
/*    acc = acc + gc*y                        (acc optional)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_chebfd_crs(CG_UINT nPartRows,
    CG_UINT numVecs,
    CG_UINT ld,
    const CG_UINT *rowPtr,
    CG_UINT elemBase,
    CG_UINT rowBase,
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

  CG_UINT row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= nPartRows)
    return;

  V_ELE tmp   = VCONST(0, 0);
  CG_UINT j0  = rowPtr[row] - elemBase;
  CG_UINT end = rowPtr[row + 1] - elemBase;
  for (CG_UINT j = j0; j < end; j++) {
    tmp += val[j] * xin[(size_t)colInd[j] * ld + vec];
  }

  size_t e = (size_t)(rowBase + row) * ld + vec;
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

/* ------------------------------------------------------------------ */
/*  High-level wrappers matching the CPU interface in solver.h.        */
/*  Only compiled when building with MTX_FMT=CRS.                     */
/* ------------------------------------------------------------------ */
#ifdef CRS
static inline dim3 blockGrid(const Matrix *m, CG_UINT numVecs)
{
  return dim3((unsigned)((m->nr + ROW_TILE - 1) / ROW_TILE),
      (unsigned)((numVecs + VEC_TILE - 1) / VEC_TILE));
}

typedef struct {
  const V_ELE *x, *p, *q, *r;
  V_ELE *y, *acc;
  V_ELE cA, cP, cQ, gc, cR;
  CG_UINT width, ld;
} ChebfdArgs;

/* See cuda_spmv_scs.cu: narrow column slices use 16 / 8 vector lanes so a
 * warp spans several rows instead of idling. */
#define LAUNCH_THREADS (VEC_TILE * ROW_TILE)
static inline unsigned vecTile(CG_UINT w)
{
  return (w <= 8) ? 8u : (w <= 16) ? 16u : (unsigned)VEC_TILE;
}

static void launchChebfdPart(const GpuPartView *v, gpuStream_t stream, void *ua)
{
  ChebfdArgs *a = (ChebfdArgs *)ua;
  CG_UINT w     = a->width;
  unsigned vt   = vecTile(w);
  unsigned rt   = LAUNCH_THREADS / vt;
  dim3 grid((unsigned)((v->count + rt - 1) / rt), (unsigned)((w + vt - 1) / vt));
  kernel_chebfd_crs<<<grid, dim3(vt, rt), 0, stream>>>(v->count,
      w,
      a->ld,
      v->ptr,
      v->elemBase,
      v->rowBase,
      v->colInd,
      v->val,
      a->x,
      a->cA,
      a->p,
      a->cP,
      a->q,
      a->cQ,
      a->y,
      a->gc,
      a->acc,
      a->r,
      a->cR);
}

typedef struct {
  const V_ELE *x;
  V_ELE *y;
  CG_UINT width, ld;
} SpmmvArgs;

static void launchSpmmvPart(const GpuPartView *v, gpuStream_t stream, void *ua)
{
  SpmmvArgs *a = (SpmmvArgs *)ua;
  CG_UINT w    = a->width;
  unsigned vt  = vecTile(w);
  unsigned rt  = LAUNCH_THREADS / vt;
  dim3 grid((unsigned)((v->count + rt - 1) / rt), (unsigned)((w + vt - 1) / vt));
  kernel_spmmv_crs<<<grid, dim3(vt, rt), 0, stream>>>(v->count,
      w,
      a->ld,
      v->ptr,
      v->elemBase,
      v->rowBase,
      v->colInd,
      v->val,
      a->x,
      a->y);
}

static void wholeMatrixView(const Matrix *m, GpuPartView *v)
{
  v->val      = m->val;
  v->colInd   = m->colInd;
  v->lens     = NULL;
  v->ptr      = m->rowPtr;
  v->elemBase = 0;
  v->rowBase  = 0;
  v->count    = m->nr;
}

extern "C" void gpu_spMVM_nosync(Matrix *m, const V_ELE *x, V_ELE *y)
{
  gpu_spmv_crs(m->nr, m->rowPtr, m->colInd, m->val, x, y);
}

extern "C" void gpu_spMMVM_nosync(Matrix *m, const DMatrix *x, DMatrix *y)
{
  kernel_spmmv_crs<<<blockGrid(m, x->nc), dim3(VEC_TILE, ROW_TILE)>>>(
      m->nr, x->nc, x->nc, m->rowPtr, 0, 0, m->colInd, m->val, x->entries, y->entries);
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
  kernel_chebfd_crs<<<blockGrid(m, x->nc), dim3(VEC_TILE, ROW_TILE)>>>(m->nr,
      x->nc,
      x->nc,
      m->rowPtr,
      0,
      0,
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
  kernel_chebfd_crs<<<blockGrid(m, w->nc), dim3(VEC_TILE, ROW_TILE)>>>(m->nr,
      w->nc,
      w->nc,
      m->rowPtr,
      0,
      0,
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
/*  Stream-aware launchers for the search-space streaming              */
/*  (cuda_vector_stream.cu) — see cuda_spmv_scs.cu.                     */
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
  launchSpmmvPart(&v, stream, &a);
}
#endif /* CRS */
