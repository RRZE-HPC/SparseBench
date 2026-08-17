/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * GPU SpMV / SpMMV / fused-ChebFD kernels for the CRS (CSR) matrix format.
 * SpMV is one thread per row. The block-vector kernels use the same
 * vector-contiguous launch shape as the SCS file — see the header comment in
 * cuda_spmv_scs.cu for why threadIdx.x indexes vectors rather than rows.
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"

#define VEC_TILE 32
#define ROW_TILE 8

/* ------------------------------------------------------------------ */
/*  CRS SpMV:  y = A * x                                             */
/*  Each thread computes one row.                                     */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmv_crs(CG_UINT numRows,
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
  for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
    sum += val[j] * x[colInd[j]];
  }

  y[row] = sum;
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
  kernel_spmv_crs<<<blocks, threads>>>(numRows, rowPtr, colInd, val, x, y);
}

/* ------------------------------------------------------------------ */
/*  CRS SpMMV:  Y = A * X   (block-vector, numVecs columns)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_crs(CG_UINT numRows,
    CG_UINT numVecs,
    const CG_UINT *rowPtr,
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
  if (row >= numRows)
    return;

  V_ELE sum   = VCONST(0, 0);
  CG_UINT end = rowPtr[row + 1];
  for (CG_UINT j = rowPtr[row]; j < end; j++) {
    sum += val[j] * x[colInd[j] * numVecs + vec];
  }

  y[row * numVecs + vec] = sum;
}

/* ------------------------------------------------------------------ */
/*  Fused ChebFD kernel — see cuda_spmv_scs.cu for the contract.      */
/*    y   = cA*(A*xin) + cP*p + cQ*q          (q optional)            */
/*    acc = acc + gc*y                        (acc optional)          */
/* ------------------------------------------------------------------ */
__global__ void kernel_chebfd_crs(CG_UINT numRows,
    CG_UINT numVecs,
    const CG_UINT *rowPtr,
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

  CG_UINT row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= numRows)
    return;

  V_ELE tmp   = VCONST(0, 0);
  CG_UINT end = rowPtr[row + 1];
  for (CG_UINT j = rowPtr[row]; j < end; j++) {
    tmp += val[j] * xin[colInd[j] * numVecs + vec];
  }

  CG_UINT e = row * numVecs + vec;
  V_ELE t   = cA * tmp + cP * p[e];
  if (q != NULL) {
    t += cQ * q[e];
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

extern "C" void gpu_spMVM_nosync(Matrix *m, const V_ELE *x, V_ELE *y)
{
  gpu_spmv_crs(m->nr, m->rowPtr, m->colInd, m->val, x, y);
}

extern "C" void gpu_spMMVM_nosync(Matrix *m, const DMatrix *x, DMatrix *y)
{
  CG_UINT numVecs = x->nc;
  kernel_spmmv_crs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(
      m->nr, numVecs, m->rowPtr, m->colInd, m->val, x->entries, y->entries);
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
  kernel_chebfd_crs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(m->nr,
      numVecs,
      m->rowPtr,
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
  kernel_chebfd_crs<<<blockGrid(m, numVecs), dim3(VEC_TILE, ROW_TILE)>>>(m->nr,
      numVecs,
      m->rowPtr,
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
#endif /* CRS */
