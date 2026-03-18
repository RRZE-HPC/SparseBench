/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * GPU SpMV / SpMMV kernels for the CRS (CSR) matrix format.
 * One thread per row — straightforward CSR parallelisation.
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"

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
/*  Thread (row, vec) computes one element of the output.             */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_crs(CG_UINT numRows,
    CG_UINT numVecs,
    const CG_UINT *rowPtr,
    const CG_UINT *colInd,
    const V_ELE *val,
    const V_ELE *x,
    V_ELE *y)
{
  CG_UINT row = blockIdx.x * blockDim.x + threadIdx.x;
  CG_UINT vec = threadIdx.y;

  if (row >= numRows)
    return;

  V_ELE sum = VCONST(0, 0);
  for (CG_UINT j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
    CG_UINT col = colInd[j];
    sum += val[j] * x[col * numVecs + vec];
  }

  y[row * numVecs + vec] = sum;
}

/* ------------------------------------------------------------------ */
/*  High-level synchronous wrappers matching the CPU solver.h API     */
/*  Only compiled when building with MTX_FMT=CRS                     */
/* ------------------------------------------------------------------ */
#ifdef CRS
extern "C" void gpu_spMVM_nosync(Matrix *m, const V_ELE *x, V_ELE *y)
{
  int threads = 256;
  int blocks  = (m->nr + threads - 1) / threads;
  kernel_spmv_crs<<<blocks, threads>>>(m->nr, m->rowPtr, m->colInd, m->val, x, y);
}

extern "C" void gpu_spMMVM_nosync(Matrix *m, const DMatrix *x, DMatrix *y)
{
  CG_UINT numVecs = x->nc;
  int threads_x   = 256;
  int blocks_x    = (m->nr + threads_x - 1) / threads_x;
  dim3 grid(blocks_x);
  dim3 block(threads_x, numVecs);
  kernel_spmmv_crs<<<grid, block>>>(
      m->nr, numVecs, m->rowPtr, m->colInd, m->val, x->entries, y->entries);
}

#ifdef CRS
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
#endif /* CRS */
