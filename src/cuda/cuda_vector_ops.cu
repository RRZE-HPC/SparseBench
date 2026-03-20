/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Common CUDA vector operations: waxpby and ddot.
 * Each kernel lives in its own .cu file for easy extensibility —
 * add new kernels by creating additional .cu files in src/cuda/.
 */
#include "cuda_kernels.h"
#include "gpu_backend.h"
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  waxpby:  w = alpha*x + beta*y                                    */
/* ------------------------------------------------------------------ */
__global__ void kernel_waxpby(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  CG_UINT i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    w[i] = alpha * x[i] + beta * y[i];
  }
}

extern "C" void gpu_waxpby(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  int blocks  = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kernel_waxpby<<<blocks, THREADS_PER_BLOCK>>>(n, alpha, x, beta, y, w);
}

/* ------------------------------------------------------------------ */
/*  ddot:  result = x^T * y   (uses shared-memory reduction)         */
/* ------------------------------------------------------------------ */
__global__ void kernel_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *partial)
{
  using block_reduce = cub::BlockReduce<V_ELE, THREADS_PER_BLOCK>;
   __shared__ typename block_reduce::TempStorage tempstore;

  CG_UINT tid = threadIdx.x;
  CG_UINT i   = blockIdx.x * blockDim.x + threadIdx.x;

  const V_ELE thread_data  = (i < n) ? VCONJ(x[i]) * y[i] : VCONST(0, 0);
  // __syncthreads();

  auto blocksum = block_reduce(tempstore).Sum(thread_data);

  if (tid == 0)
    partial[blockIdx.x] = blocksum;
}

extern "C" void gpu_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  int blocks  = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  V_ELE *d_partial;
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Malloc)(&d_partial, blocks * sizeof(V_ELE)));

  kernel_ddot<<<blocks, THREADS_PER_BLOCK>>>(n, x, y, d_partial);

  /* Final reduction on host (small array) */
  V_ELE *h_partial = (V_ELE *)malloc(blocks * sizeof(V_ELE));
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Memcpy)(
      h_partial, d_partial, blocks * sizeof(V_ELE), gpuMemcpyDeviceToHost));

  V_ELE sum = VCONST(0, 0);
  for (int i = 0; i < blocks; i++)
    sum += h_partial[i];
  *result = sum;

  free(h_partial);
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Free)(d_partial));
}

/* ------------------------------------------------------------------ */
/*  Device init / finalize                                            */
/* ------------------------------------------------------------------ */
extern "C" void gpu_init(int device)
{
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(SetDevice)(device));

  gpuDeviceProp_t prop;
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(GetDeviceProperties)(&prop, device));
  printf("%s device: %s (compute %d.%d)\n",
      GPU_BACKEND_STR,
      prop.name,
      prop.major,
      prop.minor);
}

extern "C" void gpu_finalize(void)
{
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(DeviceReset)());
}

/* ------------------------------------------------------------------ */
/*  Managed-memory allocation helpers                                 */
/* ------------------------------------------------------------------ */
extern "C" void *gpu_allocate_managed(size_t bytes)
{
  void *ptr = NULL;
  GPU_SAFE_CALL(gpuMallocManaged(&ptr, bytes));
  return ptr;
}

extern "C" void gpu_free_managed(void *ptr)
{
  GPU_SAFE_CALL(gpuFree(ptr));
}

/* ------------------------------------------------------------------ */
/*  Synchronous wrappers — behave like CPU kernels                    */
/*  Use managed memory pointers; call DeviceSynchronize after launch  */
/* ------------------------------------------------------------------ */
extern "C" void gpu_waxpby_nosync(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  int blocks  = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kernel_waxpby<<<blocks, THREADS_PER_BLOCK>>>(n, alpha, x, beta, y, w);
}

extern "C" void gpu_waxpby_sync(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  gpu_waxpby_nosync(n, alpha, x, beta, y, w);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C" void gpu_ddot_sync(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  int blocks  = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  V_ELE *partial;
  GPU_SAFE_CALL(gpuMallocManaged(&partial, blocks * sizeof(V_ELE)));

  kernel_ddot<<<blocks, THREADS_PER_BLOCK>>>(n, x, y, partial);
  GPU_SAFE_CALL(gpuDeviceSynchronize());

  V_ELE sum = VCONST(0, 0);
  for (int i = 0; i < blocks; i++)
    sum += partial[i];
  *result = sum;

  GPU_SAFE_CALL(gpuFree(partial));
}
