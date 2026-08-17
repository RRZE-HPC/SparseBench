/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Common CUDA vector operations: waxpby and ddot.
 * Each kernel lives in its own .cu file for easy extensibility —
 * add new kernels by creating additional .cu files in src/cuda/.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "gpu_backend.h"

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
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  kernel_waxpby<<<blocks, threads>>>(n, alpha, x, beta, y, w);
}

/* ------------------------------------------------------------------ */
/*  waxpby3:  w = a*x + b*y + c*z                                    */
/*                                                                    */
/*  Only one call per ChebFD filter application (against Np calls of  */
/*  the recurrence kernel), but it still has to run on the device:    */
/*  under managed memory the cost of a host-side kernel is not its    */
/*  flops, it is migrating all three blocks host-ward and back once   */
/*  per filter application.                                           */
/* ------------------------------------------------------------------ */
__global__ void kernel_waxpby3(CG_UINT n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w)
{
  CG_UINT i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    w[i] = a * x[i] + b * y[i] + c * z[i];
  }
}

extern "C" void gpu_waxpby3(CG_UINT n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w)
{
  int threads = 256;
  int blocks  = (int)((n + threads - 1) / threads);
  kernel_waxpby3<<<blocks, threads>>>(n, a, x, b, y, c, z, w);
}

extern "C" void gpu_waxpby3_sync(CG_UINT n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w)
{
  gpu_waxpby3(n, a, x, b, y, c, z, w);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

/* ------------------------------------------------------------------ */
/*  ddot:  result = x^T * y   (uses shared-memory reduction)         */
/* ------------------------------------------------------------------ */
__global__ void kernel_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *partial)
{
  extern __shared__ V_ELE sdata[];

  CG_UINT tid = threadIdx.x;
  CG_UINT i   = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid]  = (i < n) ? VCONJ(x[i]) * y[i] : VCONST(0, 0);
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    partial[blockIdx.x] = sdata[0];
}

/* Final stage: collapse the per-block partials into a single scalar
 * entirely on the device, so the host side only has to copy back
 * sizeof(V_ELE) bytes per ddot. */
__global__ void kernel_ddot_finalize(CG_UINT n, const V_ELE *partial, V_ELE *result)
{
  extern __shared__ V_ELE sfin[];

  CG_UINT tid = threadIdx.x;
  V_ELE v     = VCONST(0, 0);
  for (CG_UINT i = tid; i < n; i += blockDim.x) {
    v += partial[i];
  }
  sfin[tid] = v;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sfin[tid] += sfin[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    *result = sfin[0];
}

/* Persistent scratch for ddot — sized lazily on first / largest call,
 * freed in gpu_finalize(). Plain device memory (not managed) so there
 * is no per-call page migration and no per-call alloc/free overhead. */
static V_ELE *g_ddot_partial    = NULL;
static size_t g_ddot_partial_cap = 0; /* capacity in elements */
static V_ELE *g_ddot_result_d   = NULL;

static void ensure_ddot_scratch(size_t blocks)
{
  if (blocks > g_ddot_partial_cap) {
    if (g_ddot_partial != NULL) {
      GPU_SAFE_CALL(gpuFree(g_ddot_partial));
    }
    GPU_SAFE_CALL(gpuMalloc((void **)&g_ddot_partial, blocks * sizeof(V_ELE)));
    g_ddot_partial_cap = blocks;
  }
  if (g_ddot_result_d == NULL) {
    GPU_SAFE_CALL(gpuMalloc((void **)&g_ddot_result_d, sizeof(V_ELE)));
  }
}

extern "C" void gpu_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;

  ensure_ddot_scratch((size_t)blocks);

  kernel_ddot<<<blocks, threads, threads * sizeof(V_ELE)>>>(n, x, y, g_ddot_partial);

  int finalize_threads = 256;
  kernel_ddot_finalize<<<1, finalize_threads, finalize_threads * sizeof(V_ELE)>>>(
      blocks, g_ddot_partial, g_ddot_result_d);

  GPU_SAFE_CALL(gpuMemcpy(result, g_ddot_result_d, sizeof(V_ELE), gpuMemcpyDeviceToHost));
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
  gpu_chebfd_scratch_free();
  if (g_ddot_partial != NULL) {
    GPU_SAFE_CALL(gpuFree(g_ddot_partial));
    g_ddot_partial     = NULL;
    g_ddot_partial_cap = 0;
  }
  if (g_ddot_result_d != NULL) {
    GPU_SAFE_CALL(gpuFree(g_ddot_result_d));
    g_ddot_result_d = NULL;
  }
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
/* ------------------------------------------------------------------ */
extern "C" void gpu_waxpby_nosync(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  kernel_waxpby<<<blocks, threads>>>(n, alpha, x, beta, y, w);
}

extern "C" void gpu_waxpby_sync(
    CG_UINT n, V_ELE alpha, const V_ELE *x, V_ELE beta, const V_ELE *y, V_ELE *w)
{
  gpu_waxpby_nosync(n, alpha, x, beta, y, w);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C" void gpu_ddot_sync(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;

  ensure_ddot_scratch((size_t)blocks);

  kernel_ddot<<<blocks, threads, threads * sizeof(V_ELE)>>>(n, x, y, g_ddot_partial);

  int finalize_threads = 256;
  kernel_ddot_finalize<<<1, finalize_threads, finalize_threads * sizeof(V_ELE)>>>(
      blocks, g_ddot_partial, g_ddot_result_d);

  /* gpuMemcpy is synchronous w.r.t. the host, so it both waits for the
   * kernels above and delivers the scalar — no separate DeviceSynchronize. */
  GPU_SAFE_CALL(gpuMemcpy(result, g_ddot_result_d, sizeof(V_ELE), gpuMemcpyDeviceToHost));
}
