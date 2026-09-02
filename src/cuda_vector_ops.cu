/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Common CUDA vector operations (waxpby, waxpby3, ddot), device init /
 * finalize and the mode-selectable allocation helpers.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "gpu_backend.h"
#include "gpu_cub.h"
#include "nvtx_marker.h"

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
  NVTX_RANGE_PUSH_C("gpu.waxpby3", NVTX_C_VECTOR);
  gpu_waxpby3(n, a, x, b, y, c, z, w);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

/* ------------------------------------------------------------------ */
/*  ddot:  result = x^T * y   (one launch: BlockReduce + atomicAdd)   */
/* ------------------------------------------------------------------ */
#define DDOT_THREADS 256

/* atomicAdd for V_ELE — there is no complex atomic, so under USE_COMPLEX
 * the components accumulate separately (thrust::complex is layout-locked,
 * &real() and &imag() give plain float/double pointers into the value). */
__device__ static inline void atomicAddV(V_ELE *dst, V_ELE v)
{
#ifdef USE_COMPLEX
  atomicAdd(&dst->real(), VREAL(v));
  atomicAdd(&dst->imag(), VIMAG(v));
#else
  atomicAdd(dst, v);
#endif
}

/* Each block CUB-reduces its contiguous slice, then thread 0 folds the
 * block sum straight into the result scalar with one atomicAdd — the whole
 * dot product runs in a single launch and only *result changes hands.
 * NB: the cross-block atomicAdd order is not fixed, so the sum wobbles at
 * ULP level between runs (the in-block CUB reduction is deterministic). */
__global__ void kernel_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  using BlockReduce = gpucub::BlockReduce<V_ELE, DDOT_THREADS>;
  __shared__ typename BlockReduce::TempStorage tmp;

  CG_UINT i = blockIdx.x * blockDim.x + threadIdx.x;
  V_ELE v   = (i < n) ? VCONJ(x[i]) * y[i] : VCONST(0, 0);

  V_ELE sum = BlockReduce(tmp).Sum(v);
  if (threadIdx.x == 0) {
    atomicAddV(result, sum);
  }
}

/* The device scalar the host-facing wrappers reduce into. */
static V_ELE *g_ddot_result_d = NULL;

static void ensure_ddot_result(void)
{
  if (g_ddot_result_d == NULL) {
    GPU_SAFE_CALL(gpuMalloc((void **)&g_ddot_result_d, sizeof(V_ELE)));
  }
}

/* The whole dot product on the GPU: one kernel launch reduces x^T*y and
 * updates *result_d in device memory. Asynchronous — no host transfer,
 * no sync. Callers consume *result_d with subsequent device work or an
 * explicit copy; gpu_ddot / gpu_ddot_sync do the latter. */
extern "C" void gpu_ddot_device(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result_d)
{
  int threads = DDOT_THREADS;
  int blocks  = (int)((n + threads - 1) / threads);

  /* atomicAdd accumulates, so the scalar starts each dot from zero. The
   * async zero is enqueued on the same stream as the kernel, so ordering
   * is guaranteed and the host never blocks on it. */
  GPU_SAFE_CALL(gpuMemsetAsync(result_d, 0, sizeof(V_ELE), 0));

  kernel_ddot<<<blocks, threads>>>(n, x, y, result_d);
}

extern "C" void gpu_ddot(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  ensure_ddot_result();
  gpu_ddot_device(n, x, y, g_ddot_result_d);

  GPU_SAFE_CALL(gpuMemcpy(result, g_ddot_result_d, sizeof(V_ELE), gpuMemcpyDeviceToHost));
}

/* ------------------------------------------------------------------ */
/*  Device init / finalize                                            */
/* ------------------------------------------------------------------ */
extern "C" void gpu_init(int device)
{
  NVTX_RANGE_PUSH_C("gpu.init", NVTX_C_SETUP);
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(SetDevice)(device));

  gpuDeviceProp_t prop;
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(GetDeviceProperties)(&prop, device));
  printf("%s device: %s (compute %d.%d)\n",
      GPU_BACKEND_STR,
      prop.name,
      prop.major,
      prop.minor);
  NVTX_RANGE_POP();
}

extern "C" void gpu_finalize(void)
{
  NVTX_RANGE_PUSH_C("gpu.finalize", NVTX_C_SETUP);
  gpu_chebfd_scratch_free();
  if (g_ddot_result_d != NULL) {
    GPU_SAFE_CALL(gpuFree(g_ddot_result_d));
    g_ddot_result_d = NULL;
  }
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(DeviceReset)());
  NVTX_RANGE_POP();
}

/* ------------------------------------------------------------------ */
/*  Mode-selectable allocation helpers (see AllocType in allocate.h)  */
/* ------------------------------------------------------------------ */
/* The mode locks at the first gpu_allocate (see allocDispatch); a later
 * change is rejected instead of mixing placements whose buffers free with
 * different deallocators. main.c sets the mode before any allocation. */
static AllocType g_alloc_type = ALLOC_MANAGED;
static int g_alloc_locked     = 0;

extern "C" void gpu_set_alloc_type(AllocType type)
{
  if (g_alloc_locked && type != g_alloc_type) {
    fprintf(stderr,
        "Error: gpu_set_alloc_type(%s) refused — buffers were already "
        "allocated as %s.\n",
        allocTypeName(type),
        allocTypeName(g_alloc_type));
    exit(EXIT_FAILURE);
  }
  g_alloc_type = type;
  if (type == ALLOC_PAGEABLE) {
    fprintf(stderr,
        "Warning: pageable host memory is only device-accessible where the driver "
        "transparently maps host pages; prefer explicit or managed for buffers passed to "
        "kernels.\n");
  }
}

static void *allocDispatch(size_t bytes, int device)
{
  void *ptr = NULL;
  g_alloc_locked = 1;
  if (device && g_alloc_type == ALLOC_EXPLICIT) {
    GPU_SAFE_CALL(gpuMalloc(&ptr, bytes));
    return ptr;
  }
  switch (g_alloc_type) {
  case ALLOC_PAGEABLE:
    ptr = malloc(bytes);
    if (ptr == NULL) {
      fprintf(stderr, "Error: malloc of %zu bytes failed!\n", bytes);
      exit(EXIT_FAILURE);
    }
    return ptr;
  case ALLOC_EXPLICIT:
    GPU_SAFE_CALL(gpuMallocHost(&ptr, bytes));
    return ptr;
  case ALLOC_MANAGED:
  default:
    GPU_SAFE_CALL(gpuMallocManaged(&ptr, bytes));
    return ptr;
  }
}

extern "C" void *gpu_allocate(size_t bytes)
{
  return allocDispatch(bytes, 0);
}

extern "C" void *gpu_allocate_device(size_t bytes)
{
  return allocDispatch(bytes, 1);
}

static void freeDispatch(void *ptr, int device)
{
  /* Keep in sync with allocDispatch. */
  if (device && g_alloc_type == ALLOC_EXPLICIT) {
    GPU_SAFE_CALL(gpuFree(ptr));
    return;
  }
  switch (g_alloc_type) {
  case ALLOC_PAGEABLE:
    free(ptr);
    break;
  case ALLOC_EXPLICIT:
    GPU_SAFE_CALL(gpuFreeHost(ptr));
    break;
  case ALLOC_MANAGED:
  default:
    GPU_SAFE_CALL(gpuFree(ptr));
    break;
  }
}

extern "C" void gpu_free(void *ptr)
{
  freeDispatch(ptr, 0);
}

extern "C" void gpu_free_device(void *ptr)
{
  freeDispatch(ptr, 1);
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
  NVTX_RANGE_PUSH_C("gpu.waxpby", NVTX_C_VECTOR);
  gpu_waxpby_nosync(n, alpha, x, beta, y, w);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_ddot_sync(CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *result)
{
  NVTX_RANGE_PUSH_C("gpu.ddot", NVTX_C_VECTOR);
  ensure_ddot_result();
  gpu_ddot_device(n, x, y, g_ddot_result_d);

  /* gpuMemcpy is synchronous w.r.t. the host, so it both waits for the
   * kernel above and delivers the scalar — no separate DeviceSynchronize. */
  GPU_SAFE_CALL(gpuMemcpy(result, g_ddot_result_d, sizeof(V_ELE), gpuMemcpyDeviceToHost));
  NVTX_RANGE_POP();
}
