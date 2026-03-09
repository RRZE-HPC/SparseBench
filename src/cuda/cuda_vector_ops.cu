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
#include "gpu_backend.h"
#include "cuda_kernels.h"

/* ------------------------------------------------------------------ */
/*  waxpby:  w = alpha*x + beta*y                                    */
/* ------------------------------------------------------------------ */
__global__ void kernel_waxpby(CG_UINT n, CG_FLOAT alpha, const CG_FLOAT *x,
    CG_FLOAT beta, const CG_FLOAT *y, CG_FLOAT *w)
{
    CG_UINT i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}

extern "C"
void gpu_waxpby(CG_UINT n, CG_FLOAT alpha, const CG_FLOAT *x,
    CG_FLOAT beta, const CG_FLOAT *y, CG_FLOAT *w)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_waxpby<<<blocks, threads>>>(n, alpha, x, beta, y, w);
}

/* ------------------------------------------------------------------ */
/*  ddot:  result = x^T * y   (uses shared-memory reduction)         */
/* ------------------------------------------------------------------ */
__global__ void kernel_ddot(CG_UINT n, const CG_FLOAT *x,
    const CG_FLOAT *y, CG_FLOAT *partial)
{
    extern __shared__ CG_FLOAT sdata[];

    CG_UINT tid = threadIdx.x;
    CG_UINT i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] * y[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C"
void gpu_ddot(CG_UINT n, const CG_FLOAT *x, const CG_FLOAT *y, CG_FLOAT *result)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    CG_FLOAT *d_partial;
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Malloc)(&d_partial, blocks * sizeof(CG_FLOAT)));

    kernel_ddot<<<blocks, threads, threads * sizeof(CG_FLOAT)>>>(n, x, y, d_partial);

    /* Final reduction on host (small array) */
    CG_FLOAT *h_partial = (CG_FLOAT *)malloc(blocks * sizeof(CG_FLOAT));
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Memcpy)(h_partial, d_partial,
        blocks * sizeof(CG_FLOAT), gpuMemcpyDeviceToHost));

    CG_FLOAT sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    *result = sum;

    free(h_partial);
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(Free)(d_partial));
}

/* ------------------------------------------------------------------ */
/*  Device init / finalize                                            */
/* ------------------------------------------------------------------ */
extern "C"
void gpu_init(int device)
{
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(SetDevice)(device));

    gpuDeviceProp_t prop;
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(GetDeviceProperties)(&prop, device));
    printf("%s device: %s (compute %d.%d)\n",
        GPU_BACKEND_STR, prop.name, prop.major, prop.minor);
}

extern "C"
void gpu_finalize(void)
{
    GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(DeviceReset)());
}

/* ------------------------------------------------------------------ */
/*  Managed-memory allocation helpers                                 */
/* ------------------------------------------------------------------ */
extern "C"
void *gpu_allocate_managed(size_t bytes)
{
    void *ptr = NULL;
    GPU_SAFE_CALL(gpuMallocManaged(&ptr, bytes));
    return ptr;
}

extern "C"
void gpu_free_managed(void *ptr)
{
    GPU_SAFE_CALL(gpuFree(ptr));
}

/* ------------------------------------------------------------------ */
/*  Synchronous wrappers — behave like CPU kernels                    */
/*  Use managed memory pointers; call DeviceSynchronize after launch  */
/* ------------------------------------------------------------------ */
extern "C"
void gpu_waxpby_sync(CG_UINT n, CG_FLOAT alpha, const CG_FLOAT *x,
    CG_FLOAT beta, const CG_FLOAT *y, CG_FLOAT *w)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_waxpby<<<blocks, threads>>>(n, alpha, x, beta, y, w);
    GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C"
void gpu_ddot_sync(CG_UINT n, const CG_FLOAT *x, const CG_FLOAT *y, CG_FLOAT *result)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    CG_FLOAT *partial;
    GPU_SAFE_CALL(gpuMallocManaged(&partial, blocks * sizeof(CG_FLOAT)));

    kernel_ddot<<<blocks, threads, threads * sizeof(CG_FLOAT)>>>(n, x, y, partial);
    GPU_SAFE_CALL(gpuDeviceSynchronize());

    CG_FLOAT sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += partial[i];
    *result = sum;

    GPU_SAFE_CALL(gpuFree(partial));
}
