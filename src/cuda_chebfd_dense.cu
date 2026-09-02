/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Dense block kernels for the non-sparse ChebFD steps on resident blocks:
 * Rayleigh-Ritz projection H = Y^T A Y and the Ritz residual. Format-
 * independent: only see row-major (nr x nc) block vectors. The production
 * GPU solver streams the search space instead (cuda_vector_stream.cu, which
 * shares the Gram kernels below); these entry points serve the dispatched
 * host-side steps and the streaming parity tests.
 *
 * Must run on the device: under managed memory a host-side pass migrates
 * the whole block each way. Offsets use size_t: nr*nc overflows 32-bit.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_dense_kernels.cuh"
#include "cuda_kernels.h"
#include "gpu_backend.h"
#include "nvtx_marker.h"

#define WARP 32
#define ROWS_PER_BLOCK 8

/* Persistent scratch: Gram row-chunk partials, nSub x m x m doubles. */
static double *g_gram_partial    = NULL;
static size_t g_gram_partial_cap = 0;

extern "C" void gpu_chebfd_scratch_free(void)
{
  if (g_gram_partial != NULL) {
    GPU_SAFE_CALL(gpuFree(g_gram_partial));
    g_gram_partial     = NULL;
    g_gram_partial_cap = 0;
  }
}

/* See cuda_dense_kernels.cuh for the contract of the two Gram kernels. */
__global__ void kernel_gram_chunk(CG_UINT rows,
    int m,
    const V_ELE *A,
    CG_UINT ldA,
    const V_ELE *B,
    CG_UINT ldB,
    double *partial,
    CG_UINT rowsPerSub)
{
  __shared__ V_ELE As[GRAM_TILE][GRAM_TILE];
  __shared__ V_ELE Bs[GRAM_TILE][GRAM_TILE];

  /* Below-diagonal tile: uniform across the block, so early return
   * before __syncthreads is safe. */
  if (blockIdx.x > blockIdx.y)
    return;

  int i      = blockIdx.x * GRAM_TILE + threadIdx.x;
  int j      = blockIdx.y * GRAM_TILE + threadIdx.y;
  int ai     = blockIdx.x * GRAM_TILE + threadIdx.x;
  int bj     = blockIdx.y * GRAM_TILE + threadIdx.x;

  CG_UINT rs = (CG_UINT)blockIdx.z * rowsPerSub;
  CG_UINT re = rs + rowsPerSub;
  if (re > rows)
    re = rows;

  double acc = 0.0;
  for (CG_UINT r0 = rs; r0 < re; r0 += GRAM_TILE) {
    CG_UINT r = r0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (r < re && ai < m) ? A[(size_t)r * ldA + (size_t)ai] : VCONST(0, 0);
    Bs[threadIdx.y][threadIdx.x] =
        (r < re && bj < m) ? B[(size_t)r * ldB + (size_t)bj] : VCONST(0, 0);
    __syncthreads();
    for (int rr = 0; rr < GRAM_TILE; rr++) {
      acc += asReal(As[rr][threadIdx.x]) * asReal(Bs[rr][threadIdx.y]);
    }
    __syncthreads();
  }

  if (i < m && j < m && i <= j) {
    partial[((size_t)blockIdx.z * (size_t)m + (size_t)i) * (size_t)m + (size_t)j] = acc;
  }
}

__global__ void kernel_gram_accum(
    int m, int nSub, const double *partial, double *G, int accumulate)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i >= m || i > j)
    return;
  double s = 0.0;
  for (int c = 0; c < nSub; c++) {
    s += partial[((size_t)c * (size_t)m + (size_t)i) * (size_t)m + (size_t)j];
  }
  double v = (accumulate ? G[(size_t)i * (size_t)m + (size_t)j] : 0.0) + s;
  G[(size_t)i * (size_t)m + (size_t)j] = v;
  G[(size_t)j * (size_t)m + (size_t)i] = v;
}

/* Ritz residual: avbuf = AY*evk - evalk*(Y*evk) */
__global__ void kernel_ritz_residual(CG_UINT nr,
    int m,
    const V_ELE *Ye,
    const V_ELE *AYe,
    const double *evk,
    double evalk,
    V_ELE *avbuf)
{
  __shared__ double sV[ROWS_PER_BLOCK][WARP];
  __shared__ double sA[ROWS_PER_BLOCK][WARP];

  CG_UINT row = (CG_UINT)blockIdx.x * blockDim.y + threadIdx.y;
  double vv   = 0.0;
  double av   = 0.0;
  if (row < nr) {
    size_t base = (size_t)row * (size_t)m;
    for (int j = threadIdx.x; j < m; j += WARP) {
      double c = evk[j];
      vv += c * asReal(Ye[base + (size_t)j]);
      av += c * asReal(AYe[base + (size_t)j]);
    }
  }
  sV[threadIdx.y][threadIdx.x] = vv;
  sA[threadIdx.y][threadIdx.x] = av;
  __syncthreads();

  for (int t = WARP / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) {
      sV[threadIdx.y][threadIdx.x] += sV[threadIdx.y][threadIdx.x + t];
      sA[threadIdx.y][threadIdx.x] += sA[threadIdx.y][threadIdx.x + t];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && row < nr) {
    avbuf[row] = VCONST(sA[threadIdx.y][0] - evalk * sV[threadIdx.y][0], 0);
  }
}

/* Host entry points; signatures mirror the CPU versions in chebFDSolver.h. */

extern "C" void gpu_gramYtAY(
    CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H)
{
  NVTX_RANGE_PUSH_C("gpu.gramYtAY", NVTX_C_RR);
  gpuGrowBuffer((void **)&g_gram_partial,
      &g_gram_partial_cap,
      (size_t)gramSubs(m) * (size_t)m * (size_t)m,
      sizeof(double));
  launchGram(nr, m, Ye, (CG_UINT)m, AYe, (CG_UINT)m, g_gram_partial, H, 0, 0);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

extern "C" void gpu_computeRitzResidual(DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double evalk,
    double *evec,
    int k,
    double *evk,
    V_ELE *avbuf)
{
  NVTX_RANGE_PUSH_C("gpu.ritzResidual", NVTX_C_RESID);
  /* evec was produced by jacobiEigen on the host. */
  for (int j = 0; j < m; j++) {
    evk[j] = evec[(size_t)j * (size_t)m + (size_t)k];
  }

  int blocks = (int)((nr + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  kernel_ritz_residual<<<blocks, dim3(WARP, ROWS_PER_BLOCK)>>>(
      nr, m, Y->entries, AY->entries, evk, evalk, avbuf);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}
