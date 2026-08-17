/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Dense block kernels for the non-sparse ChebFD steps: orthogonalization
 * (CGS2), the Rayleigh-Ritz projection H = Y^T A Y, and the Ritz residual.
 *
 * These are format-independent — they only see the row-major (nr x nc) block
 * vectors — so they live outside the matrix-format .cu files.
 *
 * Why they must be on the device at all: with the filter ported, these steps
 * are what remains, and under managed memory a host-side pass over a block
 * migrates the whole block (hundreds of MB) each way. Leaving even the cheap
 * O(nr*m) tail steps (scale/compact/repack) on the host would fault the block
 * back per column and undo the win from porting the hot loops.
 *
 * Element offsets use size_t: nr*nc overflows a 32-bit CG_UINT well within the
 * problem sizes this benchmark targets (see REVIEW_TODO.md item 2, which
 * defers widening the host-side signatures to their own commit).
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "gpu_backend.h"

#define WARP 32
#define ROWS_PER_BLOCK 8
#define GRAM_TILE 16
#define PROJ_CHUNKS 256
#define DOT_BLOCKS 256
#define LIN_THREADS 256

/* Real part of a block element as a double. The Rayleigh-Ritz projection and
 * the Ritz residual are real-valued (solveChebFD rejects USE_COMPLEX), but the
 * .cu files are still compiled in a complex build, where V_ELE is a
 * thrust::complex and has no implicit conversion to double. */
__device__ __host__ static inline double asReal(V_ELE z)
{
#ifdef USE_COMPLEX
  return (double)VREAL(z);
#else
  return (double)z;
#endif
}

/* ------------------------------------------------------------------ */
/*  Persistent scratch                                                */
/* ------------------------------------------------------------------ */
static V_ELE *g_partial     = NULL; /* projection partials, PROJ_CHUNKS x nc */
static size_t g_partial_cap = 0;
static V_ELE *g_dotPartial  = NULL; /* column-norm partials, DOT_BLOCKS      */
static V_ELE *g_scalar      = NULL; /* one device scalar for readback        */
static V_ELE *g_repack      = NULL; /* out-of-place repack destination       */
static size_t g_repack_cap  = 0;

static void ensureScratch(size_t partialElems, size_t repackElems)
{
  if (partialElems > g_partial_cap) {
    if (g_partial != NULL) {
      GPU_SAFE_CALL(gpuFree(g_partial));
    }
    GPU_SAFE_CALL(gpuMalloc((void **)&g_partial, partialElems * sizeof(V_ELE)));
    g_partial_cap = partialElems;
  }
  if (repackElems > g_repack_cap) {
    if (g_repack != NULL) {
      GPU_SAFE_CALL(gpuFree(g_repack));
    }
    GPU_SAFE_CALL(gpuMalloc((void **)&g_repack, repackElems * sizeof(V_ELE)));
    g_repack_cap = repackElems;
  }
  if (g_dotPartial == NULL) {
    GPU_SAFE_CALL(gpuMalloc((void **)&g_dotPartial, DOT_BLOCKS * sizeof(V_ELE)));
  }
  if (g_scalar == NULL) {
    GPU_SAFE_CALL(gpuMalloc((void **)&g_scalar, sizeof(V_ELE)));
  }
}

extern "C" void gpu_chebfd_scratch_free(void)
{
  if (g_partial != NULL) {
    GPU_SAFE_CALL(gpuFree(g_partial));
    g_partial     = NULL;
    g_partial_cap = 0;
  }
  if (g_repack != NULL) {
    GPU_SAFE_CALL(gpuFree(g_repack));
    g_repack     = NULL;
    g_repack_cap = 0;
  }
  if (g_dotPartial != NULL) {
    GPU_SAFE_CALL(gpuFree(g_dotPartial));
    g_dotPartial = NULL;
  }
  if (g_scalar != NULL) {
    GPU_SAFE_CALL(gpuFree(g_scalar));
    g_scalar = NULL;
  }
}

/* ------------------------------------------------------------------ */
/*  CGS2 projection:  coefs[0:m] = E[:,0:m]^T * e[:,k]                */
/*                                                                    */
/*  threadIdx.x indexes j so the read of e[r*nc + j] is coalesced;    */
/*  e[r*nc + k] is uniform across the block and broadcasts. Rows are  */
/*  split over blockIdx.y into PROJ_CHUNKS partial sums, reduced by   */
/*  a second kernel — deterministic, unlike an atomicAdd fan-in.      */
/* ------------------------------------------------------------------ */
__global__ void kernel_proj(
    CG_UINT nr, const V_ELE *e, int nc, int k, int m, V_ELE *partial, CG_UINT rowsPerChunk)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m)
    return; /* safe: no shared memory or __syncthreads in this kernel */

  CG_UINT r0 = (CG_UINT)blockIdx.y * rowsPerChunk;
  CG_UINT r1 = r0 + rowsPerChunk;
  if (r1 > nr)
    r1 = nr;

  V_ELE acc = VCONST(0, 0);
  for (CG_UINT r = r0; r < r1; r++) {
    size_t base = (size_t)r * (size_t)nc;
    acc += e[base + (size_t)k] * e[base + (size_t)j];
  }
  partial[(size_t)blockIdx.y * (size_t)m + (size_t)j] = acc;
}

__global__ void kernel_proj_reduce(
    int m, int nChunks, const V_ELE *partial, V_ELE *coefs)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m)
    return;

  V_ELE s = VCONST(0, 0);
  for (int c = 0; c < nChunks; c++) {
    s += partial[(size_t)c * (size_t)m + (size_t)j];
  }
  coefs[j] = s;
}

/* ------------------------------------------------------------------ */
/*  CGS2 subtract:  e[:,k] -= E[:,0:m] * coefs                        */
/*  One warp per row, lanes striding over j (coalesced), then a       */
/*  shared-memory reduction across the warp.                          */
/* ------------------------------------------------------------------ */
__global__ void kernel_subtract(
    CG_UINT nr, V_ELE *e, int nc, int k, int m, const V_ELE *coefs)
{
  __shared__ V_ELE sred[ROWS_PER_BLOCK][WARP];

  CG_UINT row = (CG_UINT)blockIdx.x * blockDim.y + threadIdx.y;
  V_ELE s     = VCONST(0, 0);
  if (row < nr) {
    size_t base = (size_t)row * (size_t)nc;
    for (int j = threadIdx.x; j < m; j += WARP) {
      s += coefs[j] * e[base + (size_t)j];
    }
  }
  sred[threadIdx.y][threadIdx.x] = s;
  __syncthreads();

  /* Unconditional syncs: every thread reaches them, only the accumulate above
   * was predicated on row < nr. */
  for (int t = WARP / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) {
      sred[threadIdx.y][threadIdx.x] += sred[threadIdx.y][threadIdx.x + t];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && row < nr) {
    e[(size_t)row * (size_t)nc + (size_t)k] -= sred[threadIdx.y][0];
  }
}

/* ------------------------------------------------------------------ */
/*  Column 2-norm squared of e[:,k] (stride nc)                       */
/* ------------------------------------------------------------------ */
__global__ void kernel_col_dot(CG_UINT nr, const V_ELE *e, int nc, int k, V_ELE *partial)
{
  __shared__ V_ELE sd[LIN_THREADS];

  V_ELE s = VCONST(0, 0);
  for (CG_UINT r = blockIdx.x * blockDim.x + threadIdx.x; r < nr;
       r += (CG_UINT)gridDim.x * blockDim.x) {
    V_ELE v = e[(size_t)r * (size_t)nc + (size_t)k];
    s += v * v;
  }
  sd[threadIdx.x] = s;
  __syncthreads();

  for (int t = blockDim.x / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) {
      sd[threadIdx.x] += sd[threadIdx.x + t];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = sd[0];
  }
}

__global__ void kernel_reduce_scalar(int n, const V_ELE *partial, V_ELE *result)
{
  __shared__ V_ELE sd[LIN_THREADS];

  V_ELE s = VCONST(0, 0);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    s += partial[i];
  }
  sd[threadIdx.x] = s;
  __syncthreads();

  for (int t = blockDim.x / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) {
      sd[threadIdx.x] += sd[threadIdx.x + t];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *result = sd[0];
  }
}

/* ------------------------------------------------------------------ */
/*  Scale e[:,k] by inv, and compact it to column mDst if they differ */
/* ------------------------------------------------------------------ */
__global__ void kernel_scale_compact(
    CG_UINT nr, V_ELE *e, int nc, int k, int mDst, V_ELE inv)
{
  CG_UINT r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nr)
    return;

  size_t base = (size_t)r * (size_t)nc;
  V_ELE v     = e[base + (size_t)k] * inv;
  e[base + (size_t)k] = v;
  if (mDst != k) {
    e[base + (size_t)mDst] = v;
  }
}

/* ------------------------------------------------------------------ */
/*  Repack accepted columns from stride nc to stride m                */
/*                                                                    */
/*  Out-of-place, unlike the host version. In place this is only safe */
/*  serially ascending in r: row r writes [r*m, r*m+m) which overlaps */
/*  the source window [r'*nc, r'*nc+m) of an earlier row r' ~ r*m/nc, */
/*  so a parallel pass would race. Gathering into scratch and copying */
/*  back costs one extra pass and no ordering constraint.             */
/* ------------------------------------------------------------------ */
__global__ void kernel_repack_gather(
    CG_UINT nr, const V_ELE *src, int nc, int m, V_ELE *dst)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)nr * (size_t)m;
  if (idx >= total)
    return;

  size_t r = idx / (size_t)m;
  size_t i = idx - r * (size_t)m;
  dst[idx] = src[r * (size_t)nc + i];
}

__global__ void kernel_copy(size_t n, const V_ELE *src, V_ELE *dst)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

/* ------------------------------------------------------------------ */
/*  Rayleigh-Ritz projection  H = Y^T (A Y),  H is m x m              */
/*                                                                    */
/*  Tiled so both loads are coalesced: a GRAM_TILE-wide row segment   */
/*  of a row-major block is contiguous. Only tiles on or above the    */
/*  diagonal run, and each writes both H[i,j] and H[j,i] from the     */
/*  same accumulator — H must come out exactly symmetric, since       */
/*  jacobiEigen assumes it and the two triangles would otherwise      */
/*  differ in the last bits (Y_i^T A Y_j and Y_j^T A Y_i are equal    */
/*  only in exact arithmetic).                                        */
/* ------------------------------------------------------------------ */
__global__ void kernel_gram(
    CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H)
{
  __shared__ V_ELE As[GRAM_TILE][GRAM_TILE];
  __shared__ V_ELE Bs[GRAM_TILE][GRAM_TILE];

  /* Whole tile is strictly below the diagonal — uniform across the block, so
   * returning before the __syncthreads below is safe. */
  if (blockIdx.x > blockIdx.y)
    return;

  int i      = blockIdx.x * GRAM_TILE + threadIdx.x;
  int j      = blockIdx.y * GRAM_TILE + threadIdx.y;

  double acc = 0.0;
  for (CG_UINT r0 = 0; r0 < nr; r0 += GRAM_TILE) {
    CG_UINT r = r0 + threadIdx.y;
    int ai    = blockIdx.x * GRAM_TILE + threadIdx.x;
    int bj    = blockIdx.y * GRAM_TILE + threadIdx.x;

    As[threadIdx.y][threadIdx.x] =
        (r < nr && ai < m) ? Ye[(size_t)r * (size_t)m + (size_t)ai] : VCONST(0, 0);
    Bs[threadIdx.y][threadIdx.x] =
        (r < nr && bj < m) ? AYe[(size_t)r * (size_t)m + (size_t)bj] : VCONST(0, 0);
    __syncthreads();

    for (int rr = 0; rr < GRAM_TILE; rr++) {
      acc += asReal(As[rr][threadIdx.x]) * asReal(Bs[rr][threadIdx.y]);
    }
    __syncthreads();
  }

  if (i < m && j < m && i <= j) {
    H[(size_t)i * (size_t)m + (size_t)j] = acc;
    H[(size_t)j * (size_t)m + (size_t)i] = acc;
  }
}

/* ------------------------------------------------------------------ */
/*  Ritz residual:  avbuf = AY*evk - evalk*(Y*evk)                    */
/*  One warp per row; lanes stride over j so both block reads are     */
/*  coalesced, then a shared reduction across the warp.               */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/*  Host-side entry points, matching the CPU signatures in            */
/*  chebFDSolver.h so a call site switches by renaming the call.      */
/* ------------------------------------------------------------------ */

extern "C" void gpu_gramYtAY(
    CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H)
{
  int tiles = (m + GRAM_TILE - 1) / GRAM_TILE;
  kernel_gram<<<dim3(tiles, tiles), dim3(GRAM_TILE, GRAM_TILE)>>>(nr, m, Ye, AYe, H);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
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
  /* Gather the k-th Ritz vector on the host: evec is m x m and was just
   * produced there by jacobiEigen, so it is host-resident already. */
  for (int j = 0; j < m; j++) {
    evk[j] = evec[(size_t)j * (size_t)m + (size_t)k];
  }

  int blocks = (int)((nr + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  kernel_ritz_residual<<<blocks, dim3(WARP, ROWS_PER_BLOCK)>>>(
      nr, m, Y->entries, AY->entries, evk, evalk, avbuf);
  GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C" int gpu_orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol)
{
  if (nc < 1 || nr == 0) {
    return 0;
  }

  ensureScratch((size_t)PROJ_CHUNKS * (size_t)nc, (size_t)nr * (size_t)nc);

  V_ELE *coefs = NULL;
  GPU_SAFE_CALL(gpuMalloc((void **)&coefs, (size_t)nc * sizeof(V_ELE)));

  CG_UINT rowsPerChunk = (nr + PROJ_CHUNKS - 1) / PROJ_CHUNKS;
  int rowBlocks        = (int)((nr + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  int linBlocks        = (int)((nr + LIN_THREADS - 1) / LIN_THREADS);

  int m                = 0;
  for (int k = 0; k < nc; k++) {
    for (int pass = 0; pass < 2 && m > 0; pass++) {
      int jBlocks = (m + LIN_THREADS - 1) / LIN_THREADS;
      kernel_proj<<<dim3(jBlocks, PROJ_CHUNKS), LIN_THREADS>>>(
          nr, e, nc, k, m, g_partial, rowsPerChunk);
      kernel_proj_reduce<<<jBlocks, LIN_THREADS>>>(m, PROJ_CHUNKS, g_partial, coefs);
      kernel_subtract<<<rowBlocks, dim3(WARP, ROWS_PER_BLOCK)>>>(nr, e, nc, k, m, coefs);
    }

    kernel_col_dot<<<DOT_BLOCKS, LIN_THREADS>>>(nr, e, nc, k, g_dotPartial);
    kernel_reduce_scalar<<<1, LIN_THREADS>>>(DOT_BLOCKS, g_dotPartial, g_scalar);

    V_ELE nrm2;
    /* Synchronous copy: also the sync point for the kernels above. */
    GPU_SAFE_CALL(gpuMemcpy(&nrm2, g_scalar, sizeof(V_ELE), gpuMemcpyDeviceToHost));

    double nrm = sqrt(asReal(nrm2));
    if (nrm < tol) {
      continue; /* linearly dependent -> drop */
    }

    V_ELE inv = VCONST(1.0 / nrm, 0);
    kernel_scale_compact<<<linBlocks, LIN_THREADS>>>(nr, e, nc, k, m, inv);
    m++;
  }

  GPU_SAFE_CALL(gpuFree(coefs));

  if (m != nc && m > 0) {
    size_t total = (size_t)nr * (size_t)m;
    int blocks   = (int)((total + LIN_THREADS - 1) / LIN_THREADS);
    kernel_repack_gather<<<blocks, LIN_THREADS>>>(nr, e, nc, m, g_repack);
    kernel_copy<<<blocks, LIN_THREADS>>>(total, g_repack, e);
  }
  GPU_SAFE_CALL(gpuDeviceSynchronize());

  return m;
}
