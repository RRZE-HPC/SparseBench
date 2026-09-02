/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Dense block kernels for the non-sparse ChebFD steps: CGS2
 * orthogonalization, Rayleigh-Ritz projection H = Y^T A Y, Ritz residual.
 * Format-independent: only see row-major (nr x nc) block vectors.
 *
 * Must run on the device: under managed memory a host-side pass migrates
 * the whole block each way. Offsets use size_t: nr*nc overflows 32-bit.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "gpu_backend.h"
#include "gpu_cub.h"
#include "nvtx_marker.h"

#define WARP 32
#define ROWS_PER_BLOCK 8
#define GRAM_TILE 16
#define PROJ_CHUNKS 256
#define LIN_THREADS 256
/* Gram row-chunking: aim for this many blocks in flight, at most this many
 * chunks, and never more partial-buffer bytes than this. */
#define GRAM_MIN_BLOCKS 2048
#define GRAM_MAX_CHUNKS 1024
#define GRAM_PARTIAL_MAX_BYTES ((size_t)256 << 20)

/* Real part of V_ELE as double (under USE_COMPLEX V_ELE is a
 * thrust::complex with no implicit conversion to double). */
__device__ __host__ static inline double asReal(V_ELE z)
{
#ifdef USE_COMPLEX
  return (double)VREAL(z);
#else
  return (double)z;
#endif
}

/* Persistent scratch */
static V_ELE *g_partial     = NULL; /* projection partials, PROJ_CHUNKS x nc */
static size_t g_partial_cap = 0;
static V_ELE *g_scalar      = NULL; /* one device scalar for readback        */
static V_ELE *g_repack      = NULL; /* out-of-place repack destination       */
static size_t g_repack_cap  = 0;
static double *g_gram_partial    = NULL; /* Gram row-chunk partials, nChunks x m x m */
static size_t g_gram_partial_cap = 0;

static void ensureGramScratch(size_t elems)
{
  if (elems > g_gram_partial_cap) {
    if (g_gram_partial != NULL) {
      GPU_SAFE_CALL(gpuFree(g_gram_partial));
    }
    GPU_SAFE_CALL(gpuMalloc((void **)&g_gram_partial, elems * sizeof(double)));
    g_gram_partial_cap = elems;
  }
}

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
  if (g_scalar != NULL) {
    GPU_SAFE_CALL(gpuFree(g_scalar));
    g_scalar = NULL;
  }
  if (g_gram_partial != NULL) {
    GPU_SAFE_CALL(gpuFree(g_gram_partial));
    g_gram_partial     = NULL;
    g_gram_partial_cap = 0;
  }
}

/* CGS2 projection: coefs[0:m] = E[:,0:m]^T * e[:,k]. Rows split over
 * blockIdx.y into PROJ_CHUNKS partial sums, reduced by kernel_proj_reduce
 * (deterministic, unlike an atomicAdd fan-in). */
__global__ void kernel_proj(
    CG_UINT nr, const V_ELE *e, int nc, int k, int m, V_ELE *partial, CG_UINT rowsPerChunk)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m)
    return;

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

/* CGS2 subtract: e[:,k] -= E[:,0:m] * coefs */
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

  /* Syncs stay unconditional; only the accumulate is predicated on row < nr. */
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

__device__ static inline void atomicAddV(V_ELE *dst, V_ELE v)
{
#ifdef USE_COMPLEX
  atomicAdd(&dst->real(), VREAL(v));
  atomicAdd(&dst->imag(), VIMAG(v));
#else
  atomicAdd(dst, v);
#endif
}

// Column 2-norm squared of e[:,k]
__global__ void kernel_col_dot(CG_UINT nr, const V_ELE *e, int nc, int k, V_ELE *result)
{
  using BlockReduce = gpucub::BlockReduce<V_ELE, LIN_THREADS>;
  __shared__ typename BlockReduce::TempStorage tmp;

  CG_UINT r = blockIdx.x * blockDim.x + threadIdx.x;
  V_ELE v   = VCONST(0, 0);
  if (r < nr) {
    V_ELE eik = e[(size_t)r * (size_t)nc + (size_t)k];
    v         = eik * eik;
  }

  V_ELE sum = BlockReduce(tmp).Sum(v);
  if (threadIdx.x == 0) {
    atomicAddV(result, sum);
  }
}

/* Scale e[:,k] by inv; compact to column mDst if different */
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

/* Repack accepted columns from stride nc to stride m. Out-of-place: an
 * in-place parallel pass would race (row r's write window overlaps earlier
 * rows' source windows). */
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

/* Rayleigh-Ritz projection H = Y^T (A Y), m x m.
 *
 * Two kernels: kernel_gram_partial splits the rows into gridDim.z chunks so
 * the (few) m/GRAM_TILE tiles of a small m don't leave the GPU nearly idle
 * — with m=64 a single-pass tile grid is only 10 blocks streaming 2 GB
 * serially. Only tiles on or above the diagonal run; each writes its chunk
 * partial to partial[chunk][i][j]. kernel_gram_reduce sums the chunks in a
 * fixed order (deterministic) and writes both H[i,j] and H[j,i] from one
 * accumulator so H is exactly symmetric, which jacobiEigen assumes. */
__global__ void kernel_gram_partial(CG_UINT nr,
    int m,
    const V_ELE *Ye,
    const V_ELE *AYe,
    double *partial,
    CG_UINT rowsPerChunk)
{
  __shared__ V_ELE As[GRAM_TILE][GRAM_TILE];
  __shared__ V_ELE Bs[GRAM_TILE][GRAM_TILE];

  /* Below-diagonal tile: uniform across the block, so early return
   * before __syncthreads is safe. */
  if (blockIdx.x > blockIdx.y)
    return;

  int i       = blockIdx.x * GRAM_TILE + threadIdx.x;
  int j       = blockIdx.y * GRAM_TILE + threadIdx.y;
  int ai      = blockIdx.x * GRAM_TILE + threadIdx.x;
  int bj      = blockIdx.y * GRAM_TILE + threadIdx.x;

  CG_UINT rs  = (CG_UINT)blockIdx.z * rowsPerChunk;
  CG_UINT re  = rs + rowsPerChunk;
  if (re > nr)
    re = nr;

  double acc = 0.0;
  for (CG_UINT r0 = rs; r0 < re; r0 += GRAM_TILE) {
    CG_UINT r = r0 + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (r < re && ai < m) ? Ye[(size_t)r * (size_t)m + (size_t)ai] : VCONST(0, 0);
    Bs[threadIdx.y][threadIdx.x] =
        (r < re && bj < m) ? AYe[(size_t)r * (size_t)m + (size_t)bj] : VCONST(0, 0);
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

__global__ void kernel_gram_reduce(int m, int nChunks, const double *partial, double *H)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i >= m || i > j)
    return;

  double s = 0.0;
  for (int c = 0; c < nChunks; c++) {
    s += partial[((size_t)c * (size_t)m + (size_t)i) * (size_t)m + (size_t)j];
  }
  H[(size_t)i * (size_t)m + (size_t)j] = s;
  H[(size_t)j * (size_t)m + (size_t)i] = s;
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
  int tiles      = (m + GRAM_TILE - 1) / GRAM_TILE;
  int upperTiles = tiles * (tiles + 1) / 2;

  /* Enough row chunks for >= GRAM_MIN_BLOCKS resident blocks, capped so the
   * partial buffer (nChunks * m * m doubles) stays modest for wide blocks. */
  int nChunks = (GRAM_MIN_BLOCKS + upperTiles - 1) / upperTiles;
  if (nChunks > GRAM_MAX_CHUNKS)
    nChunks = GRAM_MAX_CHUNKS;
  size_t mm    = (size_t)m * (size_t)m;
  size_t capEl = GRAM_PARTIAL_MAX_BYTES / sizeof(double);
  while (nChunks > 1 && (size_t)nChunks * mm > capEl)
    nChunks--;
  /* Whole tiles per chunk: rows can't be more chunks than tile-rows. */
  CG_UINT tileRows = (nr + GRAM_TILE - 1) / GRAM_TILE;
  if ((CG_UINT)nChunks > tileRows)
    nChunks = (int)tileRows;
  if (nChunks < 1)
    nChunks = 1;
  CG_UINT rowsPerChunk = ((tileRows + nChunks - 1) / nChunks) * GRAM_TILE;

  ensureGramScratch((size_t)nChunks * mm);

  kernel_gram_partial<<<dim3(tiles, tiles, nChunks), dim3(GRAM_TILE, GRAM_TILE)>>>(
      nr, m, Ye, AYe, g_gram_partial, rowsPerChunk);
  kernel_gram_reduce<<<dim3((m + LIN_THREADS - 1) / LIN_THREADS, m), LIN_THREADS>>>(
      m, nChunks, g_gram_partial, H);
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

extern "C" int gpu_orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol)
{
  if (nc < 1 || nr == 0) {
    return 0;
  }

  NVTX_RANGE_PUSH_C("gpu.orthoMGS", NVTX_C_ORTHO);

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

    GPU_SAFE_CALL(gpuMemsetAsync(g_scalar, 0, sizeof(V_ELE), 0));
    int colBlocks = (int)((nr + LIN_THREADS - 1) / LIN_THREADS);
    kernel_col_dot<<<colBlocks, LIN_THREADS>>>(nr, e, nc, k, g_scalar);

    V_ELE nrm2;
    // Sync point 
    GPU_SAFE_CALL(gpuMemcpy(&nrm2, g_scalar, sizeof(V_ELE), gpuMemcpyDeviceToHost));

    double nrm = sqrt(asReal(nrm2));
    if (nrm < tol) {
      continue; /* linearly dependent: drop */
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
  NVTX_RANGE_POP();

  return m;
}
