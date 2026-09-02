/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * ChebFD search-space streaming: the matrix is device-resident, the dense
 * blocks Y / AY live in pinned host memory and are pulled through the
 * device in column sub-blocks (matrix passes) or row chunks (dense passes),
 * double-buffered on copyStream against computeStream. See
 * cuda_vector_stream.h for the layout and cuda_kernels.h for the C API.
 *
 * Every entry point looks synchronous to its caller (both streams drained
 * on return) so the host-side section timers keep their meaning.
 */
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "cuda_vector_stream.h"
#include "gpu_backend.h"
#include "nvtx_marker.h"

#define GRAM_TILE 16
#define LIN_THREADS 256
#define UPD_COLS 32
#define UPD_ROWS 8
#define RES_COLS 32
#define RES_ROWS 8
#define DEFAULT_CHUNK_BYTES ((size_t)128 << 20)
#define GRAM_SUBS 256 /* row sub-ranges per Gram chunk launch (upper bound) */
#define GRAM_PARTIAL_MAX_BYTES ((size_t)256 << 20)

/* Row sub-ranges for one Gram launch of an m x m result: as many as
 * GRAM_SUBS, but never more partial-buffer bytes than the cap (wide
 * blocks, e.g. NS in the thousands, would otherwise need gigabytes). */
static int gramSubs(int m)
{
  size_t mm    = (size_t)m * (size_t)m;
  size_t capEl = GRAM_PARTIAL_MAX_BYTES / sizeof(double);
  int nSub     = GRAM_SUBS;
  while (nSub > 1 && (size_t)nSub * mm > capEl) {
    nSub--;
  }
  return nSub;
}

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

/* ------------------------------------------------------------------ */
/*  Prefetch of the managed matrix                                     */
/* ------------------------------------------------------------------ */

/* cudaMemPrefetchAsync returns cudaErrorInvalidValue for memory that is
 * not managed (pinned or plain cudaMalloc); the buffer then already lives
 * where its allocator put it, so only genuine errors abort. */
static void prefetchRange(const void *p, size_t bytes, int dev)
{
  if (p == NULL || bytes == 0) {
    return;
  }
  gpuError_t rc = gpuMemPrefetch(p, bytes, dev);
  if (rc == GCXX_RUNTIME_BACKEND(ErrorInvalidValue)) {
    (void)GCXX_RUNTIME_BACKEND(GetLastError)();
    return;
  }
  GPU_SAFE_CALL(rc);
}

extern "C" void gpu_matrix_prefetch(const Matrix *m)
{
  if (m == NULL) {
    return;
  }
  NVTX_RANGE_PUSH_C("gpu.matrixPrefetch", NVTX_C_STREAM);
  int dev = 0;
  GPU_SAFE_CALL(gpuGetDevice(&dev));
#ifdef SCS
  prefetchRange(m->val, (size_t)m->nElems * sizeof(V_ELE), dev);
  prefetchRange(m->colInd, (size_t)m->nElems * sizeof(CG_UINT), dev);
  prefetchRange(m->chunkPtr, ((size_t)m->nChunks + 1) * sizeof(CG_UINT), dev);
  prefetchRange(m->chunkLens, (size_t)m->nChunks * sizeof(CG_UINT), dev);
#elif defined(CRS)
  prefetchRange(m->val, (size_t)m->nnz * sizeof(V_ELE), dev);
  prefetchRange(m->colInd, (size_t)m->nnz * sizeof(CG_UINT), dev);
  prefetchRange(m->rowPtr, ((size_t)m->nr + 1) * sizeof(CG_UINT), dev);
#endif
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

/* ------------------------------------------------------------------ */
/*  Kernels                                                           */
/* ------------------------------------------------------------------ */

/* w = a*x + b*y + c*z over n contiguous elements. */
__global__ void kernel_vs_axpby3(size_t n,
    V_ELE a,
    const V_ELE *x,
    V_ELE b,
    const V_ELE *y,
    V_ELE c,
    const V_ELE *z,
    V_ELE *w)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    w[i] = a * x[i] + b * y[i] + c * z[i];
  }
}

/* partial[sub][i][j] = sum_{r in sub-range} A[r,i] * B[r,j] for i <= j over
 * one row chunk of two row-major blocks with independent leading dims.
 * Same shared-memory tiling as kernel_gram_partial in cuda_chebfd_dense.cu. */
__global__ void kernel_vs_gram_chunk(CG_UINT rows,
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

/* G[i][j] += sum_sub partial[sub][i][j], mirrored to G[j][i]: fixed
 * summation order (deterministic) and exactly symmetric. */
__global__ void kernel_vs_gram_accum(int m, int nSub, const double *partial, double *G)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i >= m || i > j)
    return;
  double s = 0.0;
  for (int c = 0; c < nSub; c++) {
    s += partial[((size_t)c * (size_t)m + (size_t)i) * (size_t)m + (size_t)j];
  }
  double v = G[(size_t)i * (size_t)m + (size_t)j] + s;
  G[(size_t)i * (size_t)m + (size_t)j] = v;
  G[(size_t)j * (size_t)m + (size_t)i] = v;
}

/* Out[r, jo] = sum_i In[r, i] * B[i, jo]  (In: rows x m, ld ldIn; B: m x
 * mOut row-major; Out: rows x mOut, ld mOut). Threads run over output
 * columns so B reads coalesce and each In element is a warp broadcast. */
__global__ void kernel_vs_block_update(CG_UINT rows,
    int m,
    const V_ELE *In,
    CG_UINT ldIn,
    const double *B,
    int mOut,
    V_ELE *Out)
{
  CG_UINT r = (CG_UINT)blockIdx.x * blockDim.y + threadIdx.y;
  int jo    = blockIdx.y * blockDim.x + threadIdx.x;
  if (r >= rows || jo >= mOut)
    return;
  const V_ELE *in = In + (size_t)r * ldIn;
  double acc      = 0.0;
  for (int i = 0; i < m; i++) {
    acc += asReal(in[i]) * B[(size_t)i * (size_t)mOut + (size_t)jo];
  }
  Out[(size_t)r * (size_t)mOut + (size_t)jo] = VCONST(acc, 0);
}

/* Squared Ritz residual partials over one row chunk: for each selected
 * pair k = sel[t], v_r = sum_j (AY[r,j] - eval_k Y[r,j]) evec[j,k];
 * partial[block][t] = sum over the block's rows of v_r^2. */
__global__ void kernel_vs_ritz_chunk(CG_UINT rows,
    int m,
    const V_ELE *Y,
    const V_ELE *AY,
    const double *evec,
    const double *eval,
    const int *sel,
    int nsel,
    double *partial)
{
  __shared__ double sv[RES_ROWS][RES_COLS];
  CG_UINT r = (CG_UINT)blockIdx.x * RES_ROWS + threadIdx.y;
  int t     = blockIdx.y * RES_COLS + threadIdx.x;
  double v  = 0.0;
  if (r < rows && t < nsel) {
    int k       = sel[t];
    double lam  = eval[k];
    size_t base = (size_t)r * (size_t)m;
    for (int j = 0; j < m; j++) {
      double e = evec[(size_t)j * (size_t)m + (size_t)k];
      v += (asReal(AY[base + j]) - lam * asReal(Y[base + j])) * e;
    }
  }
  sv[threadIdx.y][threadIdx.x] = v * v;
  __syncthreads();
  if (threadIdx.y == 0 && t < nsel) {
    double s = 0.0;
    for (int y = 0; y < RES_ROWS; y++) {
      s += sv[y][threadIdx.x];
    }
    partial[(size_t)blockIdx.x * (size_t)nsel + (size_t)t] = s;
  }
}

/* res2[t] += sum_block partial[block][t]. One block per selected pair;
 * the threads stride over the row blocks and a fixed-shape shared
 * reduction folds them, so the order is deterministic. */
__global__ void kernel_vs_ritz_accum(int nsel, CG_UINT nBlocks, const double *partial, double *res2)
{
  __shared__ double sh[LIN_THREADS];
  int t    = blockIdx.x;
  double s = 0.0;
  for (CG_UINT b = threadIdx.x; b < nBlocks; b += blockDim.x) {
    s += partial[(size_t)b * (size_t)nsel + (size_t)t];
  }
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int w = LIN_THREADS / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) {
      sh[threadIdx.x] += sh[threadIdx.x + w];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    res2[t] += sh[0];
  }
}

/* ------------------------------------------------------------------ */
/*  Context                                                           */
/* ------------------------------------------------------------------ */

static void ensurePartial(GpuVectorStream *s, size_t elems)
{
  if (elems > s->partialCap) {
    if (s->partial != NULL) {
      GPU_SAFE_CALL(gpuFree(s->partial));
    }
    GPU_SAFE_CALL(gpuMalloc((void **)&s->partial, elems * sizeof(double)));
    s->partialCap = elems;
  }
}

extern "C" GpuVectorStream *gpu_vstream_init(
    const Matrix *A, int NS, int nb, size_t chunkBytes, int verbose)
{
  if (A == NULL || NS < 1) {
    return NULL;
  }
  GpuVectorStream *s = (GpuVectorStream *)calloc(1, sizeof(GpuVectorStream));
  if (s == NULL) {
    return NULL;
  }
  s->A  = A;
  s->nr = A->nr;
#ifdef SCS
  s->vecRows = A->nrPadded;
#else
  s->vecRows = A->nr;
#endif
  s->NS = NS;
  if (nb <= 0 || nb > NS) {
    nb = NS;
  }
  s->nb = nb;

  if (chunkBytes == 0) {
    chunkBytes = DEFAULT_CHUNK_BYTES;
  }
  size_t rowBytes = (size_t)NS * sizeof(V_ELE);
  CG_UINT rows    = (CG_UINT)(chunkBytes / rowBytes);
  rows            = (rows / GRAM_TILE) * GRAM_TILE;
  if (rows < GRAM_TILE) {
    rows = GRAM_TILE;
  }
  if (rows > s->vecRows) {
    rows = ((s->vecRows + GRAM_TILE - 1) / GRAM_TILE) * GRAM_TILE;
  }
  s->chunkRows = rows;

  NVTX_RANGE_PUSH_C("gpu.vstreamInit", NVTX_C_STREAM);
  GPU_SAFE_CALL(gpuStreamCreate(&s->computeStream));
  GPU_SAFE_CALL(gpuStreamCreate(&s->copyStream));

  size_t colBytes = (size_t)s->vecRows * (size_t)nb * sizeof(V_ELE);
  size_t rowChunk = (size_t)s->chunkRows * (size_t)NS * sizeof(V_ELE);
  for (int k = 0; k < GPU_VSTREAM_NSLOT; k++) {
    GPU_SAFE_CALL(gpuEventCreate(&s->computeDone[k]));
    GPU_SAFE_CALL(gpuEventCreate(&s->copyDone[k]));
    GPU_SAFE_CALL(gpuEventRecord(s->computeDone[k], s->computeStream));
    GPU_SAFE_CALL(gpuEventRecord(s->copyDone[k], s->copyStream));

    GPU_SAFE_CALL(gpuMalloc((void **)&s->X[k], colBytes));
    /* Columns beyond a partial last sub-block are never written by the
     * kernels; zeroed once so the full-width axpby stays finite. */
    GPU_SAFE_CALL(gpuMemset(s->X[k], 0, colBytes));

    GPU_SAFE_CALL(gpuMalloc((void **)&s->rA[k], rowChunk));
    GPU_SAFE_CALL(gpuMalloc((void **)&s->rB[k], rowChunk));
    GPU_SAFE_CALL(gpuMalloc((void **)&s->rO[k], rowChunk));
  }
  GPU_SAFE_CALL(gpuMalloc((void **)&s->U, colBytes));
  GPU_SAFE_CALL(gpuMalloc((void **)&s->W, colBytes));
  GPU_SAFE_CALL(gpuMemset(s->U, 0, colBytes));
  GPU_SAFE_CALL(gpuMemset(s->W, 0, colBytes));
  size_t nn = (size_t)NS * (size_t)NS;
  GPU_SAFE_CALL(gpuMalloc((void **)&s->G, nn * sizeof(double)));
  GPU_SAFE_CALL(gpuMalloc((void **)&s->B, nn * sizeof(double)));
  GPU_SAFE_CALL(gpuMalloc((void **)&s->eval, (size_t)NS * sizeof(double)));
  GPU_SAFE_CALL(gpuMalloc((void **)&s->sel, (size_t)NS * sizeof(int)));
  GPU_SAFE_CALL(gpuMalloc((void **)&s->res2, (size_t)NS * sizeof(double)));
  ensurePartial(s, (size_t)gramSubs(NS) * nn);
  GPU_SAFE_CALL(gpuDeviceSynchronize());

  double mib = 1.0 / (1024.0 * 1024.0);
  if (verbose) {
    printf("Search-space streaming: %d columns/sub-block (%d sub-blocks at NS=%d), "
           "%u rows/chunk; device scratch %.1f MiB sub-blocks + %.1f MiB chunks\n",
        nb,
        (NS + nb - 1) / nb,
        NS,
        (unsigned)s->chunkRows,
        4.0 * (double)colBytes * mib,
        6.0 * (double)rowChunk * mib);
  }

  /* Oversubscription is silent: the managed matrix still "fits" once the
   * scratch has taken the device, but its pages then thrash between host
   * and device and the filter runs ~6x slower per row. Warn while there
   * is still a knob to turn (a smaller cheb_nb halves the scratch). */
  size_t freeB = 0, totalB = 0;
  GPU_SAFE_CALL(GCXX_RUNTIME_BACKEND(MemGetInfo)(&freeB, &totalB));
  size_t matrixB = 0;
#ifdef SCS
  matrixB = (size_t)A->nElems * (sizeof(V_ELE) + sizeof(CG_UINT)) +
            ((size_t)A->nChunks * 2 + 1) * sizeof(CG_UINT);
#else
  matrixB = (size_t)A->nnz * (sizeof(V_ELE) + sizeof(CG_UINT)) +
            ((size_t)A->nr + 1) * sizeof(CG_UINT);
#endif
  /* freeB was sampled after this context's scratch was allocated; the
   * matrix may or may not have been prefetched yet (managed), so compare
   * against the total budget instead. */
  size_t scratchB = 4 * colBytes + 6 * rowChunk;
  if (matrixB + scratchB > (size_t)(0.92 * (double)totalB)) {
    fprintf(stderr,
        "Warning: ChebFD device footprint %.1f GiB (matrix %.1f + streaming scratch "
        "%.1f) is at/over the %.1f GiB device memory; the managed matrix will "
        "thrash. Reduce cheb_nb (%d -> %d halves the scratch) or the problem size.\n",
        (double)(matrixB + scratchB) * mib / 1024.0,
        (double)matrixB * mib / 1024.0,
        (double)scratchB * mib / 1024.0,
        (double)totalB * mib / 1024.0,
        nb,
        nb > 1 ? nb / 2 : 1);
  }
  NVTX_RANGE_POP();
  return s;
}

extern "C" void gpu_vstream_free(GpuVectorStream *s)
{
  if (s == NULL) {
    return;
  }
  GPU_SAFE_CALL(gpuStreamSynchronize(s->computeStream));
  GPU_SAFE_CALL(gpuStreamSynchronize(s->copyStream));
  for (int k = 0; k < GPU_VSTREAM_NSLOT; k++) {
    GPU_SAFE_CALL(gpuFree(s->X[k]));
    GPU_SAFE_CALL(gpuFree(s->rA[k]));
    GPU_SAFE_CALL(gpuFree(s->rB[k]));
    GPU_SAFE_CALL(gpuFree(s->rO[k]));
    GPU_SAFE_CALL(gpuEventDestroy(s->computeDone[k]));
    GPU_SAFE_CALL(gpuEventDestroy(s->copyDone[k]));
  }
  GPU_SAFE_CALL(gpuFree(s->U));
  GPU_SAFE_CALL(gpuFree(s->W));
  if (s->partial != NULL) {
    GPU_SAFE_CALL(gpuFree(s->partial));
  }
  GPU_SAFE_CALL(gpuFree(s->G));
  GPU_SAFE_CALL(gpuFree(s->B));
  GPU_SAFE_CALL(gpuFree(s->eval));
  GPU_SAFE_CALL(gpuFree(s->sel));
  GPU_SAFE_CALL(gpuFree(s->res2));
  GPU_SAFE_CALL(gpuStreamDestroy(s->computeStream));
  GPU_SAFE_CALL(gpuStreamDestroy(s->copyStream));
  free(s);
}

extern "C" void gpu_vstream_stats(const GpuVectorStream *s,
    size_t *h2dBytes,
    size_t *d2hBytes,
    unsigned long long *colPasses,
    unsigned long long *rowPasses)
{
  if (h2dBytes)
    *h2dBytes = s->h2dBytes;
  if (d2hBytes)
    *d2hBytes = s->d2hBytes;
  if (colPasses)
    *colPasses = s->colPasses;
  if (rowPasses)
    *rowPasses = s->rowPasses;
}

/* ------------------------------------------------------------------ */
/*  Column sub-block pipeline                                          */
/* ------------------------------------------------------------------ */

/* Compute callback: slot k holds columns [v0, v0+w) of the block in X[k]
 * (ld = nb); the result to write back must end up in out[k]. */
typedef void (*ColBlockFn)(GpuVectorStream *s, int k, CG_UINT w, void *ctx);

static void colBlockPipeline(GpuVectorStream *s,
    const V_ELE *inH,
    CG_UINT ncols,
    V_ELE *outH,
    V_ELE **outBuf, /* per-slot device buffer holding the result */
    ColBlockFn fn,
    void *ctx)
{
  const size_t hostPitch = (size_t)ncols * sizeof(V_ELE);
  const size_t devPitch  = (size_t)s->nb * sizeof(V_ELE);
  const CG_UINT nb       = (CG_UINT)s->nb;
  const int nBlocks      = (int)((ncols + nb - 1) / nb);

  for (int i = 0; i < nBlocks; i++) {
    int k       = i % GPU_VSTREAM_NSLOT;
    CG_UINT v0  = (CG_UINT)i * nb;
    CG_UINT w   = (nb < ncols - v0) ? nb : ncols - v0;
    size_t wB   = (size_t)w * sizeof(V_ELE);

    /* Slot k is free once its previous compute finished; first drain its
     * previous result to the host, then load the next input. */
    GPU_SAFE_CALL(gpuStreamWaitEvent(s->copyStream, s->computeDone[k], 0));
    if (outH != NULL && i >= GPU_VSTREAM_NSLOT) {
      int ip      = i - GPU_VSTREAM_NSLOT;
      CG_UINT vp  = (CG_UINT)ip * nb;
      CG_UINT wp  = (nb < ncols - vp) ? nb : ncols - vp;
      GPU_SAFE_CALL(gpuMemcpy2DAsync(outH + vp,
          hostPitch,
          outBuf[k],
          devPitch,
          (size_t)wp * sizeof(V_ELE),
          (size_t)s->vecRows,
          gpuMemcpyDeviceToHost,
          s->copyStream));
      s->d2hBytes += (size_t)wp * sizeof(V_ELE) * (size_t)s->vecRows;
    }
    GPU_SAFE_CALL(gpuMemcpy2DAsync(s->X[k],
        devPitch,
        inH + v0,
        hostPitch,
        wB,
        (size_t)s->vecRows,
        gpuMemcpyHostToDevice,
        s->copyStream));
    s->h2dBytes += wB * (size_t)s->vecRows;
    GPU_SAFE_CALL(gpuEventRecord(s->copyDone[k], s->copyStream));

    GPU_SAFE_CALL(gpuStreamWaitEvent(s->computeStream, s->copyDone[k], 0));
    fn(s, k, w, ctx);
    GPU_SAFE_CALL(gpuEventRecord(s->computeDone[k], s->computeStream));
  }

  /* Drain the last NSLOT results. */
  if (outH != NULL) {
    int first = (nBlocks > GPU_VSTREAM_NSLOT) ? nBlocks - GPU_VSTREAM_NSLOT : 0;
    for (int i = first; i < nBlocks; i++) {
      int k      = i % GPU_VSTREAM_NSLOT;
      CG_UINT v0 = (CG_UINT)i * nb;
      CG_UINT w  = (nb < ncols - v0) ? nb : ncols - v0;
      GPU_SAFE_CALL(gpuStreamWaitEvent(s->copyStream, s->computeDone[k], 0));
      GPU_SAFE_CALL(gpuMemcpy2DAsync(outH + v0,
          hostPitch,
          outBuf[k],
          devPitch,
          (size_t)w * sizeof(V_ELE),
          (size_t)s->vecRows,
          gpuMemcpyDeviceToHost,
          s->copyStream));
      s->d2hBytes += (size_t)w * sizeof(V_ELE) * (size_t)s->vecRows;
    }
  }
  GPU_SAFE_CALL(gpuStreamSynchronize(s->computeStream));
  GPU_SAFE_CALL(gpuStreamSynchronize(s->copyStream));
  s->colPasses++;
}

/* --- filter: X_sub <- p(A) X_sub over all degrees ------------------- */
typedef struct {
  V_ELE alpha, beta;
  const double *gc;
  int Np;
} FilterCtx;

static void filterSubBlock(GpuVectorStream *s, int k, CG_UINT w, void *vctx)
{
  const FilterCtx *c = (const FilterCtx *)vctx;
  gpuStream_t st     = s->computeStream;
  const Matrix *A    = s->A;
  CG_UINT ld         = (CG_UINT)s->nb;
  V_ELE *X = s->X[k], *U = s->U, *Wb = s->W;
  V_ELE two = VCONST(2.0, 0), mone = VCONST(-1.0, 0), zero = VCONST(0, 0);

  /* u = (alpha A + beta) x = T_1 x */
  gpu_launch_chebfd(A, st, X, c->alpha, X, c->beta, NULL, zero, U, zero, NULL, w, ld);
  /* w = 2 (alpha A + beta) u - x = T_2 x */
  gpu_launch_chebfd(A, st, U, two * c->alpha, U, two * c->beta, X, mone, Wb, zero, NULL, w, ld);
  /* x = gc0 x + gc1 u + gc2 w */
  size_t n   = (size_t)s->vecRows * (size_t)ld;
  int blocks = (int)((n + LIN_THREADS - 1) / LIN_THREADS);
  kernel_vs_axpby3<<<blocks, LIN_THREADS, 0, st>>>(n,
      VCONST(c->gc[0], 0),
      X,
      VCONST(c->gc[1], 0),
      U,
      VCONST(c->gc[2], 0),
      Wb,
      X);
  /* Remaining degrees; invariant U = T_{n-2}, W = T_{n-1}. U <- T_n in
   * place (row-local: y aliases q) and x += gc[n] T_n in the same pass. */
  for (int nn = 3; nn <= c->Np; nn++) {
    gpu_launch_chebfd(A,
        st,
        Wb,
        two * c->alpha,
        Wb,
        two * c->beta,
        U,
        mone,
        U,
        VCONST(c->gc[nn], 0),
        X,
        w,
        ld);
    V_ELE *t = U;
    U        = Wb;
    Wb       = t;
  }
}

extern "C" void gpu_vstream_filter(GpuVectorStream *s,
    V_ELE *Yh,
    int nc,
    double alpha,
    double beta,
    const double *gc,
    int Np)
{
  NVTX_RANGE_PUSH_C("gpu.vstream.filter", NVTX_C_FILTER);
  FilterCtx c;
  c.alpha = VCONST(alpha, 0);
  c.beta  = VCONST(beta, 0);
  c.gc    = gc;
  c.Np    = Np;
  colBlockPipeline(s, Yh, (CG_UINT)nc, Yh, s->X, filterSubBlock, &c);
  NVTX_RANGE_POP();
}

/* --- AY = A * Y ----------------------------------------------------- */
/* Output alternates between U and W per slot: slot k's result must survive
 * until its D2H copy has run, which happens while slot k^1 computes. */
static void spmmvSubBlock(GpuVectorStream *s, int k, CG_UINT w, void *ctx)
{
  V_ELE **out = (V_ELE **)ctx;
  gpu_launch_spmmv(s->A, s->computeStream, s->X[k], out[k], w, (CG_UINT)s->nb);
}

extern "C" void gpu_vstream_spmmv(GpuVectorStream *s, const V_ELE *Yh, V_ELE *AYh, int nc)
{
  NVTX_RANGE_PUSH_C("gpu.vstream.spmmv", NVTX_C_MATVEC);
  V_ELE *out[GPU_VSTREAM_NSLOT] = { s->U, s->W };
  colBlockPipeline(s, Yh, (CG_UINT)nc, AYh, out, spmmvSubBlock, out);
  NVTX_RANGE_POP();
}

/* ------------------------------------------------------------------ */
/*  Row-chunk pipeline                                                 */
/* ------------------------------------------------------------------ */

/* Compute callback: slot k holds rows [r0, r0+rows) of A (and B if given)
 * in rA[k] / rB[k] with leading dimension ld; results for write-back go
 * to rO[k] with leading dimension ldOut. */
typedef void (*RowChunkFn)(GpuVectorStream *s, int k, CG_UINT r0, CG_UINT rows, void *ctx);

static void rowChunkPipeline(GpuVectorStream *s,
    const V_ELE *Ah,
    const V_ELE *Bh,
    CG_UINT ld,
    V_ELE *outH,
    CG_UINT ldOut,
    RowChunkFn fn,
    void *ctx)
{
  const CG_UINT total = s->vecRows;
  const CG_UINT cr    = s->chunkRows;
  const int nChunks   = (int)((total + cr - 1) / cr);

  for (int i = 0; i < nChunks; i++) {
    int k        = i % GPU_VSTREAM_NSLOT;
    CG_UINT r0   = (CG_UINT)i * cr;
    CG_UINT rows = (cr < total - r0) ? cr : total - r0;

    GPU_SAFE_CALL(gpuStreamWaitEvent(s->copyStream, s->computeDone[k], 0));
    if (outH != NULL && i >= GPU_VSTREAM_NSLOT) {
      int ip       = i - GPU_VSTREAM_NSLOT;
      CG_UINT rp   = (CG_UINT)ip * cr;
      CG_UINT rowp = (cr < total - rp) ? cr : total - rp;
      size_t bytes = (size_t)rowp * (size_t)ldOut * sizeof(V_ELE);
      GPU_SAFE_CALL(gpuMemcpyAsync(outH + (size_t)rp * ldOut,
          s->rO[k],
          bytes,
          gpuMemcpyDeviceToHost,
          s->copyStream));
      s->d2hBytes += bytes;
    }
    size_t inBytes = (size_t)rows * (size_t)ld * sizeof(V_ELE);
    GPU_SAFE_CALL(gpuMemcpyAsync(
        s->rA[k], Ah + (size_t)r0 * ld, inBytes, gpuMemcpyHostToDevice, s->copyStream));
    s->h2dBytes += inBytes;
    if (Bh != NULL) {
      GPU_SAFE_CALL(gpuMemcpyAsync(
          s->rB[k], Bh + (size_t)r0 * ld, inBytes, gpuMemcpyHostToDevice, s->copyStream));
      s->h2dBytes += inBytes;
    }
    GPU_SAFE_CALL(gpuEventRecord(s->copyDone[k], s->copyStream));

    GPU_SAFE_CALL(gpuStreamWaitEvent(s->computeStream, s->copyDone[k], 0));
    fn(s, k, r0, rows, ctx);
    GPU_SAFE_CALL(gpuEventRecord(s->computeDone[k], s->computeStream));
  }

  if (outH != NULL) {
    int first = (nChunks > GPU_VSTREAM_NSLOT) ? nChunks - GPU_VSTREAM_NSLOT : 0;
    for (int i = first; i < nChunks; i++) {
      int k        = i % GPU_VSTREAM_NSLOT;
      CG_UINT r0   = (CG_UINT)i * cr;
      CG_UINT rows = (cr < total - r0) ? cr : total - r0;
      size_t bytes = (size_t)rows * (size_t)ldOut * sizeof(V_ELE);
      GPU_SAFE_CALL(gpuStreamWaitEvent(s->copyStream, s->computeDone[k], 0));
      GPU_SAFE_CALL(gpuMemcpyAsync(outH + (size_t)r0 * ldOut,
          s->rO[k],
          bytes,
          gpuMemcpyDeviceToHost,
          s->copyStream));
      s->d2hBytes += bytes;
    }
  }
  GPU_SAFE_CALL(gpuStreamSynchronize(s->computeStream));
  GPU_SAFE_CALL(gpuStreamSynchronize(s->copyStream));
  s->rowPasses++;
}

/* --- Gram: G = A^T B (B == A for Y^T Y) ----------------------------- */
typedef struct {
  int m;
  int useB;
} GramCtx;

static void gramChunk(GpuVectorStream *s, int k, CG_UINT r0, CG_UINT rows, void *vctx)
{
  (void)r0;
  const GramCtx *c = (const GramCtx *)vctx;
  int m            = c->m;
  int tiles        = (m + GRAM_TILE - 1) / GRAM_TILE;
  CG_UINT tileRows = (rows + GRAM_TILE - 1) / GRAM_TILE;
  int nSub         = gramSubs(m);
  if ((CG_UINT)nSub > tileRows) {
    nSub = (int)tileRows;
  }
  if (nSub < 1) {
    nSub = 1;
  }
  CG_UINT rowsPerSub = ((tileRows + nSub - 1) / nSub) * GRAM_TILE;
  const V_ELE *Bp    = c->useB ? s->rB[k] : s->rA[k];

  kernel_vs_gram_chunk<<<dim3(tiles, tiles, nSub), dim3(GRAM_TILE, GRAM_TILE), 0, s->computeStream>>>(
      rows, m, s->rA[k], (CG_UINT)m, Bp, (CG_UINT)m, s->partial, rowsPerSub);
  kernel_vs_gram_accum<<<dim3((m + LIN_THREADS - 1) / LIN_THREADS, m), LIN_THREADS, 0, s->computeStream>>>(
      m, nSub, s->partial, s->G);
}

extern "C" void gpu_vstream_gram(
    GpuVectorStream *s, const V_ELE *Ah, const V_ELE *Bh, int m, double *Gh)
{
  NVTX_RANGE_PUSH_C("gpu.vstream.gram", NVTX_C_RR);
  size_t mm = (size_t)m * (size_t)m;
  ensurePartial(s, (size_t)gramSubs(m) * mm);
  GPU_SAFE_CALL(gpuMemsetAsync(s->G, 0, mm * sizeof(double), s->computeStream));
  GramCtx c;
  c.m    = m;
  c.useB = (Bh != NULL && Bh != Ah);
  rowChunkPipeline(s, Ah, c.useB ? Bh : NULL, (CG_UINT)m, NULL, 0, gramChunk, &c);
  GPU_SAFE_CALL(gpuMemcpy(Gh, s->G, mm * sizeof(double), gpuMemcpyDeviceToHost));
  NVTX_RANGE_POP();
}

/* --- block update: Y (stride m) <- Y B (m x mOut), written at stride mOut */
typedef struct {
  int m, mOut;
} UpdCtx;

static void updateChunk(GpuVectorStream *s, int k, CG_UINT r0, CG_UINT rows, void *vctx)
{
  (void)r0;
  const UpdCtx *c = (const UpdCtx *)vctx;
  dim3 grid((rows + UPD_ROWS - 1) / UPD_ROWS, (c->mOut + UPD_COLS - 1) / UPD_COLS);
  kernel_vs_block_update<<<grid, dim3(UPD_COLS, UPD_ROWS), 0, s->computeStream>>>(
      rows, c->m, s->rA[k], (CG_UINT)c->m, s->B, c->mOut, s->rO[k]);
}

extern "C" void gpu_vstream_update(
    GpuVectorStream *s, V_ELE *Yh, int m, const double *Bh, int mOut)
{
  NVTX_RANGE_PUSH_C("gpu.vstream.update", NVTX_C_ORTHO);
  GPU_SAFE_CALL(gpuMemcpy(
      s->B, Bh, (size_t)m * (size_t)mOut * sizeof(double), gpuMemcpyHostToDevice));
  UpdCtx c;
  c.m    = m;
  c.mOut = mOut;
  /* In place: chunk i is fully on the device before its rows are
   * overwritten, and the write region [r0*mOut, r1*mOut) never reaches the
   * unread rows >= r1 since mOut <= m. Chunks are issued in ascending row
   * order on one copy stream, which keeps that argument valid with the
   * NSLOT-deep prefetch. */
  rowChunkPipeline(s, Yh, NULL, (CG_UINT)m, Yh, (CG_UINT)mOut, updateChunk, &c);
  NVTX_RANGE_POP();
}

/* --- Ritz residual norms for the selected pairs ---------------------- */
typedef struct {
  int m, nsel;
} ResCtx;

static void ritzChunk(GpuVectorStream *s, int k, CG_UINT r0, CG_UINT rows, void *vctx)
{
  (void)r0;
  const ResCtx *c = (const ResCtx *)vctx;
  CG_UINT nBlocks = (rows + RES_ROWS - 1) / RES_ROWS;
  ensurePartial(s, (size_t)nBlocks * (size_t)c->nsel);
  dim3 grid(nBlocks, (c->nsel + RES_COLS - 1) / RES_COLS);
  kernel_vs_ritz_chunk<<<grid, dim3(RES_COLS, RES_ROWS), 0, s->computeStream>>>(
      rows, c->m, s->rA[k], s->rB[k], s->B, s->eval, s->sel, c->nsel, s->partial);
  kernel_vs_ritz_accum<<<c->nsel, LIN_THREADS, 0, s->computeStream>>>(
      c->nsel, nBlocks, s->partial, s->res2);
}

extern "C" void gpu_vstream_ritzResiduals(GpuVectorStream *s,
    const V_ELE *Yh,
    const V_ELE *AYh,
    int m,
    const double *eval,
    const double *evec,
    const int *sel,
    int nsel,
    double *res2)
{
  if (nsel < 1) {
    return;
  }
  NVTX_RANGE_PUSH_C("gpu.vstream.ritzResiduals", NVTX_C_RESID);
  /* ensurePartial may reallocate inside the pipeline; size it up front for
   * the largest chunk so the pointer stays stable while kernels run. */
  CG_UINT maxBlocks = (s->chunkRows + RES_ROWS - 1) / RES_ROWS;
  ensurePartial(s, (size_t)maxBlocks * (size_t)nsel);
  GPU_SAFE_CALL(gpuMemcpy(
      s->B, evec, (size_t)m * (size_t)m * sizeof(double), gpuMemcpyHostToDevice));
  GPU_SAFE_CALL(gpuMemcpy(s->eval, eval, (size_t)m * sizeof(double), gpuMemcpyHostToDevice));
  GPU_SAFE_CALL(gpuMemcpy(s->sel, sel, (size_t)nsel * sizeof(int), gpuMemcpyHostToDevice));
  GPU_SAFE_CALL(gpuMemset(s->res2, 0, (size_t)nsel * sizeof(double)));
  ResCtx c;
  c.m    = m;
  c.nsel = nsel;
  rowChunkPipeline(s, Yh, AYh, (CG_UINT)m, NULL, 0, ritzChunk, &c);
  GPU_SAFE_CALL(gpuMemcpy(res2, s->res2, (size_t)nsel * sizeof(double), gpuMemcpyDeviceToHost));
  NVTX_RANGE_POP();
}

/* ------------------------------------------------------------------ */
/*  Pinned host blocks                                                 */
/* ------------------------------------------------------------------ */
extern "C" void *gpu_allocate_host(size_t bytes)
{
  void *p = NULL;
  GPU_SAFE_CALL(gpuMallocHost(&p, bytes));
  return p;
}

extern "C" void gpu_free_host(void *p)
{
  if (p != NULL) {
    GPU_SAFE_CALL(gpuFreeHost(p));
  }
}
