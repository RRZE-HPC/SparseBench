/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CUDA_DENSE_KERNELS_CUH_
#define __CUDA_DENSE_KERNELS_CUH_

/* Device helpers shared by the dense ChebFD block kernels
 * (cuda_chebfd_dense.cu) and the search-space streaming
 * (cuda_vector_stream.cu). GPU builds only. */

#include "gpu_backend.h"
#include "vtype.h"

#define GRAM_TILE 16
#define LIN_THREADS 256
/* Row sub-ranges per Gram launch (upper bound), and the cap on the partial
 * buffer they need (wide blocks, e.g. NS in the thousands, would otherwise
 * need gigabytes). */
#define GRAM_SUBS 256
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

/* Grow-only device buffer: (re)allocate *p to hold at least `elems`
 * elements of `elemSize` bytes, tracking the capacity in *capElems. */
static inline void gpuGrowBuffer(void **p, size_t *capElems, size_t elems, size_t elemSize)
{
  if (elems > *capElems) {
    if (*p != NULL) {
      GPU_SAFE_CALL(gpuFree(*p));
    }
    GPU_SAFE_CALL(gpuMalloc(p, elems * elemSize));
    *capElems = elems;
  }
}

/* Row sub-ranges for one Gram launch of an m x m result: as many as
 * GRAM_SUBS, but never more partial-buffer bytes than the cap. */
static inline int gramSubs(int m)
{
  size_t mm    = (size_t)m * (size_t)m;
  size_t capEl = GRAM_PARTIAL_MAX_BYTES / sizeof(double);
  int nSub     = GRAM_SUBS;
  while (nSub > 1 && (size_t)nSub * mm > capEl) {
    nSub--;
  }
  return nSub;
}

/* partial[sub][i][j] = sum_{r in sub-range} A[r,i] * B[r,j] for i <= j over
 * `rows` rows of two row-major blocks with independent leading dims. The
 * rows are split into gridDim.z sub-ranges so the few m/GRAM_TILE tiles of a
 * small m don't leave the GPU nearly idle. Defined in cuda_chebfd_dense.cu. */
__global__ void kernel_gram_chunk(CG_UINT rows,
    int m,
    const V_ELE *A,
    CG_UINT ldA,
    const V_ELE *B,
    CG_UINT ldB,
    double *partial,
    CG_UINT rowsPerSub);

/* G[i][j] (+)= sum_sub partial[sub][i][j], mirrored to G[j][i]: fixed
 * summation order (deterministic) and exactly symmetric, as jacobiEigen
 * assumes. accumulate == 0 overwrites G, != 0 adds to it. */
__global__ void kernel_gram_accum(
    int m, int nSub, const double *partial, double *G, int accumulate);

/* One Gram pass over `rows` rows: G (+)= A^T B. `partial` must hold at least
 * gramSubs(m) * m * m doubles. */
static inline void launchGram(CG_UINT rows,
    int m,
    const V_ELE *A,
    CG_UINT ldA,
    const V_ELE *B,
    CG_UINT ldB,
    double *partial,
    double *G,
    int accumulate,
    gpuStream_t stream)
{
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

  kernel_gram_chunk<<<dim3(tiles, tiles, nSub), dim3(GRAM_TILE, GRAM_TILE), 0, stream>>>(
      rows, m, A, ldA, B, ldB, partial, rowsPerSub);
  kernel_gram_accum<<<dim3((m + LIN_THREADS - 1) / LIN_THREADS, m), LIN_THREADS, 0, stream>>>(
      m, nSub, partial, G, accumulate);
}

#endif /* __CUDA_DENSE_KERNELS_CUH_ */
