/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CUDA_MATRIX_STREAM_H_
#define __CUDA_MATRIX_STREAM_H_

/* Internal header for host-resident matrix streaming (cuda_matrix_stream.cu
 * and the sweep entry points in cuda_spmv_{scs,crs}.cu). GPU builds only:
 * includes gpu_backend.h, so it must never be included from a gcc-compiled
 * translation unit (use the opaque declarations in cuda_kernels.h there).
 *
 * The matrix stays where allocate() put it (host under gpu_alloc explicit /
 * pageable) and is streamed to the device in "parts" — contiguous runs of
 * SCS chunks / CRS rows — through two rotating device buffers, with the H2D
 * copies on copyStream overlapped against the consuming kernels on
 * computeStream. Block vectors stay device-resident throughout: the SpMV
 * gather xin[colInd * ld + vec] reaches arbitrary rows, so only the matrix
 * is partitionable.
 *
 * Both formats describe their units with one monotone array of element
 * offsets (SCS: chunkPtr, units are C-row chunks, storage padded to
 * chunkLens[i]*C; CRS: rowPtr, units are single rows), which is what makes
 * the partition and the sweep below format-independent. */

#include "gpu_backend.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_STREAM_NBUF 2

/* One contiguous run of units, computed once at init. */
typedef struct {
  CG_UINT start;     /* SCS: first chunk; CRS: first row */
  CG_UINT count;     /* units (chunks / rows) in the part */
  CG_UINT elemStart; /* unitPtr[start]: first element of the val/colInd slice */
  CG_UINT elemCount; /* elements in the slice (padded storage for SCS) */
  size_t payloadBytes;
} StreamPart;

/* View of one part resident in device buffer k, handed to the kernel
 * launcher callback. ptr/lens hold absolute element ids (chunkPtr values
 * are NOT rebased), matching the generalized kernels in cuda_spmv_*.cu. */
typedef struct {
  V_ELE *val;
  CG_UINT *colInd;
  CG_UINT *lens; /* SCS only, NULL for CRS */
  CG_UINT *ptr;
  CG_UINT elemBase; /* = unitPtr[start] */
  CG_UINT rowBase;  /* = start * rowsPerUnit */
  CG_UINT count;    /* units in this part */
} GpuPartView;

/* Enqueues the kernels for one part on the given stream (typically a
 * subblock loop over the block-vector columns). Defined per format next
 * to the kernels it launches — cross-TU __global__ launches would need
 * -rdc=true, which the build does not use. */
typedef void (*GpuPartLaunchFn)(const GpuPartView *v, gpuStream_t stream, void *args);

typedef struct GpuMatrixStream {
  /* Host-side matrix arrays and geometry, captured at init (read-only). */
  const V_ELE *h_val;
  const CG_UINT *h_colInd;
  const CG_UINT *h_lens; /* SCS only */
  const CG_UINT *h_ptr;  /* chunkPtr / rowPtr (nUnits+1 monotone) */
  CG_UINT nUnits;        /* nChunks / nr */
  CG_UINT rowsPerUnit;   /* C / 1 */
  CG_UINT C;             /* SCS chunk height (unused for CRS) */

  StreamPart *parts;
  int nParts;

  /* Rotating device buffers, sized for the largest part. */
  V_ELE *d_val[GPU_STREAM_NBUF];
  CG_UINT *d_colInd[GPU_STREAM_NBUF];
  CG_UINT *d_lens[GPU_STREAM_NBUF]; /* SCS only */
  CG_UINT *d_ptr[GPU_STREAM_NBUF];
  size_t bufElems;
  size_t bufUnits;

  gpuStream_t computeStream, copyStream;
  /* copyDone[k] gates compute on buffer k; computeDone[k] gates its
   * refill. Both are re-recorded in program order on their own stream,
   * so a wait always pairs with the most recent record. */
  gpuEvent_t copyDone[GPU_STREAM_NBUF], computeDone[GPU_STREAM_NBUF];
  gpuEvent_t copyStart, copyStop; /* copyStream span of one sweep */

  /* Statistics, accumulated over all sweeps. */
  size_t bytesCopied;
  unsigned long long sweeps;
  double copyMs;
} GpuMatrixStream;

/* Split the matrix into parts of ~partBytes payload and set up buffers,
 * streams and events. Returns NULL on invalid arguments or GPU error
 * (GPU_SAFE_CALL exits on launch errors, so NULL means bad args).
 * partBytes is a byte count so tests can force multi-part sweeps on tiny
 * matrices; the parameter layer converts the gpu_stream_mb knob. */
GpuMatrixStream *gpu_matrix_stream_init(const Matrix *m, size_t partBytes, int verbose);

void gpu_matrix_stream_free(GpuMatrixStream *s);

/* Double-buffered sweep: stream every part in, running launch(v, stream,
 * args) on computeStream as each part arrives. Returns after synchronizing
 * computeStream, so host timers around the caller keep their meaning. */
void gpu_matrix_stream_sweep(GpuMatrixStream *s, GpuPartLaunchFn launch, void *args);

/* cudaMemPrefetchAsync + device sync: one-time pull of a managed block
 * (the ChebFD vector blocks) to the device before the first sweep. */
void gpu_matrix_stream_prefetch(const V_ELE *p, size_t bytes);

void gpu_matrix_stream_stats(const GpuMatrixStream *s,
    size_t *h2dBytes,
    int *nParts,
    unsigned long long *nSweeps,
    double *copyMs);

#ifdef __cplusplus
}
#endif

#endif /* __CUDA_MATRIX_STREAM_H_ */
