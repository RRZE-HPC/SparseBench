/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * Format-independent machinery for host-resident matrix streaming — see
 * cuda_matrix_stream.h. The kernels themselves stay in cuda_spmv_{scs,crs}.cu
 * (cross-TU __global__ launches would need -rdc=true); this file only moves
 * bytes and drives the pipeline.
 *
 * Streams are created with default (blocking) flags on purpose: every other
 * wrapper in the repo launches on the legacy default stream, and blocking
 * streams serialize against it in both directions. Cross-phase ordering
 * (e.g. orthoMGS after a sweep, the next sweep after the ortho repack) is
 * then correct by construction without extra events.
 */
#include "cuda_matrix_stream.h"

#include <stdio.h>
#include <stdlib.h>

#include "cuda_kernels.h"
#include "nvtx_marker.h"

/* Greedy partition of [0, nUnits) into runs of units whose payload
 * (val+colInd slice, plus the lens/ptr index slices) reaches partBytes.
 * A part always keeps at least one unit, so a single unit larger than the
 * target becomes one oversized part rather than being split. */
static StreamPart *partitionUnits(const CG_UINT *unitPtr,
    CG_UINT nUnits,
    int hasLens,
    size_t partBytes,
    int *nPartsOut,
    size_t *maxElemsOut,
    size_t *maxUnitsOut)
{
  StreamPart *parts = NULL;
  int nParts = 0;
  size_t maxElems = 0, maxUnits = 0;
  CG_UINT start = 0;

  while (start < nUnits) {
    CG_UINT end     = start;
    size_t elems    = 0;
    size_t payBytes = 0;
    do {
      size_t unitElems = (size_t)(unitPtr[end + 1] - unitPtr[end]);
      size_t unitPay =
          unitElems * (sizeof(V_ELE) + sizeof(CG_UINT)) /* val + colInd */
          + (hasLens ? sizeof(CG_UINT) : 0)             /* lens */
          + sizeof(CG_UINT);                            /* ptr entry */
      if (end > start && payBytes + unitPay > partBytes)
        break;
      elems += unitElems;
      payBytes += unitPay;
      end++;
    } while (end < nUnits);

    StreamPart *grown =
        (StreamPart *)realloc(parts, (size_t)(nParts + 1) * sizeof(StreamPart));
    if (grown == NULL) {
      free(parts);
      return NULL;
    }
    parts = grown;
    parts[nParts].start        = start;
    parts[nParts].count        = end - start;
    parts[nParts].elemStart    = unitPtr[start];
    parts[nParts].elemCount    = (CG_UINT)elems;
    parts[nParts].payloadBytes = payBytes + sizeof(CG_UINT); /* +1 ptr entry */
    if (elems > maxElems)
      maxElems = elems;
    if ((size_t)(end - start) > maxUnits)
      maxUnits = (size_t)(end - start);
    nParts++;
    start = end;
  }

  *nPartsOut  = nParts;
  *maxElemsOut = maxElems;
  *maxUnitsOut = maxUnits;
  return parts;
}

GpuMatrixStream *gpu_matrix_stream_init(const Matrix *m, size_t partBytes, int verbose)
{
  if (m == NULL || partBytes == 0) {
    return NULL;
  }

  GpuMatrixStream *s = (GpuMatrixStream *)calloc(1, sizeof(GpuMatrixStream));
  if (s == NULL) {
    return NULL;
  }

#ifdef SCS
  s->h_val    = m->val;
  s->h_colInd = m->colInd;
  s->h_lens   = m->chunkLens;
  s->h_ptr    = m->chunkPtr;
  s->nUnits   = m->nChunks;
  s->rowsPerUnit = m->C;
  s->C       = m->C;
#elif defined(CRS)
  s->h_val    = m->val;
  s->h_colInd = m->colInd;
  s->h_lens   = NULL;
  s->h_ptr    = m->rowPtr;
  s->nUnits   = m->nr;
  s->rowsPerUnit = 1;
  s->C       = 1;
#else
  fprintf(stderr, "gpu_matrix_stream_init: unsupported matrix format\n");
  free(s);
  return NULL;
#endif

  s->parts = partitionUnits(s->h_ptr,
      s->nUnits,
      s->h_lens != NULL,
      partBytes,
      &s->nParts,
      &s->bufElems,
      &s->bufUnits);
  if (s->parts == NULL || s->nParts < 1) {
    fprintf(stderr, "gpu_matrix_stream_init: partitioning failed\n");
    free(s);
    return NULL;
  }

  NVTX_RANGE_PUSH_C("gpu.matrixStreamInit", NVTX_C_STREAM);

  GPU_SAFE_CALL(gpuStreamCreate(&s->computeStream));
  GPU_SAFE_CALL(gpuStreamCreate(&s->copyStream));
  for (int k = 0; k < GPU_STREAM_NBUF; k++) {
    GPU_SAFE_CALL(gpuEventCreate(&s->copyDone[k]));
    GPU_SAFE_CALL(gpuEventCreate(&s->computeDone[k]));
    /* Initial record: makes the first round's refill wait well-defined
     * even where a never-recorded event is documented as a no-op (HIP's
     * behaviour there is murkier than CUDA's). */
    GPU_SAFE_CALL(gpuEventRecord(s->computeDone[k], s->computeStream));
    GPU_SAFE_CALL(gpuEventSynchronize(s->computeDone[k]));
  }
  GPU_SAFE_CALL(gpuEventCreate(&s->copyStart));
  GPU_SAFE_CALL(gpuEventCreate(&s->copyStop));

  for (int k = 0; k < GPU_STREAM_NBUF; k++) {
    GPU_SAFE_CALL(gpuMalloc(&s->d_val[k], s->bufElems * sizeof(V_ELE)));
    GPU_SAFE_CALL(gpuMalloc(&s->d_colInd[k], s->bufElems * sizeof(CG_UINT)));
    if (s->h_lens != NULL) {
      GPU_SAFE_CALL(gpuMalloc(&s->d_lens[k], s->bufUnits * sizeof(CG_UINT)));
    }
    GPU_SAFE_CALL(gpuMalloc(&s->d_ptr[k], (s->bufUnits + 1) * sizeof(CG_UINT)));
  }

  if (verbose) {
    printf("Matrix streaming: %d parts (~%.1f MiB target), device buffers "
           "2x%.2f MiB\n",
        s->nParts,
        (double)partBytes / (1024.0 * 1024.0),
        (double)(s->bufElems * (sizeof(V_ELE) + sizeof(CG_UINT))) / (1024.0 * 1024.0));
    if (s->nParts == 1) {
      printf("Matrix streaming: matrix fits one part; no copy/compute overlap "
             "(consider a smaller gpu_stream_mb).\n");
    }
  }

  NVTX_RANGE_POP();
  return s;
}

void gpu_matrix_stream_free(GpuMatrixStream *s)
{
  if (s == NULL) {
    return;
  }
  NVTX_RANGE_PUSH_C("gpu.matrixStreamFree", NVTX_C_STREAM);
  GPU_SAFE_CALL(gpuStreamSynchronize(s->computeStream));
  GPU_SAFE_CALL(gpuStreamSynchronize(s->copyStream));
  for (int k = 0; k < GPU_STREAM_NBUF; k++) {
    GPU_SAFE_CALL(gpuFree(s->d_val[k]));
    GPU_SAFE_CALL(gpuFree(s->d_colInd[k]));
    if (s->h_lens != NULL) {
      GPU_SAFE_CALL(gpuFree(s->d_lens[k]));
    }
    GPU_SAFE_CALL(gpuFree(s->d_ptr[k]));
    GPU_SAFE_CALL(gpuEventDestroy(s->copyDone[k]));
    GPU_SAFE_CALL(gpuEventDestroy(s->computeDone[k]));
  }
  GPU_SAFE_CALL(gpuEventDestroy(s->copyStart));
  GPU_SAFE_CALL(gpuEventDestroy(s->copyStop));
  GPU_SAFE_CALL(gpuStreamDestroy(s->computeStream));
  GPU_SAFE_CALL(gpuStreamDestroy(s->copyStream));
  free(s->parts);
  free(s);
  NVTX_RANGE_POP();
}

void gpu_matrix_stream_sweep(GpuMatrixStream *s, GpuPartLaunchFn launch, void *args)
{
  GPU_SAFE_CALL(gpuEventRecord(s->copyStart, s->copyStream));

  for (int r = 0; r < s->nParts; r++) {
    int k              = r % GPU_STREAM_NBUF;
    const StreamPart *p = &s->parts[r];

    /* Refill buffer k only after the compute that used it before. */
    GPU_SAFE_CALL(
        gpuStreamWaitEvent(s->copyStream, s->computeDone[k], 0));
    GPU_SAFE_CALL(gpuMemcpyAsync(s->d_val[k],
        s->h_val + p->elemStart,
        (size_t)p->elemCount * sizeof(V_ELE),
        gpuMemcpyHostToDevice,
        s->copyStream));
    GPU_SAFE_CALL(gpuMemcpyAsync(s->d_colInd[k],
        s->h_colInd + p->elemStart,
        (size_t)p->elemCount * sizeof(CG_UINT),
        gpuMemcpyHostToDevice,
        s->copyStream));
    if (s->h_lens != NULL) {
      GPU_SAFE_CALL(gpuMemcpyAsync(s->d_lens[k],
          s->h_lens + p->start,
          (size_t)p->count * sizeof(CG_UINT),
          gpuMemcpyHostToDevice,
          s->copyStream));
    }
    GPU_SAFE_CALL(gpuMemcpyAsync(s->d_ptr[k],
        s->h_ptr + p->start,
        ((size_t)p->count + 1) * sizeof(CG_UINT),
        gpuMemcpyHostToDevice,
        s->copyStream));
    GPU_SAFE_CALL(gpuEventRecord(s->copyDone[k], s->copyStream));

    /* Compute on the part as soon as its copy lands. */
    GPU_SAFE_CALL(gpuStreamWaitEvent(s->computeStream, s->copyDone[k], 0));
    GpuPartView view;
    view.val     = s->d_val[k];
    view.colInd  = s->d_colInd[k];
    view.lens    = (s->h_lens != NULL) ? s->d_lens[k] : NULL;
    view.ptr     = s->d_ptr[k];
    view.elemBase = p->elemStart;
    view.rowBase  = p->start * s->rowsPerUnit;
    view.count    = p->count;
    launch(&view, s->computeStream, args);
    GPU_SAFE_CALL(gpuEventRecord(s->computeDone[k], s->computeStream));

    s->bytesCopied += p->payloadBytes;
  }

  GPU_SAFE_CALL(gpuEventRecord(s->copyStop, s->copyStream));
  /* Host-timer invariant: the sweep looks synchronous to its caller (the
   * per-step timers in chebFDSolver.c rely on every step draining the
   * device). copyStream must be drained too: EventElapsedTime below (and
   * any later legacy-stream op) may only run once copyStart/copyStop have
   * actually completed, and the event records are processed asynchronously
   * even though the last compute already waited on the last copy. */
  GPU_SAFE_CALL(gpuStreamSynchronize(s->computeStream));
  GPU_SAFE_CALL(gpuStreamSynchronize(s->copyStream));

  float ms = 0.0f;
  GPU_SAFE_CALL(gpuEventElapsedTime(&ms, s->copyStart, s->copyStop));
  s->copyMs += (double)ms;
  s->sweeps++;
}

void gpu_matrix_stream_prefetch(const V_ELE *p, size_t bytes)
{
  if (p == NULL || bytes == 0) {
    return;
  }
  NVTX_RANGE_PUSH_C("gpu.matrixStreamPrefetch", NVTX_C_STREAM);
  /* Prefetch to the current device; the follow-up sync keeps the legacy
   * default stream ordered against it. */
  int dev = 0;
  GPU_SAFE_CALL(gpuGetDevice(&dev));
  GPU_SAFE_CALL(gpuMemPrefetch(p, bytes, dev));
  GPU_SAFE_CALL(gpuDeviceSynchronize());
  NVTX_RANGE_POP();
}

void gpu_matrix_stream_stats(const GpuMatrixStream *s,
    size_t *h2dBytes,
    int *nParts,
    unsigned long long *nSweeps,
    double *copyMs)
{
  if (h2dBytes != NULL)
    *h2dBytes = s->bytesCopied;
  if (nParts != NULL)
    *nParts = s->nParts;
  if (nSweeps != NULL)
    *nSweeps = s->sweeps;
  if (copyMs != NULL)
    *copyMs = s->copyMs;
}
