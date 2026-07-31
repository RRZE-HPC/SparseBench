/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __PROFILER_H_
#define __PROFILER_H_
#include "comm.h"
#include <stddef.h>

#ifdef LIKWID_PERFMON
#define PROFILE(tag, call)                                                               \
  _Pragma("omp parallel")                                                                \
  {                                                                                      \
    LIKWID_MARKER_START(#tag);                                                           \
  }                                                                                      \
  ts = getTimeStamp();                                                                   \
  call;                                                                                  \
  T[tag] += (getTimeStamp() - ts);                                                       \
  _Pragma("omp parallel")                                                                \
  {                                                                                      \
    LIKWID_MARKER_STOP(#tag);                                                            \
  }
#else /* LIKWID_PERFMON */
#define PROFILE(tag, call)                                                               \
  ts = getTimeStamp();                                                                   \
  call;                                                                                  \
  T[tag] += (getTimeStamp() - ts);
#endif /* LIKWID_PERFMON */

typedef enum {
  WAXPBY = 0,
  SPMVM,
  SPMMVM,
  SPMVM_LOCAL,
  SPMVM_EXT,
  DDOT,
  COMM, // packing and posting the halo exchange
  COMM_WAIT, // exposed (non-overlapped) halo exchange time
  NUMREGIONS
} RegionsType;

extern double T[NUMREGIONS];
extern void profilerInit(size_t *facFlops, size_t *facWords);
extern void profilerPrint(CommType *c, int *seq, int numSeq, int iterations);
extern void profilerFinalize(void);
#endif // __PROFILER_H
