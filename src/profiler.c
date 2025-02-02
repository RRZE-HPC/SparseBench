/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "profiler.h"
#include "comm.h"
#include "likwid-marker.h"
#include "solver.h"
#include "util.h"

typedef struct {
  char* label;
  size_t words;
  size_t flops;
} workType;

double _t[NUMREGIONS];

static workType _regions[NUMREGIONS] = { { "waxpby:  ", 3, 6 },
  { "spMVM:   ", 0, 2 },
  { "ddot:    ", 2, 4 },
  { "comm:    ", 0, 0 } };

void profilerInit(Solver* s)
{
  LIKWID_MARKER_INIT;
  _Pragma("omp parallel")
  {
    LIKWID_MARKER_REGISTER("WAXPBY");
    LIKWID_MARKER_REGISTER("SPMVM");
    LIKWID_MARKER_REGISTER("DDOT");
    LIKWID_MARKER_REGISTER("COMM");
  }

  for (int i = 0; i < NUMREGIONS; i++) {
    _t[i] = 0.0;
  }

  _regions[DDOT].flops *= s->A.nr;
  _regions[DDOT].words *= s->A.nr;
  _regions[WAXPBY].flops *= s->A.nr;
  _regions[WAXPBY].words *= s->A.nr;
  _regions[SPMVM].flops *= s->A.nnz;
  _regions[SPMVM].words = sizeof(CG_FLOAT) * s->A.nnz +
                          sizeof(CG_UINT) * s->A.nnz;
}

void profilerPrint(Comm* c, Solver* s, int iterations)
{
  if (c->size > 1) {

  } else {
    printf(HLINE);
    printf("Function   Rate(MB/s)  Rate(MFlop/s)  Walltime(s)\n");
    for (int j = 0; j < NUMREGIONS - 1; j++) {
      double bytes = (double)_regions[j].words * iterations;
      double flops = (double)_regions[j].flops * iterations;

      if (flops > 0) {
        printf("%s%11.2f %11.2f %11.2f\n",
            _regions[j].label,
            1.0E-06 * bytes / _t[j],
            1.0E-06 * flops / _t[j],
            _t[j]);
      } else {
        printf("%s%11.2f - %11.2f\n",
            _regions[j].label,
            1.0E-06 * bytes / _t[j],
            _t[j]);
      }
    }
    printf(HLINE);
  }
}

void profilerFinalize(void) { LIKWID_MARKER_CLOSE; }
