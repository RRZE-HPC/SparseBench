/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "profiler.h"
#include "comm.h"
#include "likwid-marker.h"
#include "util.h"
#include <stddef.h>

typedef struct {
  char *label;
  size_t words;
  size_t flops;
} workType;

double _t[NUMREGIONS];

static workType _regions[NUMREGIONS] = {{"waxpby:  ", 3, 6},
                                        {"spMVM:   ", 0, 2},
                                        {"ddot:    ", 2, 4},
                                        {"comm:    ", 0, 0}};

void profilerInit(size_t *facFlops, size_t *facWords) {
  LIKWID_MARKER_INIT;
  _Pragma("omp parallel") {
    LIKWID_MARKER_REGISTER("WAXPBY");
    LIKWID_MARKER_REGISTER("SPMVM");
    LIKWID_MARKER_REGISTER("DDOT");
    LIKWID_MARKER_REGISTER("COMM");
  }

  for (int i = 0; i < NUMREGIONS; i++) {
    _t[i] = 0.0;
    _regions[i].flops *= facFlops[i];
    _regions[i].words *= facWords[i];
  }

  _regions[SPMVM].words = facWords[SPMVM];
}

void profilerPrint(Comm *c, int iterations) {

  if (c->size > 1) {
#ifdef _MPI
    double tmin[NUMREGIONS];
    double tmax[NUMREGIONS];
    double tavg[NUMREGIONS];

    MPI_Reduce(_t, tmin, NUMREGIONS, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(_t, tmax, NUMREGIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(_t, tavg, NUMREGIONS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int i = 0; i < NUMREGIONS; i++) {
      tavg[i] /= c->size;
    }

    int commWords = 0;
    for (int i = 0; i < c->outdegree; i++) {
      commWords += c->sendCounts[i];
    }
    for (int i = 0; i < c->indegree; i++) {
      commWords += c->recvCounts[i];
    }

    _regions[COMM].words = sizeof(CG_FLOAT) * commWords;
    int commVolume[c->size];
    MPI_Gather(&commWords, 1, MPI_INT, commVolume, 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    double commTime[c->size];
    MPI_Gather(&_t[COMM], 1, MPI_DOUBLE, commTime, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    if (commIsMaster(c)) {
      printf(HLINE);
      printf("Function   avg MB/s  avg MFlop/s  Walltime(s) min, max, avg\n");
      for (int j = 0; j < NUMREGIONS - 1; j++) {
        double bytes = (double)_regions[j].words * iterations;
        double flops = (double)_regions[j].flops * iterations;

        printf("%s%11.2f %11.2f %11.2f %11.2f %11.2f\n", _regions[j].label,
               1.0E-06 * bytes / tavg[j], 1.0E-06 * flops / tavg[j], tmin[j],
               tmax[j], tavg[j]);
      }
      printf(HLINE);
      double totalVolume = 0.0;
      printf("Communication\n");
      printf("rank\tkB\tkB/s\tWalltime(s)\n");
      for (int i = 0; i < c->size; i++) {
        double dataVolume = 1.0E-03 * commVolume[i];
        printf("%d %11.2f %11.2f %11.2e\n", i, dataVolume,
               dataVolume / commTime[i], commTime[i]);
        totalVolume += commVolume[i];
      }

      printf("Total data volume %.2f kB\n", 1.0E-03 * totalVolume);
      printf("Walltime(s): min %.2e s, max %.2e s, avg %.2e s\n", tmin[COMM],
             tmax[COMM], tavg[COMM]);
      printf(HLINE);
    }
#endif
  } else {
    printf(HLINE);
    printf("Function   Rate(MB/s)  Rate(MFlop/s)  Walltime(s)\n");
    for (int j = 0; j < NUMREGIONS - 1; j++) {
      double bytes = (double)_regions[j].words * iterations;
      double flops = (double)_regions[j].flops * iterations;

      printf("%s%11.2f %11.2f %11.2f\n", _regions[j].label,
             1.0E-06 * bytes / _t[j], 1.0E-06 * flops / _t[j], _t[j]);
    }
    printf(HLINE);
  }
}

void profilerFinalize(void) { LIKWID_MARKER_CLOSE; }
