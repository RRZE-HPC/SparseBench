/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "allocate.h"
#include "cli.h"
#include "comm.h"
#include "matrix.h"
#include "matrixBinfile.h"
#include "parameter.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "util.h"

static void initMatrix(CommType *c, Parameter *p, GMatrix *m)
{
  if (strcmp(p->filename, "generate") == 0) {
    matrixGenerate(m, p, c->rank, c->size, false);
  } else if (strcmp(p->filename, "generate7P") == 0) {
    matrixGenerate(m, p, c->rank, c->size, true);
  } else {
    char *dot = strrchr(p->filename, '.');
    if (strcmp(dot, ".mtx") == 0) {
      MMMatrix mm;
      MMMatrix mmLocal;

      if (commIsMaster(c)) {
        printf("Read MTX matrix\n");
        MMMatrixRead(&mm, p->filename);
      }

      commDistributeMatrix(c, &mm, &mmLocal);
      matrixConvertfromMM(&mmLocal, m);
    } else if (strcmp(dot, ".bmx") == 0) {
#ifdef _MPI
      if (commIsMaster(c)) {
        printf("Read BMX matrix\n");
      }
      matrixBinRead(m, c, p->filename);
#else
      printf("Binary matrix files are only supported with MPI!\n");
      exit(EXIT_SUCCESS);
#endif
    } else {
      printf("Unknown matrix file format!\n");
    }
  }
}

int main(int argc, char **argv)
{
  Parameter param;
  CommType comm;

  commInit(&comm, argc, argv);
  initParameter(&param);
  parseArguments(&comm, &param, argc, argv);
  commPrintBanner(&comm);
  // printParameter(&param);

  double ts;
  GMatrix m;
  double timeStart = getTimeStamp();
  initMatrix(&comm, &param, &m);
  commBarrier();
  double timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf("Init matrix took %.2fs\n", timeStop - timeStart);
  }
  timeStart = getTimeStamp();
  commLocalization(&comm, &m);

  Matrix sm;
  convertMatrix(&sm, &m);
  commBarrier();
  timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf(
        "Parallel localization and matrix conversion took %.2fs\n", timeStop - timeStart);
  }

  size_t factorFlops[NUMREGIONS];
  size_t factorWords[NUMREGIONS];

  factorFlops[DDOT]   = m.totalNr;
  factorWords[DDOT]   = 3 * sizeof(CG_FLOAT) * m.totalNr / 2;
  factorFlops[WAXPBY] = m.totalNr;
  factorWords[WAXPBY] = 3 * sizeof(CG_FLOAT) * m.totalNr;
  factorFlops[SPMVM]  = m.totalNnz;
  factorWords[SPMVM]  = (sizeof(CG_FLOAT) * m.totalNnz) + (sizeof(CG_UINT) * m.totalNnz);
  // TODO: EXTEND SPMMV
  factorFlops[SPMMV] = m.totalNnz;
  factorWords[SPMMV] = (sizeof(CG_FLOAT) * m.totalNnz) + (sizeof(CG_UINT) * m.totalNnz);

  profilerInit(factorFlops, factorWords);
  int numSeq = 0;
  int *seq   = NULL;

  int k      = 0;
  switch (BenchType) {
  case CG:
    numSeq       = 3;
    int seqCg[3] = { DDOT, WAXPBY, SPMVM };
    seq          = seqCg;
    if (commIsMaster(&comm)) {
      printf("Test type: CG\n");
    }
    k = solveCG(&comm, &param, &sm);
    break;
  case SPMV:
    numSeq          = 1;
    int secSpmvm[1] = { SPMV };
    seq             = secSpmvm;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    const int itermax = param.itermax;
    CG_FLOAT *x       = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m.nc * sizeof(CG_FLOAT));
    CG_FLOAT *y       = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m.nr * sizeof(CG_FLOAT));

    for (int i = 0; i < m.nr; i++) {
      x[i] = (CG_FLOAT)1.0;
      y[i] = (CG_FLOAT)1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, spMVM(&sm, x, y));
    }
    break;

  case SPMMV: {
    numSeq          = 1;
    int secSpmmv[1] = { SPMMV };
    seq             = secSpmmv;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    int itermax = param.itermax;
    DMatrix x;
    DMatrix y;
    x.entries = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, NUMVEC * m.nc * sizeof(CG_FLOAT));
    y.entries = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, NUMVEC * m.nr * sizeof(CG_FLOAT));

    for (int i = 0; i < NUMVEC * m.nc; i++) {
      x.entries[i] = (CG_FLOAT)1.0;
    }
    for (int i = 0; i < NUMVEC * m.nr; i++) {
      y.entries[i] = (CG_FLOAT)1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMMVM, spMMVM(&sm, &x, &y));
    }
  } break;

  case GMRES:
    if (commIsMaster(&comm)) {
      printf("Test type: GMRES\n");
      printf("GMRES not implemented yet\n");
    }
    commAbort(&comm, "GMRES not implemented yet\n");
    break;

  case CHEBFD:
    if (commIsMaster(&comm)) {
      printf("Test type: CHEBFD\n");
      printf("CHEBFD not implemented yet\n");
    }
    commAbort(&comm, "CHEBFD not implemented yet\n");
    break;
  default:;
  }

  profilerPrint(&comm, seq, numSeq, k);
  profilerFinalize();
  commFinalize(&comm);

  return EXIT_SUCCESS;
}
