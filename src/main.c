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
#include "vtype.h"

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
  if (param.verbose > 0 && commIsMaster(&comm)) {
    printParameter(&param);
  }

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
#if SCS
  sm.C     = param.C;
  sm.sigma = param.Sigma;
#endif
  convertMatrix(&sm, &m);
  commBarrier();
  timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf(
        "Parallel localization and matrix conversion took %.2fs\n", timeStop - timeStart);
  }

  size_t factorFlops[NUMREGIONS] = { 0 };
  size_t factorWords[NUMREGIONS] = { 0 };

  // TODO : update the flops based on V_ELE type
  factorFlops[DDOT]   = m.totalNr;
  factorWords[DDOT]   = 3 * sizeof(CG_FLOAT) * m.totalNr / 2;
  factorFlops[WAXPBY] = m.totalNr;
  factorWords[WAXPBY] = 3 * sizeof(CG_FLOAT) * m.totalNr;
  /* m.nnz / m.totalNnz are allocation upper bounds for generated matrices
   * (27 entries per row regardless of stencil and of boundary truncation), so
   * they would over-report every SpMV rate. The real local count is rowPtr[nr];
   * sum it up to get the global one. */
  CG_FLOAT nnzSum = (CG_FLOAT)m.rowPtr[m.nr];
  commReduction(&nnzSum, SUM);
  size_t globalNnz    = (size_t)nnzSum;

  factorFlops[SPMVM]  = globalNnz;
  factorWords[SPMVM]  = (sizeof(CG_FLOAT) * globalNnz) + (sizeof(CG_UINT) * globalNnz);
  factorFlops[SPMMVM] = factorFlops[SPMVM] * param.blockwidth;
  factorWords[SPMMVM] = factorWords[SPMVM] * param.blockwidth;

#ifdef CRS
  /* The split kernels each only touch part of the matrix, so they need their
   * own nnz counts - charging both the full nnz would report twice the work
   * that is actually done. By construction local + external == globalNnz. */
  CG_UINT localNnzLocal = 0;
  for (CG_UINT i = 0; i < sm.nr; i++) {
    localNnzLocal += sm.rowLocalEnd[i] - sm.rowPtr[i];
  }

  CG_FLOAT localNnzSum = (CG_FLOAT)localNnzLocal;
  CG_FLOAT extNnzSum   = (CG_FLOAT)(sm.rowPtr[sm.nr] - localNnzLocal);
  commReduction(&localNnzSum, SUM);
  commReduction(&extNnzSum, SUM);

  size_t localNnz          = (size_t)localNnzSum;
  size_t extNnz            = (size_t)extNnzSum;

  factorFlops[SPMVM_LOCAL] = localNnz;
  factorWords[SPMVM_LOCAL] = (sizeof(CG_FLOAT) * localNnz) + (sizeof(CG_UINT) * localNnz);
  factorFlops[SPMVM_EXT]   = extNnz;
  factorWords[SPMVM_EXT]   = (sizeof(CG_FLOAT) * extNnz) + (sizeof(CG_UINT) * extNnz);
#endif

  profilerInit(factorFlops, factorWords);
  int numSeq = 0;
  int *seq   = NULL;

  int k      = 0;
  switch (BenchType) {
  case CG:
#ifdef USE_OVERLAP_SPMVM
    numSeq       = 5;
    int seqCg[5] = { DDOT, WAXPBY, SPMVM_LOCAL, SPMVM_EXT, COMM_WAIT };
#else
    numSeq       = 3;
    int seqCg[3] = { DDOT, WAXPBY, SPMVM };
#endif
    seq = seqCg;
    if (commIsMaster(&comm)) {
      printf("Test type: CG\n");
    }
    k = solveCG(&comm, &param, &sm);
    break;
  case SPMV:
    numSeq          = 1;
    int secSpmvm[1] = { SPMVM };
    seq             = secSpmvm;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    const int itermax = param.itermax;
    V_ELE *x          = (V_ELE *)allocate(ARRAY_ALIGNMENT, m.nc * sizeof(V_ELE));
    V_ELE *y          = (V_ELE *)allocate(ARRAY_ALIGNMENT, m.nr * sizeof(V_ELE));

    for (int i = 0; i < m.nr; i++) {
      x[i] = 1.0;
      y[i] = 1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, spMVM(&sm, x, y));
    }
    break;

  case SPMMV: {
    numSeq          = 1;
    int secSpmmv[1] = { SPMMVM };
    seq             = secSpmmv;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMMVM\n");
    }
    int itermax = param.itermax;
    DMatrix x   = { .nr = sm.nc, .nc = param.blockwidth, .entries = NULL };
    DMatrix y   = { .nr = sm.nr, .nc = param.blockwidth, .entries = NULL };
    x.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, x.nr * x.nc * sizeof(V_ELE));
    y.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, y.nr * y.nc * sizeof(V_ELE));

    for (int i = 0; i < x.nr * x.nc; i++) {
      x.entries[i] = 1.0;
    }
    for (int i = 0; i < y.nr * y.nc; i++) {
      y.entries[i] = 0.0;
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
