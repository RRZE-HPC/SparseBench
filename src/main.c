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
#include "chebFDSolver.h"
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

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "cuda_kernels.h"
#endif

void omp_init(V_ELE *data_ptr, size_t elem_count, V_ELE value)
{
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < elem_count; i++) {
    data_ptr[i] = 1.0;
  }
}

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
      // In the 1-rank build mmLocal.entries aliases mm.entries freeing local is enough
      freeMMMatrix(&mmLocal);
#ifdef _MPI
      if (commIsMaster(c)) {
        freeMMMatrix(&mm);
      }
#endif
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
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_init(param.device);
#endif
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

  size_t factorFlops[NUMREGIONS];
  size_t factorWords[NUMREGIONS];

  // TODO : update the flops based on V_ELE type
  factorFlops[DDOT]   = m.totalNr;
  factorWords[DDOT]   = 3 * sizeof(CG_FLOAT) * m.totalNr / 2;
  factorFlops[WAXPBY] = m.totalNr;
  factorWords[WAXPBY] = 3 * sizeof(CG_FLOAT) * m.totalNr;
  factorFlops[SPMVM]  = m.totalNnz;
  factorWords[SPMVM]  = (sizeof(CG_FLOAT) * m.totalNnz) + (sizeof(CG_UINT) * m.totalNnz);
  factorFlops[SPMMVM] = factorFlops[SPMVM] * param.blockwidth;
  factorWords[SPMMVM] = factorWords[SPMVM] * param.blockwidth;

  profilerInit(factorFlops, factorWords);

  // previously using stack local stored data resulting in undefine behaviour
  int seqCg[3]     = { DDOT, WAXPBY, SPMVM };
  int seqSpmv[1]   = { SPMVM };
  int seqSpmmv[1]  = { SPMMVM };
  int seqChebfd[1] = { SPMVM };

  int numSeq       = 0;
  int *seq         = NULL;

  // SCS spMVM/spMMVM has padded rows so update accordingly
#ifdef SCS
  CG_UINT vecSize = sm.nrPadded;
#else
  CG_UINT vecSize = sm.nr;
#endif

  int k = 0;
  switch (BenchType) {
  case CG:
    numSeq = 3;
    seq    = seqCg;
    if (commIsMaster(&comm)) {
      printf("Test type: CG\n");
    }
    k = solveCG(&comm, &param, &sm);
    break;

  case SPMV: {
    numSeq = 1;
    seq    = seqSpmv;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    const int itermax = param.itermax;
    V_ELE *x          = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecSize * sizeof(V_ELE));
    V_ELE *y          = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecSize * sizeof(V_ELE));

    // Parallel init for NUMA first-touch — must match spMVM's schedule.
    omp_init(x, vecSize, 1.0);
    omp_init(y, vecSize, 0.0);

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, spMVM(&sm, x, y));
    }
    deallocate(x);
    deallocate(y);
  } break;

  case SPMMV: {
    numSeq = 1;
    seq    = seqSpmmv;
    if (commIsMaster(&comm)) {
      printf("Test type: SPMMVM\n");
    }
    int itermax = param.itermax;
    DMatrix x   = { .nr = vecSize, .nc = param.blockwidth, .entries = NULL };
    DMatrix y   = { .nr = vecSize, .nc = param.blockwidth, .entries = NULL };
    x.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, x.nr * x.nc * sizeof(V_ELE));
    y.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, y.nr * y.nc * sizeof(V_ELE));

    // Parallel init for NUMA first-touch — must match spMMVM's schedule.
#pragma omp parallel for schedule(OMP_SCHEDULE)
    for (int i = 0; i < x.nr * x.nc; i++) {
      x.entries[i] = 1.0;
    }
#pragma omp parallel for schedule(OMP_SCHEDULE)
    for (int i = 0; i < y.nr * y.nc; i++) {
      y.entries[i] = 0.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMMVM, spMMVM(&sm, &x, &y));
    }
    deallocate(x.entries);
    deallocate(y.entries);
  } break;

  case GMRES:
    if (commIsMaster(&comm)) {
      printf("Test type: GMRES\n");
      printf("GMRES not implemented yet\n");
    }
    commAbort(&comm, "GMRES not implemented yet\n");
    break;

  case CHEBFD: {
    if (commIsMaster(&comm)) {
      printf("Test type: CHEBFD\n");
    }
    numSeq    = 1;
    seq       = seqChebfd;
    int found = solveChebFD(&comm, &param, &sm);
    k         = found > 0 ? found : 0;
    break;
  }

  default:;
  }

  profilerPrint(&comm, seq, numSeq, k);
  profilerFinalize();
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_finalize();
#endif
  commFinalize(&comm);

  freeMatrix(&sm);
  freeGMatrix(&m);

  return EXIT_SUCCESS;
}
