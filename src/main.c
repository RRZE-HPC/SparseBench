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
#include "kernel_dispatch.h"
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

/* NUMA first-touch fill. Not named omp_*: that prefix is reserved by the OpenMP
 * specification for the runtime API, and this has external linkage. */
static void firstTouchFill(V_ELE *data_ptr, size_t elem_count, V_ELE value)
{
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (size_t i = 0; i < elem_count; i++) {
    data_ptr[i] = value;
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
    if (dot == NULL) {
      commAbort(c, "Unknown matrix file format (filename has no extension)!\n");
    } else if (strcmp(dot, ".mtx") == 0) {
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
      // Like the sibling arms: an input this build cannot read is a failure, so
      // it must not exit 0 and let a driver record the run as successful.
      commAbort(c, "Binary matrix files are only supported with MPI!\n");
#endif
    } else {
      commAbort(c, "Unknown matrix file format!\n");
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
  /* Multi-rank GPU runs are not supported yet and must not be silently wrong:
   * gpu_ddot has no counterpart to the commReductionV that solver.c's ddot
   * does, so every rank would converge on its own rank-local dot products.
   * Lifting this needs a device-pointer halo exchange in comm.c plus the
   * allreduce in gpu_ddot — see GPU-Port-Plan.md. */
  if (comm.size > 1) {
    commAbort(&comm,
        "GPU builds are single-rank only (gpu_ddot performs no MPI reduction); "
        "run with one rank.\n");
  }
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

  // previously using stack local stored data resulting in undefined behaviour
  int seqCg[3]    = { DDOT, WAXPBY, SPMVM };
  int seqSpmv[1]  = { SPMVM };
  int seqSpmmv[1] = { SPMMVM };

  int numSeq      = 0;
  int *seq        = NULL;
  int rc          = EXIT_SUCCESS;

  // input vectors must span nc.
  // Output vectors must span nr padded
#ifdef SCS
  CG_UINT inSize  = MAX(sm.nc, sm.nrPadded);
  CG_UINT outSize = sm.nrPadded;
#else
  CG_UINT inSize  = sm.nc;
  CG_UINT outSize = sm.nr;
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
    V_ELE *x = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)inSize * sizeof(V_ELE));
    V_ELE *y = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)outSize * sizeof(V_ELE));

    // Parallel init for NUMA first-touch — must match spMVM's schedule.
    firstTouchFill(x, inSize, 1.0);
    firstTouchFill(y, outSize, 0.0);

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, SPMVMFUNC(&sm, x, y));
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
#ifdef SCS
    /* spMMVM stacks a per-thread V_ELE tmp[C * blockwidth] VLA; reject a width
     * that would overflow the worker stack (or a non-positive one, which is a
     * zero-length VLA / a huge unsigned nc) instead of crashing in the kernel.
     * Same limit ChebFD applies to cheb_NS. */
    if (!spMMVMBlockWidthOk(sm.C, param.blockwidth)) {
      if (commIsMaster(&comm)) {
        printf("SPMMV: block width %d is invalid for the SCS spMMVM stack "
               "scratch (C=%llu, limit ~%u bytes/thread); reduce -w or raise "
               "OMP_STACKSIZE.\n",
            param.blockwidth,
            (unsigned long long)sm.C,
            (unsigned)SCS_MAX_SPMMVM_VLA_BYTES);
      }
      rc     = EXIT_FAILURE;
      numSeq = 0;
      break;
    }
#endif
    DMatrix x = { .nr = inSize, .nc = param.blockwidth, .entries = NULL };
    DMatrix y = { .nr = outSize, .nc = param.blockwidth, .entries = NULL };
    x.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)x.nr * x.nc * sizeof(V_ELE));
    y.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)y.nr * y.nc * sizeof(V_ELE));

    // Parallel init for NUMA first-touch — must match spMMVM's schedule.
    firstTouchFill(x.entries, (size_t)x.nr * x.nc, 1.0);
    firstTouchFill(y.entries, (size_t)y.nr * y.nc, 0.0);

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMMVM, SPMMVMFUNC(&sm, &x, &y));
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
#if defined(_MPI)
    if (commIsMaster(&comm)) {
      printf("CURRENTLY CHEB FD doesn't support MPI\n");
    }
    // Fall through to the shared cleanup tail (free sm/m, finalize
    // GPU/LIKWID/comm); skip solveChebFD and the profiler report.
    rc = EXIT_FAILURE;
    break;
#endif
    // ChebFD does its own timing/reporting, so it is left out of the profiler sequence.
    // A negative return means a configuration/validation failure -> propagate a non-zero exit.
    int found = solveChebFD(&comm, &param, &sm);
    if (found < 0) {
      rc = EXIT_FAILURE;
    } else {
      k = found;
    }
    break;
  }

  default:;
  }

  if (rc == EXIT_SUCCESS && numSeq > 0) {
    profilerPrint(&comm, seq, numSeq, k);
  }
  profilerFinalize();
  freeMatrix(&sm);
  freeGMatrix(&m);

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
  gpu_finalize();
#endif
  commFinalize(&comm);

  return rc;
}
