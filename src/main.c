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
#include "comm.h"
#include "matrix.h"
#include "matrixBinfile.h"
#include "parameter.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "util.h"

#define NUMVEC 10 // TODO:move this some where better

typedef enum { CG = 0, SPMV, SPMMV, GMRES, CHEBFD, NUMTYPES } types;

#define HELPTEXT                                                               \
  "Usage: sparseBench [options]\n\n"                                           \
  "Options:\n"                                                                 \
  "  -h         Show this help text\n"                                         \
  "  -c <file name>   Convert MM matrix to binary matrix file.\n"              \
  "  -f <parameter file>   Load options from a parameter file\n"               \
  "  -m <MM matrix>   Load a matrix market file\n"                             \
  "  -t <bench type>   Benchmark type, can be cg, spmv, or gmres. Default "    \
  "cg.\n"                                                                      \
  "  -x <int>   Size in x for generated matrix, ignored if MM file is "        \
  "loaded. Default 100.\n"                                                     \
  "  -y <int>   Size in y for generated matrix, ignored if MM file is "        \
  "loaded. Default 100.\n"                                                     \
  "  -z <int>   Size in z for generated matrix, ignored if MM file is "        \
  "loaded. Default 100.\n"                                                     \
  "  -i <int>   Number of solver iterations. Default 150.\n"                   \
  "  -e <float>  Convergence criteria epsilon. Default 0.0.\n"

#ifdef _MPI
static void writeBinMatrix(Comm* c, char* filename)
{
  MMMatrix mm, mmLocal;
  GMatrix m;
  if (commIsMaster(c)) {
    MMMatrixRead(&mm, filename);
  }
  commDistributeMatrix(c, &mm, &mmLocal);
  matrixConvertfromMM(&mmLocal, &m);
  matrixBinWrite(&m, c, changeFileEnding(filename, ".bmx"));
}
#endif

static void initMatrix(Comm* c, Parameter* p, GMatrix* m)
{
  if (strcmp(p->filename, "generate") == 0) {
    matrixGenerate(m, p, c->rank, c->size, false);
  } else if (strcmp(p->filename, "generate7P") == 0) {
    matrixGenerate(m, p, c->rank, c->size, true);
  } else {
    char* dot = strrchr(p->filename, '.');
    if (strcmp(dot, ".mtx") == 0) {
      MMMatrix mm, mmLocal;

      if (commIsMaster(c)) {
        printf("Read MTX matrix\n");
        MMMatrixRead(&mm, p->filename);
        printf("DEBUG: Rank 0 after MMMatrixRead - totalNr=%d totalNnz=%d nr=%d nnz=%d count=%zu\n", 
               mm.totalNr, mm.totalNnz, mm.nr, mm.nnz, mm.count);
      }

      commDistributeMatrix(c, &mm, &mmLocal);
      printf("DEBUG: Rank %d after commDistributeMatrix - totalNr=%d totalNnz=%d nr=%d nnz=%d count=%zu\n", 
             c->rank, mmLocal.totalNr, mmLocal.totalNnz, mmLocal.nr, mmLocal.nnz, mmLocal.count);
      matrixConvertfromMM(&mmLocal, m);
      printf("DEBUG: Rank %d after matrixConvertfromMM - totalNr=%u totalNnz=%u nr=%u nnz=%u\n", 
             c->rank, m->totalNr, m->totalNnz, m->nr, m->nnz);
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

int main(int argc, char** argv)
{
  Parameter param;
  Comm comm;

  commInit(&comm, argc, argv);
  initParameter(&param);

  char* cvalue = NULL;
  int index;
  int type  = CG;
  bool stop = false;
  int c;

  opterr = 0;

  while ((c = getopt(argc, argv, "hc:t:f:m:x:y:z:i:e:")) != -1)
    switch (c) {
    case 'h':
      if (commIsMaster(&comm)) {
        printf(HELPTEXT);
      }
      commAbort(&comm, "Finish write matrix");
      break;
    case 'c':
#ifdef _MPI
      writeBinMatrix(&comm, optarg);
      commAbort(&comm, "Finish write matrix");
#else
      printf("Binary matrix files are only supported with MPI!\n");
      exit(EXIT_SUCCESS);
#endif
    case 'f':
      readParameter(&param, optarg);
      break;
    case 'm':
      param.filename = optarg;
      break;
    case 't':
      if (strcmp(optarg, "cg") == 0) type = CG;
      else if (strcmp(optarg, "spmv") == 0)
        type = SPMV;
      else if (strcmp(optarg, "spmmv") == 0)
        type = SPMMV;
      else if (strcmp(optarg, "gmres") == 0)
        type = GMRES;
      else if (strcmp(optarg, "cheb") == 0)
        type = CHEBFD;
      else {
        printf("Unknown solver type %s\n", optarg);
        return 1;
      }
      break;
    case 'x':
      param.nx = atoi(optarg);
      break;
    case 'y':
      param.ny = atoi(optarg);
      break;
    case 'z':
      param.nz = atoi(optarg);
      break;
    case 'i':
      param.itermax = atoi(optarg);
      break;
    case 'e':
      param.eps = atof(optarg);
      break;
    case '?':
      if (optopt == 'c')
        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
      else if (isprint(optopt))
        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
      else
        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
      return 1;
    default:
      abort();
    }

  for (index = optind; index < argc; index++) {
    printf("Non-option argument %s\n", argv[index]);
  }

  if (stop) {
    commAbort(&comm, "Wrong command line arguments");
  }

  commPrintBanner(&comm);

  double timeStart, timeStop, ts;
  GMatrix m;
  timeStart = getTimeStamp();
  initMatrix(&comm, &param, &m);
  commBarrier();
  timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf("Init matrix took %.2fs\n", timeStop - timeStart);
  }
  timeStart = getTimeStamp();
  commPartition(&comm, &m);
  // commPrintConfig(&comm, m.nr, m.nnz, m.startRow, m.stopRow);

  printf("DEBUG: Before convertMatrix - GMatrix: totalNr=%u totalNnz=%u nr=%u nnz=%u\n", 
         m.totalNr, m.totalNnz, m.nr, m.nnz);
  Matrix sm;
  convertMatrix(&sm, &m);
  printf("DEBUG: After convertMatrix - Matrix: totalNr=%u totalNnz=%u nr=%u nnz=%u\n", 
         sm.totalNr, sm.totalNnz, sm.nr, sm.nnz);
  commBarrier();
  timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf("Parallel localization and matrix conversion took %.2fs\n",
        timeStop - timeStart);
  }

  size_t factorFlops[NUMREGIONS];
  size_t factorWords[NUMREGIONS];

  printf("DEBUG: Matrix values - totalNr=%u totalNnz=%u nc=%u nr=%u\n", 
         m.totalNr, m.totalNnz, m.nc, m.nr);

  factorFlops[DDOT]   = m.totalNr;
  factorWords[DDOT]   = 3 * sizeof(CG_FLOAT) * m.totalNr / 2;
  factorFlops[WAXPBY] = m.totalNr;
  factorWords[WAXPBY] = 3 * sizeof(CG_FLOAT) * m.totalNr;
  factorFlops[SPMVM]  = m.totalNnz;
  factorWords[SPMVM]  = sizeof(CG_FLOAT) * (m.nc + m.nr + m.totalNnz) +
                       sizeof(CG_UINT) * m.totalNnz;
  factorFlops[SPMMVM]  = NUMVEC * m.totalNnz;
  factorWords[SPMMVM]  = sizeof(CG_FLOAT) * (NUMVEC * m.nc + NUMVEC * m.nr + m.totalNnz) +
                       sizeof(CG_UINT) * m.totalNnz;

  printf("DEBUG: Calculated factors:\n");
  printf("  DDOT: flops=%zu words=%zu\n", factorFlops[DDOT], factorWords[DDOT]);
  printf("  WAXPBY: flops=%zu words=%zu\n", factorFlops[WAXPBY], factorWords[WAXPBY]);
  printf("  SPMVM: flops=%zu words=%zu\n", factorFlops[SPMVM], factorWords[SPMVM]);
  printf("  SPMMVM: flops=%zu words=%zu\n", factorFlops[SPMMVM], factorWords[SPMMVM]);

  profilerInit(factorFlops, factorWords);

  int k = 0;
  switch (type) {
  case CG:
    if (commIsMaster(&comm)) {
      printf("Test type: CG\n");
    }
    k = solveCG(&comm, &param, &sm);
    break;
  case SPMV:
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    int itermax = param.itermax;
    CG_FLOAT* x = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m.nc * sizeof(CG_FLOAT));
    CG_FLOAT* y = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m.nr * sizeof(CG_FLOAT));

    for (int i = 0; i < m.nr; i++) {
      x[i] = (CG_FLOAT)1.0;
      y[i] = (CG_FLOAT)1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, spMVM(&sm, x, y));
    }
    break;

  case SPMMV: {
    if (commIsMaster(&comm)) {
      printf("Test type: SPMVM\n");
    }
    int itermax = param.itermax;
    DMatrix x;
    DMatrix y;
    x.entries = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,NUMVEC * m.nc * sizeof(CG_FLOAT));
    y.entries = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,NUMVEC * m.nr * sizeof(CG_FLOAT));

    for (int i = 0; i < NUMVEC * m.nc; i++) {
      x.entries[i] = (CG_FLOAT)1.0;
    }
    for (int i = 0; i < NUMVEC * m.nr; i++) {
      y.entries[i] = (CG_FLOAT)1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMMVM, spMMVM(&sm, &x, &y));
    }
  } 
  break;

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
  }

  profilerPrint(&comm, k);
  profilerFinalize();
  commFinalize(&comm);

  return EXIT_SUCCESS;
}



/*
NOTE DEBUG:
*/