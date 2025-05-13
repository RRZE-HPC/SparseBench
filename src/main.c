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

typedef enum { CG = 0, SPMV, GMRES, CHEBFD, NUMTYPES } types;

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

static void writeBinMatrix(Comm *c, char *filename) {
  MMMatrix mm, mmLocal;
  GMatrix m;
  if (commIsMaster(c)) {
    MMMatrixRead(&mm, filename);
  }
  commDistributeMatrix(c, &mm, &mmLocal);
  matrixConvertfromMM(&mmLocal, &m);
  matrixBinWrite(&m, c, changeFileEnding(filename, ".bmx"));
}

static void initMatrix(Comm *c, Parameter *p, GMatrix *m) {
  if (strcmp(p->filename, "generate") == 0) {
    matrixGenerate(m, p, c->rank, c->size, false);
  } else if (strcmp(p->filename, "generate7P") == 0) {
    matrixGenerate(m, p, c->rank, c->size, true);
  } else {
    char *dot = strrchr(p->filename, '.');
    if (strcmp(dot, ".mtx") == 0) {
      MMMatrix mm, mmLocal;

      if (commIsMaster(c)) {
        printf("Read MTX matrix\n");
        MMMatrixRead(&mm, p->filename);
      }

      commDistributeMatrix(c, &mm, &mmLocal);
      matrixConvertfromMM(&mmLocal, m);
    } else if (strcmp(dot, ".bmx") == 0) {
      if (commIsMaster(c)) {
        printf("Read BMX matrix\n");
      }
      matrixBinRead(m, c, p->filename);
    } else {
      printf("Unknown matrix file format!\n");
    }
  }
}

int main(int argc, char **argv) {
  Parameter param;
  Comm comm;

  commInit(&comm, argc, argv);
  initParameter(&param);

  char *cvalue = NULL;
  int index;
  int type = CG;
  bool stop = false;
  int c;

  opterr = 0;

  while ((c = getopt(argc, argv, "hc:t:f:m:x:y:z:i:e:")) != -1)
    switch (c) {
    case 'h':
      if (commIsMaster(&comm)) {
        printf(HELPTEXT);
      }
      stop = true;
      break;
    case 'c':
      writeBinMatrix(&comm, optarg);
      commFinalize(&comm);
      return EXIT_SUCCESS;
    case 'f':
      readParameter(&param, optarg);
      break;
    case 'm':
      param.filename = optarg;
      break;
    case 't':
      if (strcmp(optarg, "cg") == 0)
        type = CG;
      else if (strcmp(optarg, "spmv") == 0)
        type = SPMV;
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
    commAbort("Wrong command line arguments");
  }

  commPrintBanner(&comm);

  double timeStart, timeStop, ts;
  GMatrix m;
  timeStart = getTimeStamp();
  initMatrix(&comm, &param, &m);
  // commGMatrixDump(&comm, &m);
  commAbort("DEBUG");

  size_t factorFlops[NUMREGIONS];
  size_t factorWords[NUMREGIONS];
  factorFlops[DDOT] = m.totalNr;
  factorWords[DDOT] = sizeof(CG_FLOAT) * m.totalNr;
  factorFlops[WAXPBY] = m.totalNr;
  factorWords[WAXPBY] = sizeof(CG_FLOAT) * m.totalNr;
  factorFlops[SPMVM] = m.totalNnz;
  factorWords[SPMVM] =
      sizeof(CG_FLOAT) * m.totalNnz + sizeof(CG_UINT) * m.totalNnz;
  profilerInit(factorFlops, factorWords);
  commPartition(&comm, &m);
#ifdef VERBOSE
  commPrintConfig(&comm, m.nr, m.startRow, m.stopRow);
#endif

  Matrix sm;
  printf("BEFORE\n");
  convertMatrix(&sm, &m);
  printf("AFTER\n");
  timeStop = getTimeStamp();
  if (commIsMaster(&comm)) {
    printf("Setup took %.2fs\n", timeStop - timeStart);
  }

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
    CG_FLOAT *x =
        (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m.nc * sizeof(CG_FLOAT));
    CG_FLOAT *y =
        (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m.nr * sizeof(CG_FLOAT));

    for (int i = 0; i < m.nr; i++) {
      x[i] = (CG_FLOAT)1.0;
      y[i] = (CG_FLOAT)1.0;
    }

    for (k = 1; k < itermax; k++) {
      PROFILE(SPMVM, spMVM(&sm, x, y));
    }
    break;
  case GMRES:
    if (commIsMaster(&comm)) {
      printf("Test type: GMRES\n");
    }

    break;
  }

  profilerPrint(&comm, k);
  profilerFinalize();
  commFinalize(&comm);

  return EXIT_SUCCESS;
}
