/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "comm.h"
#include "matrix.h"
#include "parameter.h"
#include "profiler.h"
#include "solver.h"

static void initMatrix(Comm *c, Parameter *p, GMatrix *m) {
  if (strcmp(p->filename, "generate") == 0) {
    matrixGenerate(m, p, c->rank, c->size, false);
  } else if (strcmp(p->filename, "generate7P") == 0) {
    matrixGenerate(m, p, c->rank, c->size, true);
  } else {
    MMMatrix mm, mmLocal;
    if (commIsMaster(c)) {
      MMMatrixRead(&mm, p->filename);
    }

    commDistributeMatrix(c, &mm, &mmLocal);
    matrixConvertfromMM(&mm, m);
  }
}

int main(int argc, char **argv) {
  Parameter param;
  Comm comm;

  commInit(&comm, argc, argv);
  initParameter(&param);

  char *cvalue = NULL;
  int index;
  int c;

  opterr = 0;

  while ((c = getopt(argc, argv, "f:x:y:z:i:e:")) != -1)
    switch (c) {
    case 'f':
      readParameter(&param, optarg);
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

  commPrintBanner(&comm);

  GMatrix m;
  initMatrix(&comm, &param, &m);
  size_t factorFlops[NUMREGIONS];
  size_t factorWords[NUMREGIONS];
  factorFlops[DDOT] = m.nr;
  factorWords[DDOT] = sizeof(CG_FLOAT) * m.nr;
  factorFlops[WAXPBY] = m.nr;
  factorWords[WAXPBY] = sizeof(CG_FLOAT) * m.nr;
  factorFlops[SPMVM] = m.nnz;
  factorWords[SPMVM] = sizeof(CG_FLOAT) * m.nnz + sizeof(CG_UINT) * m.nnz;
  profilerInit(factorFlops, factorWords);
  commPartition(&comm, &m);
#ifdef VERBOSE
  commPrintConfig(&comm, s.A.nr, s.A.startRow, s.A.stopRow);
#endif

  Matrix sm;
  convertMatrix(&sm, &m);

  int k = solveCG(&comm, &param, &sm);
  profilerPrint(&comm, k);
  // solverCheckResidual(&s, &comm);
  profilerFinalize();
  commFinalize(&comm);

  return EXIT_SUCCESS;
}
