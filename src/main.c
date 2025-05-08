/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
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

  if (argc != 2) {
    if (commIsMaster(&comm)) {
      printf("Usage: %s <configFile>\n", argv[0]);
    }
    commFinalize(&comm);
    exit(EXIT_SUCCESS);
  }

  readParameter(&param, argv[1]);
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
