/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli.h"
#include "matrixBinfile.h"
#include "parameter.h"

int BenchType = CG;

#ifdef _MPI
static void writeBinMatrix(CommType *c, char *filename)
{
  MMMatrix mm;
  MMMatrix mmLocal;
  GMatrix m;
  if (commIsMaster(c)) {
    MMMatrixRead(&mm, filename);
  }
  commDistributeMatrix(c, &mm, &mmLocal);
  matrixConvertfromMM(&mmLocal, &m);
  matrixBinWrite(&m, c, changeFileEnding(filename, ".bmx"));
}
#endif

void parseArguments(CommType *comm, Parameter *param, int argc, char **argv)
{
  char *cvalue = NULL;
  int index;
  bool stop = false;
  int c;

  opterr = 0;

  while ((c = getopt(argc, argv, "hc:t:f:m:x:y:z:i:e:")) != -1) {
    switch (c) {
    case 'h':
      if (commIsMaster(comm)) {
        printf(HELPTEXT);
      }
      commAbort(comm, "Finish write matrix");
      break;
    case 'c':
#ifdef _MPI
      writeBinMatrix(comm, optarg);
      commAbort(comm, "Finish write matrix");
#else
      printf("Binary matrix files are only supported with MPI!\n");
      exit(EXIT_SUCCESS);
#endif
    case 'f':
      readParameter(param, optarg);
      break;
    case 'm':
      param->filename = optarg;
      break;
    case 't':
      if (strcmp(optarg, "cg") == 0) {
        BenchType = CG;
      } else if (strcmp(optarg, "spmv") == 0) {
        BenchType = SPMV;
      } else if (strcmp(optarg, "gmres") == 0) {
        BenchType = GMRES;
      } else if (strcmp(optarg, "cheb") == 0) {
        BenchType = CHEBFD;
      } else {
        printf("Unknown solver type %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;
    case 'x':
      param->nx = (int)strtol(optarg, NULL, INT_BASE);
      break;
    case 'y':
      param->ny = (int)strtol(optarg, NULL, INT_BASE);
      break;
    case 'z':
      param->nz = (int)strtol(optarg, NULL, INT_BASE);
      break;
    case 'i':
      param->itermax = (int)strtol(optarg, NULL, INT_BASE);
      break;
    case 'e':
      param->eps = strtod(optarg, NULL);
      break;
    case '?':
      if (optopt == 'c') {
        FPRINTF(stderr, "Option -%c requires an argument.\n", optopt);
      } else if (isprint(optopt)) {
        FPRINTF(stderr, "Unknown option `-%c'.\n", optopt);
      } else {
        FPRINTF(stderr, "Unknown option character `\\x%x'.\n", optopt);
      }
      exit(EXIT_FAILURE);
    default:
      abort();
    }
  }

  for (index = optind; index < argc; index++) {
    printf("Non-option argument %s\n", argv[index]);
  }

  if (stop) {
    commAbort(comm, "Wrong command line arguments");
  }
}
