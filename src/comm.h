/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __COMM_H_
#define __COMM_H_
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#if defined(_MPI)
#include <mpi.h>
#endif

#include "matrix.h"

#define MAX_EXTERNAL       100000
#define MAX_NUM_MESSAGES   500
#define MAX_NUM_NEIGHBOURS MAX_NUM_MESSAGES

#define BANNER                                                                 \
  "                                                                          " \
  "        \n"                                                                 \
  "  _|_|_|    _|_|_|              _|_|_|                                  "   \
  "_|        \n"                                                               \
  "_|        _|                    _|    _|    _|_|    _|_|_|      _|_|_|  "   \
  "_|_|_|    \n"                                                               \
  "_|        _|  _|_|  _|_|_|_|_|  _|_|_|    _|_|_|_|  _|    _|  _|        "   \
  "_|    _|  \n"                                                               \
  "_|        _|    _|              _|    _|  _|        _|    _|  _|        "   \
  "_|    _|  \n"                                                               \
  "  _|_|_|    _|_|_|              _|_|_|      _|_|_|  _|    _|    _|_|_|  "   \
  "_|    _|  \n"

enum op { MAX = 0, SUM };

typedef struct {
  int rank;
  int size;
  FILE* logFile;
#if defined(_MPI)
  int externalCount;
  int totalSendCount;
  int* elementsToSend;
  int indegree;
  int outdegree;
  int* sources;
  int* recvCounts;
  int* rdispls;
  int* destinations;
  int* sendCounts;
  int* sdispls;
  CG_FLOAT* sendBuffer;
  MPI_Comm communicator;
#endif
} Comm;

extern void commInit(Comm* c, int argc, char** argv);
extern void commFinalize(Comm* c);
extern void commDistributeMatrix(Comm* c, MmMatrix* m, MmMatrix* mLocal);
extern void commPartition(Comm* c, Matrix* m);
extern void commPrintConfig(Comm* c, int nr, int startRow, int stopRow);
extern void commMMMatrixDump(Comm* c, MmMatrix* m);
extern void commMatrixDump(Comm* c, Matrix* m);
extern void commExchange(Comm* c, Matrix* A, double* x);
extern void commReduction(double* v, int op);
extern void commPrintBanner(Comm* c);

static inline int commIsMaster(Comm* c) { return c->rank == 0; }
static inline void commAbort(char* msg)
{
  printf("ERROR: %s\n", msg);
#if defined(_MPI)
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  exit(EXIT_FAILURE);
}
static inline void commBarrier(void)
{
#if defined(_MPI)
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}
#endif // __COMM_H_
