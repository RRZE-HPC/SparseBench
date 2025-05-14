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

#define MAX_EXTERNAL 6000000

#define BANNER                                                                 \
  "/ _\\_ __   __ _ _ __ ___  ___  / __\\ ___ _ __   ___| |__  \n"             \
  "\\ \\| '_ \\ / _` | '__/ __|/ _ \\/__\\/// _ \\ '_ \\ / __| '_ \\ \n"       \
  "_\\ \\ |_) | (_| | |  \\__ \\  __/ \\/  \\  __/ | | | (__| | | |\n"         \
  "\\__/ .__/ \\__,_|_|  |___/\\___\\_____/\\___|_| |_|\\___|_| |_|\n"         \
  "   |_|                                                    \n"

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
extern void commDistributeMatrix(Comm* c, MMMatrix* m, MMMatrix* mLocal);
extern void commPartition(Comm* c, GMatrix* m);
extern void commPrintConfig(
    Comm* c, CG_UINT nr, CG_UINT nnz, CG_UINT startRow, CG_UINT stopRow);
extern void commGMatrixDump(Comm* c, GMatrix* m);
extern void commMatrixDump(Comm* c, Matrix* m);
extern void commVectorDump(Comm* c, CG_FLOAT* v, CG_UINT size, char* name);
extern void commExchange(Comm* c, CG_UINT numRows, CG_FLOAT* x);
extern void commReduction(CG_FLOAT* v, int op);
extern void commPrintBanner(Comm* c);
extern void commAbort(Comm* c, char* msg);

static inline int commIsMaster(Comm* c) { return c->rank == 0; }
static inline void commBarrier(void)
{
#if defined(_MPI)
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}
#endif // __COMM_H_
