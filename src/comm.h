/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __COMM_H_
#define __COMM_H_
#include "util.h"
#if defined(_MPI)
#include <mpi.h>
#endif

#include "matrix.h"

#define MAX_EXTERNAL       100000
#define MAX_NUM_MESSAGES   500
#define MAX_NUM_NEIGHBOURS MAX_NUM_MESSAGES

enum op { MAX = 0, SUM };

typedef struct {
  int rank;
  int size;
#if defined(_MPI)
  int numNeighbors;
  int numExternal;
  int totalSendCount;
  int* elementsToSend;
  int neighbors[MAX_NUM_NEIGHBOURS];
  int recvCount[MAX_NUM_NEIGHBOURS];
  int sendCount[MAX_NUM_NEIGHBOURS];
  CG_FLOAT* send_buffer;
#endif
} Comm;

extern void commInit(Comm* c, int argc, char** argv);
extern void commFinalize(Comm* c);
extern void commPartition(Comm* c, Matrix* m);
extern void commPrintConfig(Comm* c);
extern void commMatrixDump(Comm* c, Matrix* m);
extern void commExchange(Comm* c, Matrix* A, double* x);
extern void commReduction(double* v, int op);

static inline int commIsMaster(Comm* c) { return c->rank == 0; }
#endif // __COMM_H_
