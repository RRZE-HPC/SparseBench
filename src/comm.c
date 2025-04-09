/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "matrix.h"
#include "util.h"
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <sys/types.h>
#define gettid() (int)syscall(SYS_gettid)
#endif

#ifdef _OPENMP
#include "affinity.h"
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif

#include "allocate.h"
#include "bstree.h"
#include "comm.h"

#ifdef _MPI
static int sizeOfRank(int rank, int size, int N)
{
  return N / size + ((N % size > rank) ? 1 : 0);
}

static void buildIndexMapping(Comm* c,
    Matrix* A,
    Bstree* externals,
    int* externalIndex,
    int* externalsReordered,
    int* externalRank)
{
  int externalCount = c->externalCount;
  /*Go through the external elements. For each newly encountered external
  assign it the next index in the local sequence. Then look for other
  external elements who are updated by the same rank and assign them the next
  set of index numbers in the local sequence (ie. elements updated by the same
  rank have consecutive indices).*/
  int* externalLocalIndex = (int*)allocate(ARRAY_ALIGNMENT,
      externalCount * sizeof(int));
  int* newExternalRank    = (int*)allocate(ARRAY_ALIGNMENT,
      externalCount * sizeof(int));

  int count = A->nr;

  for (int i = 0; i < externalCount; i++) {
    externalLocalIndex[i] = -1;
  }

  int index = 0;

  for (int i = 0; i < externalCount; i++) {
    if (externalLocalIndex[i] == -1) {
      externalLocalIndex[i]    = count++;
      newExternalRank[index++] = externalRank[i];

      for (int j = i + 1; j < externalCount; j++) {
        if (externalRank[j] == externalRank[i]) {
          externalLocalIndex[j]    = count++;
          newExternalRank[index++] = externalRank[j];
        }
      }
    }
  }

#ifdef VERBOSE
  fprintf(c->logFile, "REORDER last index %d\n", index);
#endif

  // Update externalRank for new ordering
  for (int i = 0; i < externalCount; i++) {
    externalRank[i] = newExternalRank[i];
  }

  // map column ids in the matrix to the new local index
  CG_UINT* rowPtr  = A->rowPtr;
  CG_UINT* colInd  = A->colInd;
  CG_UINT numRows  = A->nr;
  CG_UINT startRow = A->startRow;
  CG_UINT stopRow  = A->stopRow;

  for (int i = 0; i < numRows; i++) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      CG_UINT curIndex = colInd[j];

      if (startRow <= curIndex && curIndex <= stopRow) {
        colInd[j] -= startRow;
      } else {
        colInd[j] = externalLocalIndex[bstFind(externals, curIndex)];
      }
    }
  }

  for (int i = 0; i < externalCount; i++) {
    externalsReordered[externalLocalIndex[i] - numRows] = externalIndex[i];
  }

  free(externalLocalIndex);
  free(newExternalRank);
}

static void buildElementsToSend(
    Comm* c, int startRow, int* externalRank, int* externalReordered)
{
  c->totalSendCount = 0;
  for (int i = 0; i < c->outdegree; i++) {
    c->totalSendCount += c->sendCounts[i];
  }

  c->sendBuffer  = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(CG_FLOAT));
  int MPI_MY_TAG = 100;
  MPI_Request request[c->outdegree];
  c->elementsToSend   = (int*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(int));
  int* elementsToSend = c->elementsToSend;

  int j = 0;

  for (int i = 0; i < c->outdegree; i++) {
    c->sdispls[i] = j;
    MPI_Irecv(elementsToSend + j,
        c->sendCounts[i],
        MPI_INT,
        c->destinations[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD,
        request + i);

    j += c->sendCounts[i];
  }

  j = 0;

  for (int i = 0; i < c->indegree; i++) {
    c->rdispls[i] = j;
    MPI_Send(externalReordered + j,
        c->recvCounts[i],
        MPI_INT,
        c->sources[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD);

    j += c->recvCounts[i];
  }

  MPI_Waitall(c->outdegree, request, MPI_STATUSES_IGNORE);

  // map global indices to local indices
  for (int i = 0; i < c->totalSendCount; i++) {
    elementsToSend[i] -= startRow;
  }
}
#endif

void commPrintBanner(Comm* c)
{
  int rank = c->rank;
  int size = c->size;

  char host[_POSIX_HOST_NAME_MAX];
  pid_t master_pid = getpid();
  gethostname(host, _POSIX_HOST_NAME_MAX);

  if (c->size > 1) {
    if (commIsMaster(c)) {
      printf(BANNER "\n");
      printf("MPI parallel using %d ranks\n", c->size);
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp single
        printf("OpenMP enabled using %d threads\n", omp_get_num_threads());
      }
#endif
    }
    commBarrier();
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        printf("Process with rank %d running on Node %s with pid %d\n",
            rank,
            host,
            master_pid);
      }

#ifdef VERBOSE_AFFINITY
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp critical
        {
          printf("Rank %d Thread %d running on Node %s core %d with pid %d "
                 "and tid "
                 "%d\n",
              rank,
              omp_get_thread_num(),
              host,
              sched_getcpu(),
              master_pid,
              gettid());
          affinity_getmask();
        }
#endif
      }
#endif
    }
    commBarrier();
  } else {
    printf(BANNER "\n");
    printf("Running with only one process!\n");
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
      printf("OpenMP enabled using %d threads\n", omp_get_num_threads());

#ifdef VERBOSE_AFFINITY
#pragma omp critical
      {
        printf("Rank %d Thread %d running on Node %s core %d with pid %d "
               "and tid "
               "%d\n",
            rank,
            omp_get_thread_num(),
            host,
            sched_getcpu(),
            master_pid,
            gettid());
        affinity_getmask();
      }
#endif
    }
#endif
  }
}

static void scanMM(
    MmMatrix* m, int startRow, int stopRow, int* entryCount, int* entryOffset)
{
  Entry* e = m->entries;
  int in   = 0;

  for (size_t i = 0; i < m->count; i++) {
    if (e[i].row == startRow && in == 0) {
      *entryOffset = i;
      in           = 1;
    }
    if (e[i].row == (stopRow + 1)) {
      *entryCount = (i - *entryOffset);
      break;
    }
    if (i == m->count - 1) {
      *entryCount = (i - *entryOffset + 1);
      break;
    }
  }
}

void commDistributeMatrix(Comm* c, MmMatrix* m, MmMatrix* mLocal)
{
#ifdef _MPI
  int rank = c->rank;
  int size = c->size;
  int totalCounts[2];

  if (rank == 0) {
    totalCounts[0] = m->nr;
    totalCounts[1] = m->nnz;
  }

  MPI_Bcast(&totalCounts, 2, MPI_INT, 0, MPI_COMM_WORLD);
  int totalNr = totalCounts[0], totalNnz = totalCounts[1];
  MPI_Aint displ[2];
  {
    Entry* dummy = m->entries;
    MPI_Aint base_address;
    MPI_Get_address(dummy, &base_address);
    MPI_Get_address(&(dummy->row), &displ[0]);
    MPI_Get_address(&(dummy->val), &displ[1]);
    displ[0] = MPI_Aint_diff(displ[0], base_address);
    displ[1] = MPI_Aint_diff(displ[1], base_address);
  }

  MPI_Datatype entryType;
  int blocklengths[2]   = { 2, 1 };
  MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
  MPI_Type_create_struct(2, blocklengths, displ, types, &entryType);
  MPI_Type_commit(&entryType);

  int sendcounts[size];
  int senddispls[size];

  if (commIsMaster(c)) {
    int cursor = 0;
    for (int i = 0; i < size; i++) {
      int numRows  = sizeOfRank(i, size, totalNr);
      int startRow = cursor;
      cursor += numRows;
      int stopRow = cursor - 1;
      scanMM(m, startRow, stopRow, &sendcounts[i], &senddispls[i]);
      // printf("Rank %d count %d displ %d start %d stop %d\n",
      //     i,
      //     sendcounts[i],
      //     senddispls[i],
      //     startRow,
      //     stopRow);
    }
  }

  int count;
  MPI_Scatter(sendcounts, 1, MPI_INT, &count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  mLocal->count    = count;
  mLocal->totalNr  = totalNr;
  mLocal->totalNnz = totalNnz;
  mLocal->entries  = (Entry*)allocate(ARRAY_ALIGNMENT, count * sizeof(Entry));

  MPI_Scatterv(m->entries,
      sendcounts,
      senddispls,
      entryType,
      mLocal->entries,
      count,
      entryType,
      0,
      MPI_COMM_WORLD);

  mLocal->startRow = mLocal->entries[0].row;
  mLocal->stopRow  = mLocal->entries[count - 1].row;
  mLocal->nr       = mLocal->stopRow - mLocal->startRow + 1;
  mLocal->nnz      = count;

  MPI_Type_free(&entryType);
#else
  mLocal->startRow = 0;
  mLocal->stopRow  = m->nr - 1;
  mLocal->count    = m->count;
  mLocal->nr       = m->nr;
  mLocal->nnz      = m->nnz;
  mLocal->entries  = m->entries;
#endif /* ifdef _MPI */
}

void commPartition(Comm* c, Matrix* A)
{
#ifdef _MPI
  int rank = c->rank;
  int size = c->size;

  CG_UINT startRow     = A->startRow;
  CG_UINT stopRow      = A->stopRow;
  CG_UINT numRowsTotal = A->totalNr;
  CG_UINT numRows      = A->nr;
  CG_UINT* rowPtr      = A->rowPtr;
  CG_UINT* colInd      = A->colInd;

#ifdef VERBOSE
  fprintf(c->logFile,
      "Rank %d of %d owns %d rows: %d to %d\n",
      rank,
      size,
      numRows,
      startRow,
      stopRow);
#endif

  /***********************************************************************
   *    Step 1: Identify externals and create lookup maps
   ************************************************************************/

  Bstree* externals;
  externals = bstNew();
  // int* externals = (int*)allocate(ARRAY_ALIGNMENT, numRowsTotal *
  // sizeof(int));
  int externalCount = 0; // local number of external indices

  int* externalIndex = (int*)allocate(ARRAY_ALIGNMENT,
      MAX_EXTERNAL * sizeof(int));

#ifdef VERBOSE
  fprintf(c->logFile, "STEP 1 \n");
#endif
  for (int i = 0; i < numRows; i++) {
    for (CG_UINT j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      CG_UINT curIndex = A->colInd[j];

      // convert local column references to local numbering
      if (curIndex < startRow || curIndex > stopRow) {
        // find out if we have already set up this point
        if (!bstExists(externals, curIndex)) {
          bstInsert(externals, curIndex, externalCount);

          if (externalCount <= MAX_EXTERNAL) {
            externalIndex[externalCount] = curIndex;
          } else {
            printf("Must increase MAX_EXTERNAL\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
          }
          externalCount++;
        }
      }
    }
  }

#ifdef VERBOSE
  printf("Rank %d: %d externals\n", c->rank, externalCount);
#endif

  /***********************************************************************
   *    Step 2:  Build dist Graph topology and init neigbors
   ************************************************************************/
  int* externalRank  = (int*)allocate(ARRAY_ALIGNMENT,
      externalCount * sizeof(int));
  int* recvNeighbors = (int*)allocate(ARRAY_ALIGNMENT, size * sizeof(int));

  c->externalCount = externalCount;

  for (int i = 0; i < size; i++) {
    recvNeighbors[i] = -1;
  }

  {
    int globalIndexOffsets[size];
    int sourceCount = 0;

    MPI_Allgather(&startRow,
        1,
        MPI_INT,
        globalIndexOffsets,
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    // Go through list of externals and find the processor that owns it
    for (int i = 0; i < externalCount; i++) {
      int globalIndex = externalIndex[i];

      for (int j = size - 1; j >= 0; j--) {
        if (globalIndexOffsets[j] <= globalIndex) {
          externalRank[i] = j;
          if (recvNeighbors[j] < 0) {
            recvNeighbors[j] = 1;
            sourceCount++;
          } else {
            recvNeighbors[j]++;
          }
          break;
        }
      }
    }

    int sources[sourceCount];
    int degrees[sourceCount];
    int destinations[sourceCount];
    int weights[sourceCount];
    int cursor = 0;

    for (int i = 0; i < size; i++) {
      if (recvNeighbors[i] > 0) {
        sources[cursor]   = i;
        weights[cursor++] = recvNeighbors[i];
      }
    }

    for (int i = 0; i < sourceCount; i++) {
      degrees[i]      = 1;
      destinations[i] = rank;
    }

    MPI_Dist_graph_create(MPI_COMM_WORLD,
        sourceCount,
        sources,
        degrees,
        destinations,
        weights,
        MPI_INFO_NULL,
        0,
        &c->communicator);
  }

  {
    int weighted;
    MPI_Dist_graph_neighbors_count(c->communicator,
        &c->indegree,
        &c->outdegree,
        &weighted);

#ifdef VERBOSE
    printf("Rank %d: In %d Out %d Weighted %d\n",
        rank,
        c->indegree,
        c->outdegree,
        weighted);
#endif

    c->sources      = (int*)malloc(c->indegree * sizeof(int));
    c->recvCounts   = (int*)malloc(c->indegree * sizeof(int));
    c->rdispls      = (int*)malloc(c->indegree * sizeof(int));
    c->destinations = (int*)malloc(c->outdegree * sizeof(int));
    c->sendCounts   = (int*)malloc(c->outdegree * sizeof(int));
    c->sdispls      = (int*)malloc(c->outdegree * sizeof(int));

    MPI_Dist_graph_neighbors(c->communicator,
        c->indegree,
        c->sources,
        c->recvCounts,
        c->outdegree,
        c->destinations,
        c->sendCounts);
  }

  /***********************************************************************
   *    Step 3:  Build and apply index mapping
   ************************************************************************/
  int* externalsReordered = (int*)allocate(ARRAY_ALIGNMENT,
      externalCount * sizeof(int));

  buildIndexMapping(c,
      A,
      externals,
      externalIndex,
      externalsReordered,
      externalRank);

  free(externalIndex);
  bstFree(externals);

#ifdef VERBOSE
  fprintf(c->logFile, "STEP 3 \n");
  fprintf(c->logFile,
      "Rank %d of %d: %d externals\n",
      rank,
      size,
      externalCount);

  for (int i = 0; i < externalCount; i++) {
    fprintf(c->logFile,
        "Rank %d of %d: external[%d] owned by %d\n",
        rank,
        size,
        i,
        externalRank[i]);
  }
#endif

  A->nc = A->nc + externalCount;

  /***********************************************************************
   *    Step 4:  Build global index list for external communication
   ************************************************************************/
  buildElementsToSend(c, A->startRow, externalRank, externalsReordered);

  free(externalsReordered);
#endif
}

void commExchange(Comm* c, Matrix* A, CG_FLOAT* x)
{
#ifdef _MPI
  CG_FLOAT* sendBuffer = c->sendBuffer;
  CG_FLOAT* externals  = x + A->nr;
  int* elementsToSend  = c->elementsToSend;

// Copy values for all ranks into send buffer
#pragma omp parallel for
  for (int i = 0; i < c->totalSendCount; i++) {
    sendBuffer[i] = x[elementsToSend[i]];
  }

  MPI_Neighbor_alltoallv(sendBuffer,
      c->sendCounts,
      c->sdispls,
      MPI_FLOAT_TYPE,
      externals,
      c->recvCounts,
      c->rdispls,
      MPI_FLOAT_TYPE,
      c->communicator);

#endif
}

void commReduction(CG_FLOAT* v, int op)
{
#ifdef _MPI
  if (op == MAX) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_FLOAT_TYPE, MPI_MAX, MPI_COMM_WORLD);
  } else if (op == SUM) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
}

void commPrintConfig(Comm* c, int nr, int startRow, int stopRow)
{
#ifdef _MPI
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (commIsMaster(c)) {
    printf("Communication setup:\n");
  }

  for (int i = 0; i < c->size; i++) {
    if (i == c->rank) {
      printf("Rank %d has %d rows (%d to %d) with %d externals\n",
          c->rank,
          nr,
          startRow,
          stopRow,
          c->externalCount);

      for (int k = 0; k < c->size; k++) {
        if (k == c->rank) {
          for (int i = 0; i < c->indegree; i++) {
            printf("Rank %d: Source[%d] %d Recv count %d\n",
                c->rank,
                i,
                c->sources[i],
                c->recvCounts[i]);
          }
          for (int i = 0; i < c->outdegree; i++) {
            printf("Rank %d: Dest[%d] %d Send count %d\n",
                c->rank,
                i,
                c->destinations[i],
                c->sendCounts[i]);
          }
          fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      /*printf("\tSend %d elements: [", c->totalSendCount);*/
      /*for (int j = 0; j < c->totalSendCount; j++) {*/
      /*  printf("%d ", c->elementsToSend[j]);*/
      /*}*/
      /*printf("]\n");*/
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

void commMMMatrixDump(Comm* c, MmMatrix* m)
{
  int rank        = c->rank;
  int size        = c->size;
  CG_UINT numRows = m->nr;

  for (int i = 0; i < size; i++) {
    if (i == rank) {
      printf("RANK %d with %lu entries %d nonzeros and %d rows\n",
          rank,
          m->count,
          m->nnz,
          numRows);
      dumpMMMatrix(m);
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

void commMatrixDump(Comm* c, Matrix* m)
{
  int rank        = c->rank;
  int size        = c->size;
  CG_UINT numRows = m->nr;
  CG_UINT* rowPtr = m->rowPtr;
  CG_UINT* colInd = m->colInd;
  CG_FLOAT* val   = m->val;

  if (commIsMaster(c)) {
    printf("Matrix: %d total non zeroes, total number of rows %d\n",
        m->totalNnz,
        m->totalNr);
  }

  for (int i = 0; i < size; i++) {
    if (i == rank) {
      printf("Rank %d: number of rows %d\n", rank, numRows);

      for (int rowID = 0; rowID < numRows; rowID++) {
        printf("Row [%d]: ", rowID);

        for (int rowEntry = rowPtr[rowID]; rowEntry < rowPtr[rowID + 1];
            rowEntry++) {
          printf("[%d]:%.2f ", colInd[rowEntry], val[rowEntry]);
        }

        printf("\n");
      }
      fflush(stdout);
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

void commInit(Comm* c, int argc, char** argv)
{
#ifdef _MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &(c->rank));
  MPI_Comm_size(MPI_COMM_WORLD, &(c->size));
#else
  c->rank = 0;
  c->size = 1;
#endif
#ifdef VERBOSE
  char filename[30];
  sprintf(filename, "out-%d.txt", c->rank);
  c->logFile = fopen(filename, "w");
#endif
}

void commFinalize(Comm* c)
{
#ifdef _MPI
  free(c->sources);
  free(c->recvCounts);
  free(c->rdispls);
  free(c->destinations);
  free(c->sendCounts);
  free(c->sdispls);
  free(c->elementsToSend);
  free(c->sendBuffer);
  MPI_Finalize();
#endif

#ifdef VERBOSE
  fclose(c->logFile);
#endif
}
