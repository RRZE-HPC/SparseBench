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
#include <omp.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif

#include "affinity.h"
#include "allocate.h"
#include "comm.h"

#ifdef _MPI
static int sizeOfRank(int rank, int size, int N)
{
  return N / size + ((N % size > rank) ? 1 : 0);
}

static void probeNeighbors(
    int* sendList, int numSendNeighbors, int* recvList, int numRecvNeighbors)
{

  int val;

  for (int i = 0; i < numSendNeighbors; i++) {
    sendList[i] = -1;
  }

  int MPI_MY_TAG = 99;
  MPI_Request request[MAX_NUM_MESSAGES];

  for (int i = 0; i < numSendNeighbors; i++) {
    MPI_Irecv(&val,
        1,
        MPI_INT,
        MPI_ANY_SOURCE,
        MPI_MY_TAG,
        MPI_COMM_WORLD,
        request + i);
  }

  for (int i = 0; i < numRecvNeighbors; i++) {
    MPI_Send(&val, 1, MPI_INT, recvList[i], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  // Receive message from each send neighbor to construct 'sendList'.
  for (int i = 0; i < numSendNeighbors; i++) {
    MPI_Status status;
    MPI_Wait(request + i, &status);
    sendList[i] = status.MPI_SOURCE;
  }
}

static void buildIndexMapping(Comm* c,
    Matrix* A,
    int* externals,
    int* externalIndex,
    int* externalsReordered,
    int* externalRank)
{
  int externalCount = c->externalCount;
  /*Go through the external elements. For each newly encountered external
  point assign it the next index in the local sequence. Then look for other
  external elements who are updated by the same rank and assign them the next
  set of index numbers in the local sequence (ie. elements updated by the same
  rank have consecutive indices).*/
  int* externalLocalIndex = (int*)allocate(ARRAY_ALIGNMENT,
      externalCount * sizeof(int));

  int count = A->nr;

  for (int i = 0; i < externalCount; i++) {
    externalLocalIndex[i] = -1;
  }

  for (int i = 0; i < externalCount; i++) {
    if (externalLocalIndex[i] == -1) {
      externalLocalIndex[i] = count++;

      for (int j = i + 1; j < externalCount; j++) {
        if (externalRank[j] == externalRank[i]) {
          externalLocalIndex[j] = count++;
        }
      }
    }
  }

  // map all external column ids in the matrix to the new local index
  CG_UINT* rowPtr = A->rowPtr;
  CG_UINT* colInd = A->colInd;
  CG_UINT numRows = A->nr;

  for (int i = 0; i < numRows; i++) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      if (colInd[j] < 0) {
        int cur_ind = -colInd[j];
        colInd[j]   = externalLocalIndex[externals[cur_ind]];
      }
    }
  }

  for (int i = 0; i < externalCount; i++) {
    externalsReordered[externalLocalIndex[i] - numRows] = externalIndex[i];
  }

  free(externalLocalIndex);
}

static void buildNeighborlist(Comm* c, int* externalRank, int externalCount)
{
  int rank = c->rank;
  int size = c->size;

  /* Count the number of neighbors from which we receive information to update
   our external elements. Additionally, fill the array sendNeighborEncoding in
   the following way: sendNeighborEncoding[i] = 0   ==>  No external elements
   are updated by processor i. sendNeighborEncoding[i] = x   ==>  (x-1)/size
   elements are updated from processor i.*/

  int sendNeighborEncoding[size];

  for (int i = 0; i < size; i++) {
    sendNeighborEncoding[i] = 0;
  }

  int recvNeighborCount = 0;
  int length            = 1;

  // Encoding both number of ranks that need values from this rank and the
  // total number of values by adding one for any additional rank and adding
  // size for every additional value.
  for (int i = 0; i < externalCount; i++) {
    if (sendNeighborEncoding[externalRank[i]] == 0) {
      recvNeighborCount++;
      sendNeighborEncoding[externalRank[i]] = 1;
    }
    sendNeighborEncoding[externalRank[i]] += size;
  }

  // sum over all processors all the sendNeighborEncoding arrays
  MPI_Allreduce(MPI_IN_PLACE,
      sendNeighborEncoding,
      size,
      MPI_INT,
      MPI_SUM,
      MPI_COMM_WORLD);

  /* decode the combined 'sendNeighborEncoding'  array from all ranks */
  // Number of ranks that receive values from us
  int sendNeighborCount = sendNeighborEncoding[rank] % size;

  /* decode 'sendNeighborEncoding[rank] to deduce total number of elements we
   * must send */
  c->totalSendCount = (sendNeighborEncoding[rank] - sendNeighborCount) / size;

  /* Check to see if we have enough memory allocated.  This could be
   dynamically modified, but let's keep it simple for now...*/
  if (sendNeighborCount > MAX_NUM_MESSAGES) {
    printf("Must increase MAX_NUM_MESSAGES. Must be at least %d\n",
        sendNeighborCount);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }

  if (c->totalSendCount > MAX_EXTERNAL) {
    printf("Must increase MAX_EXTERNAL. Must be at least %d\n",
        c->totalSendCount);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }

#ifdef VERBOSE
  printf("Rank %d of %d: tmp_neighbors = %d\n",
      rank,
      size,
      tmp_neighbors[rank]);
  printf("Rank %d of %d: Number of send neighbors = %d\n",
      rank,
      size,
      num_send_neighbors);
  printf("Rank %d of %d: Number of receive neighbors = %d\n",
      rank,
      size,
      num_recv_neighbors);
  printf("Rank %d of %d: Total number of elements to send = %d\n",
      rank,
      size,
      c->totalSendCount);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  /* Make a list of the neighbors that will send information to update our
   external elements (in the order that we will receive this information).*/
  int* recvNeighborList = allocate(ARRAY_ALIGNMENT,
      MAX_NUM_MESSAGES * sizeof(int));

  {
    int j                 = 0;
    recvNeighborList[j++] = externalRank[0];

    for (int i = 1; i < externalCount; i++) {
      if (externalRank[i - 1] != externalRank[i]) {
        recvNeighborList[j++] = externalRank[i];
      }
    }
  }

  // Ensure that all the neighbors we expect to receive from also send to us
  int sendNeighborList[sendNeighborCount];

  probeNeighbors(sendNeighborList,
      sendNeighborCount,
      recvNeighborList,
      recvNeighborCount);

  //  Compare the two lists. In most cases they should be the same.
  //  However, if they are not then add new entries to the recv list
  //  that are in the send list (but not already in the recv list).
  for (int j = 0; j < sendNeighborCount; j++) {
    int found = 0;
    for (int i = 0; i < recvNeighborCount; i++) {
      if (recvNeighborList[i] == sendNeighborList[j]) found = 1;
    }

    if (found == 0) {
#ifdef VERBOSE
      printf("Process %d of %d: recv_list[%d] = %d\n",
          rank,
          size,
          num_recv_neighbors,
          send_list[j]);
#endif
      recvNeighborList[recvNeighborCount] = sendNeighborList[j];
      recvNeighborCount++;
    }
  }

  // From here on there is only one neighbor list for both send and recv
  c->neighborCount = recvNeighborCount;

  if (c->neighborCount > MAX_NUM_MESSAGES) {
    printf("Must increase MAX_EXTERNAL\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < c->neighborCount; i++) {
    c->neighbors[i] = recvNeighborList[i];
  }

  free(recvNeighborList);
}

static void buildMessageCounts(Comm* c, int* externalRank)
{
  int neighborCount = c->neighborCount;
  int externalCount = c->externalCount;
  int* neighbors    = c->neighbors;
  int lengths[neighborCount];
  int MPI_MY_TAG = 100;
  MPI_Request request[MAX_NUM_MESSAGES];

  // First post receives
  for (int i = 0; i < neighborCount; i++) {
    MPI_Irecv(lengths + i,
        1,
        MPI_INT,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD,
        request + i);
  }

  int* recvCount = c->recvCount;
  int* sendCount = c->sendCount;

  int j = 0;

  for (int i = 0; i < neighborCount; i++) {
    int count = 0;

    // go through list of external elements until updating rank changes
    while ((j < externalCount) && (externalRank[j] == neighbors[i])) {
      count++;
      j++;
      if (j == externalCount) break;
    }

    recvCount[i] = count;
    MPI_Send(&count, 1, MPI_INT, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  MPI_Waitall(neighborCount, request, MPI_STATUSES_IGNORE);

  // Complete the receives of the number of externals
  for (int i = 0; i < neighborCount; i++) {
    sendCount[i] = lengths[i];
  }
}

static void buildElementsToSend(
    Comm* c, int startRow, int* externalRank, int* externalReordered)
{
  int neighborCount = c->neighborCount;
  int externalCount = c->externalCount;
  int* neighbors    = c->neighbors;
  int MPI_MY_TAG    = 100;
  MPI_Request request[MAX_NUM_MESSAGES];
  c->elementsToSend   = (int*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(int));
  int* elementsToSend = c->elementsToSend;

  int j = 0;

  for (int i = 0; i < neighborCount; i++) {
    MPI_Irecv(elementsToSend + j,
        c->sendCount[i],
        MPI_INT,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD,
        request + i);

    j += c->sendCount[i];
  }

  j = 0;

  for (int i = 0; i < neighborCount; i++) {
    int start = j;

    // Go through list of external elements
    // until updating processor changes.  This is redundant, but
    // saves us from recording this information.
    while ((j < externalCount) && (externalRank[j] == neighbors[i])) {
      j++;
      if (j == externalCount) break;
    }

    MPI_Send(externalReordered + start,
        j - start,
        MPI_INT,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD);
  }

  MPI_Waitall(neighborCount, request, MPI_STATUSES_IGNORE);

  /// replace global indices by local indices
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
    }
    commBarrier();
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        printf("Process with rank %d running on Node %s with pid %d\n",
            rank,
            host,
            master_pid);
      }
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp single
        printf("OpenMP enabled using %d threads\n", omp_get_num_threads());
#pragma omp barrier

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
  // printf("Rank %d count: %d start %d stop %d\n",
  //     rank,
  //     count,
  //     mLocal->startRow,
  //     mLocal->stopRow);

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

  /***********************************************************************
   *    Step 1: Identify externals and create lookup maps
   ************************************************************************/
  // FIXME: Use a lookup table with size total number of rows. For lower
  // memory consumption a hashmap would be a better choice.
  int* externals = (int*)allocate(ARRAY_ALIGNMENT, numRowsTotal * sizeof(int));
  int externalCount = 0; // local number of external indices

  // column indices that are not processed yet are marked with -1
  for (int i = 0; i < numRowsTotal; i++) {
    externals[i] = -1;
  }

  int* externalIndex = (int*)allocate(ARRAY_ALIGNMENT,
      MAX_EXTERNAL * sizeof(int));

  for (int i = 0; i < numRows; i++) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      int cur_ind = A->colInd[j];

#ifdef VERBOSE
      printf("Rank %d of %d getting entry %d:index %d in local row %d\n",
          rank,
          size,
          j,
          cur_ind,
          i);
#endif

      // convert local column references to local numbering
      if (startRow <= cur_ind && cur_ind <= stopRow) {
        colInd[j] -= startRow;
      } else {
        // find out if we have already set up this point
        if (externals[cur_ind] == -1) {
          externals[cur_ind] = externalCount;

          if (externalCount <= MAX_EXTERNAL) {
            externalIndex[externalCount] = cur_ind;
            // mark in local column index as external by negating it
            colInd[j] = -colInd[j];
          } else {
            printf("Must increase MAX_EXTERNAL\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
          }
          externalCount++;
        } else {
          // Mark index as external by adding 1 and negating it
          colInd[j] = -colInd[j];
        }
      }
    }
  }

  /***********************************************************************
   *    Step 2:  Identify owning rank for externals
   ************************************************************************/
  int externalRank[externalCount];
  c->externalCount = externalCount;

  {
    int globalIndexOffsets[size];

    MPI_Allgather(&startRow,
        1,
        MPI_INT,
        globalIndexOffsets,
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    // Go through list of externals and find the processor that owns each
    for (int i = 0; i < externalCount; i++) {
      int globalIndex = externalIndex[i];

      for (int j = size - 1; j >= 0; j--) {
        if (globalIndexOffsets[j] <= globalIndex) {
          externalRank[i] = j;
          break;
        }
      }
    }
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
  free(externals);

#ifdef VERBOSE
  printf("Rank %d of %d: %d externals\n", rank, size, num_external);

  for (int i = 0; i < num_external; i++) {
    printf("Rank %d of %d: external[%d] owned by %d\n",
        rank,
        size,
        i,
        external_processor[i]);
  }
#endif

  /***********************************************************************
   *    Step 4:  Build list of communication neighbor ranks
   ************************************************************************/
  buildNeighborlist(c, externalRank, externalCount);
  A->nc         = A->nc + externalCount;
  c->sendBuffer = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(CG_FLOAT));

  /***********************************************************************
   *    Step 5:  Build message counts for all communication partners
   ************************************************************************/
  buildMessageCounts(c, externalRank);

  /***********************************************************************
   *    Step 6:  Build global index list for external communication
   ************************************************************************/
  buildElementsToSend(c, A->startRow, externalRank, externalsReordered);

  free(externalsReordered);
#endif
}

void commExchange(Comm* c, Matrix* A, CG_FLOAT* x)
{
#ifdef _MPI
  int neighborCount    = c->neighborCount;
  int* neighbors       = c->neighbors;
  int* recvCount       = c->recvCount;
  int* sendCount       = c->sendCount;
  CG_FLOAT* sendBuffer = c->sendBuffer;
  int* elementsToSend  = c->elementsToSend;

  int MPI_MY_TAG = 99;
  MPI_Request request[neighborCount];

  // Externals are at end of locals
  CG_FLOAT* externals = x + A->nr;

  // Post receives
  for (int i = 0; i < neighborCount; i++) {
    int count = recvCount[i];

    MPI_Irecv(externals,
        count,
        MPI_FLOAT_TYPE,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD,
        request + i);

    externals += count;
  }

  // Copy values for all ranks into send buffer
  // FIXME: Add openmp parallel for
  for (int i = 0; i < c->totalSendCount; i++) {
    sendBuffer[i] = x[elementsToSend[i]];
  }

  // Send to each neighbor
  for (int i = 0; i < neighborCount; i++) {
    int count = sendCount[i];

    MPI_Send(sendBuffer,
        count,
        MPI_FLOAT_TYPE,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD);

    sendBuffer += count;
  }

  MPI_Waitall(neighborCount, request, MPI_STATUSES_IGNORE);
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
      printf("Rank %d has %d rows (%d to %d) and %d neighbors with %d "
             "externals:\n",
          c->rank,
          nr,
          startRow,
          stopRow,
          c->neighborCount,
          c->externalCount);
      for (int j = 0; j < c->neighborCount; j++) {
        printf("\t%d: receive %d send %d\n",
            c->neighbors[j],
            c->recvCount[j],
            c->sendCount[j]);
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
}

void commFinalize(Comm* c)
{
#ifdef _MPI
  MPI_Finalize();
#endif
}
