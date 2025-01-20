/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "util.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>
#endif

#include "allocate.h"
#include "comm.h"

// subroutines local to this module
static int sizeOfRank(int rank, int size, int N)
{
  return N / size + ((N % size > rank) ? 1 : 0);
}

// Ensure that all the neighbors we expect to receive from also send to us
// sendList - All ranks receive values from us
// numSendNeighbors - Number of entries in sendList
// recvList - All ranks that send values from us
// numRecvNeighbors - Number of entries in recvList
// FIXME: What if ranks want to send to us and are not in sendlist?
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
  MPI_Status status;
  for (int i = 0; i < numSendNeighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      printf("MPI_Wait error\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
    sendList[i] = status.MPI_SOURCE;
  }
}

static void buildNeighbors(Comm* c, int* external_processor)
{
  int numNeighbors = c->numNeighbors;
  int numExternal  = c->numExternal;
  int* neighbors   = c->neighbors;
  int lengths[numNeighbors];
  int MPI_MY_TAG = 100;
  MPI_Request request[MAX_NUM_MESSAGES];

  // First post receives
  for (int i = 0; i < numNeighbors; i++) {
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

  for (int i = 0; i < numNeighbors; i++) {
    int count = 0;

    // go through list of external elements until updating
    // processor changes
    while ((j < numExternal) && (external_processor[j] == neighbors[i])) {
      count++;
      j++;
      if (j == numExternal) break;
    }

    recvCount[i] = count;
    MPI_Send(&count, 1, MPI_INT, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  MPI_Status status;
  // Complete the receives of the number of externals
  for (int i = 0; i < numNeighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      printf("MPI_Wait error\n");
      exit(EXIT_FAILURE);
    }
    sendCount[i] = lengths[i];
  }
}

static void buildElementsToSend(
    Comm* c, int startRow, int* external_processor, int* new_external)
{
  int numNeighbors = c->numNeighbors;
  int numExternal  = c->numExternal;
  int* neighbors   = c->neighbors;
  int MPI_MY_TAG   = 100;
  MPI_Request request[MAX_NUM_MESSAGES];
  c->elementsToSend   = (int*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(int));
  int* elementsToSend = c->elementsToSend;

  int j = 0;

  for (int i = 0; i < numNeighbors; i++) {
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

  for (int i = 0; i < numNeighbors; i++) {
    int start = j;

    // Go through list of external elements
    // until updating processor changes.  This is redundant, but
    // saves us from recording this information.
    while ((j < numExternal) && (external_processor[j] == neighbors[i])) {
      j++;
      if (j == numExternal) break;
    }

    MPI_Send(new_external + start,
        j - start,
        MPI_INT,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD);
  }

  MPI_Status status;
  // receive from each neighbor the global index list of external elements
  for (int i = 0; i < numNeighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      printf("MPI_Wait error\n");
      exit(EXIT_FAILURE);
    }
  }

  /// replace global indices by local indices
  for (int i = 0; i < c->totalSendCount; i++) {
    elementsToSend[i] -= startRow;
  }
}

void commPartition(Comm* c, Matrix* A)
{
#ifdef _MPI
  int rank = c->rank;
  int size = c->size;

  // Extract Matrix pieces
  CG_UINT start_row  = A->startRow;
  CG_UINT stop_row   = A->stopRow;
  CG_UINT total_nrow = A->totalNr;
  CG_UINT total_nnz  = A->totalNnz;
  CG_UINT local_nrow = A->nr;
  CG_UINT local_nnz  = A->nnz;
  CG_UINT* row_ptr   = A->rowPtr;
  CG_UINT* col_ind   = A->colInd;

  // We need to convert the index values for the rows on this processor
  // to a local index space. We need to:
  // - Determine if each index reaches to a local value or external value
  // - If local, subtract start_row from index value to get local index
  // - If external, find out if it is already accounted for.
  //   - If so, then do nothing,
  //   - otherwise
  //     - add it to the list of external indices,
  //     - find out which processor owns the value.
  //     - Set up communication for sparse MV operation.

  // FIXME: Use a lookup table with size total number of rows. For lower memory
  // consumption a map would be better choice.
  int* externals   = (int*)allocate(ARRAY_ALIGNMENT, total_nrow * sizeof(int));
  int num_external = 0; // local number of external indices

  // column indices that are not processed yet are marked with -1
  for (int i = 0; i < total_nrow; i++) {
    externals[i] = -1;
  }

  int* external_index = (int*)allocate(ARRAY_ALIGNMENT,
      MAX_EXTERNAL * sizeof(int));

  for (int i = 0; i < local_nrow; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
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
      if (start_row <= cur_ind && cur_ind <= stop_row) {
        col_ind[j] -= start_row;
      } else {
        // find out if we have already set up this point
        if (externals[cur_ind] == -1) {
          externals[cur_ind] = num_external;

          if (num_external <= MAX_EXTERNAL) {
            external_index[num_external] = cur_ind;
            // mark in local column index as external by negating it
            // col_ind[j] = -(col_ind[j] + 1); // FIXME: Offset?
            col_ind[j] = -col_ind[j];
          } else {
            printf("Must increase MAX_EXTERNAL\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
          }
          num_external++;
        } else {
          // Mark index as external by adding 1 and negating it
          // col_ind[j] = -(col_ind[j] + 1); // FIXME: Offset?
          col_ind[j] = -col_ind[j];
        }
      }
    }
  }

  /**************************************************************************
  Go through list of externals to find out which processors must be accessed.
  **************************************************************************/
  int external_processor[num_external];

  {
    int globalIndexOffsets[size];

    MPI_Allgather(&start_row,
        1,
        MPI_INT,
        globalIndexOffsets,
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    // for (int i = 0; i < size; i++) {
    //   printf("Rank %d: i = %d: OFFSET %d\n", rank, i, globalIndexOffsets[i]);
    // }

    // Go through list of externals and find the processor that owns each

    for (int i = 0; i < num_external; i++) {
      int globalIndex = external_index[i];
      for (int j = size - 1; j >= 0; j--) {
        if (globalIndexOffsets[j] <= globalIndex) {
          external_processor[i] = j;
          break;
        }
      }
    }
  }
  /*Go through the external elements. For each newly encountered external
  point assign it the next index in the local sequence. Then look for other
  external elements who are updated by the same rank and assign them the next
  set of index numbers in the local sequence (ie. elements updated by the same
  rank have consecutive indices).*/
  int* external_local_index = (int*)allocate(ARRAY_ALIGNMENT,
      num_external * sizeof(int));

  int count = local_nrow;

  for (int i = 0; i < num_external; i++) {
    external_local_index[i] = -1;
  }

  for (int i = 0; i < num_external; i++) {
    if (external_local_index[i] == -1) {
      external_local_index[i] = count++;

      for (int j = i + 1; j < num_external; j++) {
        if (external_processor[j] == external_processor[i]) {
          external_local_index[j] = count++;
        }
      }
    }
  }

  // map all external ids to the new local index
  CG_UINT* rowPtr = A->rowPtr;

  for (int i = 0; i < local_nrow; i++) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      if (col_ind[j] < 0) {
        // size_t cur_ind = -(col_ind[j] - 1); // FIXME: Offset by 1??
        int cur_ind = -col_ind[j];
        col_ind[j]  = external_local_index[externals[cur_ind]];
      }
    }
  }

  free(externals);
  // int new_external_processor[num_external];
  //
  // for (int i = 0; i < num_external; i++) {
  //   new_external_processor[i] = -1;
  // }
  //
  // // setup map from external id to partition
  // for (int i = 0; i < num_external; i++) {
  //   int id                     = external_local_index[i] - local_nrow;
  //   new_external_processor[id] = external_processor[i];
  //   printf("Rank %d of %d: %d new_external_processor[%d] = %d\n",
  //       rank,
  //       size,
  //       i,
  //       id,
  //       external_processor[i]);
  // }
  // commFinalize(c);
  // exit(EXIT_SUCCESS);
  //
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

  /* Count the number of neighbors from which we receive information to update
   our external elements. Additionally, fill the array tmp_neighbors in the
   following way:
        tmp_neighbors[i] = 0   ==>  No external elements are updated by
                                processor i.
        tmp_neighbors[i] = x   ==>  (x-1)/size elements are updated from
                                processor i.*/

  int tmp_neighbors[size];

  for (int i = 0; i < size; i++) {
    tmp_neighbors[i] = 0;
  }

  int num_recv_neighbors = 0;
  int length             = 1;

  // Encoding both number of ranks that need values from this rank and the total
  // number of values by adding one for any additional rank and adding size for
  // every additional value.
  for (int i = 0; i < num_external; i++) {
    if (tmp_neighbors[external_processor[i]] == 0) {
      num_recv_neighbors++;
      tmp_neighbors[external_processor[i]] = 1;
    }
    tmp_neighbors[external_processor[i]] += size;
  }

  // sum over all processors all the tmp_neighbors arrays
  MPI_Allreduce(MPI_IN_PLACE,
      tmp_neighbors,
      size,
      MPI_INT,
      MPI_SUM,
      MPI_COMM_WORLD);

  /* decode the combined 'tmp_neighbors'  array from all ranks */
  // Number of ranks that receive values from us
  int num_send_neighbors = tmp_neighbors[rank] % size;

  /* decode 'tmp_neighbors[rank] to deduce total number of elements we must send
   */
  c->totalSendCount = (tmp_neighbors[rank] - num_send_neighbors) / size;

  /* Check to see if we have enough memory allocated.  This could be
   dynamically modified, but let's keep it simple for now...*/
  if (num_send_neighbors > MAX_NUM_MESSAGES) {
    printf("Must increase MAX_NUM_MESSAGES. Must be at least %d\n",
        num_send_neighbors);
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
  int* recv_list = allocate(ARRAY_ALIGNMENT, MAX_NUM_MESSAGES * sizeof(int));

  {
    int j          = 0;
    recv_list[j++] = external_processor[0];

    for (int i = 1; i < num_external; i++) {
      if (external_processor[i - 1] != external_processor[i]) {
        recv_list[j++] = external_processor[i];
      }
    }
  }

  // Ensure that all the neighbors we expect to receive from also send to us
  int send_list[num_send_neighbors];

  probeNeighbors(send_list, num_send_neighbors, recv_list, num_recv_neighbors);

  /*  Compare the two lists. In most cases they should be the same.
  //  However, if they are not then add new entries to the recv list
  //  that are in the send list (but not already in the recv list).
  FIXME: WHY!! This ensures that the recv_list is equal to the sendlist
  But why is this required? -> Just One neighbour list??*/
  for (int j = 0; j < num_send_neighbors; j++) {
    int found = 0;
    for (int i = 0; i < num_recv_neighbors; i++) {
      if (recv_list[i] == send_list[j]) found = 1;
    }

    if (found == 0) {
#ifdef VERBOSE
      printf("Process %d of %d: recv_list[%d] = %d\n",
          rank,
          size,
          num_recv_neighbors,
          send_list[j]);
#endif
      recv_list[num_recv_neighbors] = send_list[j];
      num_recv_neighbors++;
    }
  }

  // From here on only have one neighbor list for both send and recv
  c->numNeighbors = num_recv_neighbors;
  for (int i = 0; i < c->numNeighbors; i++) {
    c->neighbors[i] = recv_list[i];
  }

  if (c->numNeighbors > MAX_NUM_MESSAGES) {
    printf("Must increase MAX_EXTERNAL\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }

  // Create 'new_external' which explicitly put the external elements in the
  // order given by 'external_local_index'
  int* new_external = (int*)allocate(ARRAY_ALIGNMENT,
      num_external * sizeof(int));

  for (int i = 0; i < num_external; i++) {
    new_external[external_local_index[i] - local_nrow] = external_index[i];
  }

  free(external_local_index);
  free(external_index);
  c->numExternal = num_external;

  buildNeighbors(c, external_processor);

  // Send each processor the global index list of the external elements in the
  // order that I will want to receive them when updating my external elements
  // Build "elementsToSend" list. These are the x elements the current rank
  // owns that need to be sent to other ranks
  buildElementsToSend(c, A->startRow, external_processor, new_external);

  A->nc          = A->nc + num_external;
  c->send_buffer = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
      c->totalSendCount * sizeof(CG_FLOAT));

  free(recv_list);
  free(new_external);
#endif
}

void commExchange(Comm* c, Matrix* A, CG_FLOAT* x)
{
#ifdef _MPI
  int numNeighbors     = c->numNeighbors;
  int* neighbors       = c->neighbors;
  int* recvCount       = c->recvCount;
  int* sendCount       = c->sendCount;
  CG_FLOAT* sendBuffer = c->send_buffer;
  int* elementsToSend  = c->elementsToSend;

  int MPI_MY_TAG = 99;
  MPI_Request request[numNeighbors];

  // Externals are at end of locals
  CG_FLOAT* externals = x + A->nr;

  // Post receives
  for (int i = 0; i < numNeighbors; i++) {
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
  for (int i = 0; i < c->totalSendCount; i++) {
    sendBuffer[i] = x[elementsToSend[i]];
  }

  // Send to each neighbor
  for (int i = 0; i < numNeighbors; i++) {
    int count = sendCount[i];

    MPI_Send(sendBuffer,
        count,
        MPI_FLOAT_TYPE,
        neighbors[i],
        MPI_MY_TAG,
        MPI_COMM_WORLD);

    sendBuffer += count;
  }

  // Complete the receives issued above
  MPI_Status status;
  for (int i = 0; i < numNeighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      printf("MPI_Wait error\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
  }
#endif
}

void commReduction(double* v, int op)
{
#ifdef _MPI
  if (op == MAX) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  } else if (op == SUM) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
}

void commPrintConfig(Comm* c)
{
#ifdef _MPI
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (commIsMaster(c)) {
    printf("Communication setup:\n");
  }

  for (int i = 0; i < c->size; i++) {
    if (i == c->rank) {
      printf("Rank %d has %d neighbors with %d externals:\n",
          c->rank,
          c->numNeighbors,
          c->numExternal);
      for (int j = 0; j < c->numNeighbors; j++) {
        printf("\t%d: receive %d send %d\n",
            c->neighbors[j],
            c->recvCount[j],
            c->sendCount[j]);
      }
      printf("\tSend %d elements: [", c->totalSendCount);
      for (int j = 0; j < c->totalSendCount; j++) {
        printf("%d, ", c->elementsToSend[j]);
      }
      printf("]\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
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
      printf("Rank %d: %d non zeroes, number of rows %d\n",
          rank,
          m->nnz,
          numRows);

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
