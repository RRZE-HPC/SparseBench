/* Copyright (C) NHR@FAU, Universit Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
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

#include "allocate.h"
#include "bstree.h"
#include "comm.h"

#define MPI_TAG_EXCHANGE 100

#ifdef _MPI
#include <mpi.h>

static int sizeOfRank(int rank, int size, int N)
{
  return (N / size) + ((N % size > rank) ? 1 : 0);
}

/**
 * @brief Reorder external elements to group those from the same owning rank consecutively.
 *
 * This function reorganizes external elements so that all externals belonging to the same
 * source rank are assigned consecutive indices in the local extended RHS vector. This
 * ordering enables efficient MPI communication by ensuring that each rank's data forms
 * a contiguous block in memory, which aligns with MPI_Neighbor_alltoallv requirements.
 *
 * @param numRows Number of local rows owned by this rank
 * @param extCount Total number of external elements
 * @param[out] extLocalIndex Maps from original external index to new local RHS index
 *                           (values will be in range [numRows, numRows+extCount-1])
 * @param[in,out] extOwningRank On input: owning rank for each external (original order)
 *                              On output: owning rank for each external (reordered)
 *
 * Algorithm:
 * 1. For each unique owning rank encountered in extOwningRank (in order):
 *    a. Assign the next consecutive local index (starting from numRows)
 *    b. Search remaining externals for same owner and assign consecutive indices
 * 2. Update extOwningRank array to reflect the new ordering
 *
 */
static void reorderExternals(
    const int numRows, const int extCount, int *extLocalIndex, int *extOwningRank)
{
  int *newExtOwningRank = (int *)allocate(ARRAY_ALIGNMENT, extCount * sizeof(int));

  for (int i = 0; i < extCount; i++) {
    extLocalIndex[i] = -1;
  }

  int count = numRows;
  int index = 0;

  for (int i = 0; i < extCount; i++) {
    if (extLocalIndex[i] == -1) {
      extLocalIndex[i]          = count++;
      newExtOwningRank[index++] = extOwningRank[i];

      for (int j = i + 1; j < extCount; j++) {
        if (extOwningRank[j] == extOwningRank[i]) {
          extLocalIndex[j]          = count++;
          newExtOwningRank[index++] = extOwningRank[j];
        }
      }
    }
  }

  for (int i = 0; i < extCount; i++) {
    extOwningRank[i] = newExtOwningRank[i];
  }

  free(newExtOwningRank);
}

/**
 * @brief Remap all matrix column indices to local indexing (0-based local indices).
 *
 * This function transforms the global column indices in the matrix to local indices
 * that reference the extended local RHS vector (local elements + externals). After
 * this transformation, all SpMV operations can use purely local indexing.
 *
 * @param[in,out] A The distributed matrix to localize
 * @param extLookup Binary search tree mapping global column index → external array index
 * @param extLocalIndex Maps external array index → local RHS vector index
 *
 * Algorithm:
 * For each matrix entry:
 * - If column index is in local range [startRow, stopRow]: convert to 0-based (col - startRow)
 * - If column index is external: lookup in extLookup to get external index, then use
 *   extLocalIndex to get the local RHS index (in range [numRows, numRows+extCount-1])
 */
static void localizeMatrix(GMatrix *A, Bstree *extLookup, const int *extLocalIndex)
{
  CG_UINT *rowPtr  = A->rowPtr;
  Entry *entries   = A->entries;
  CG_UINT numRows  = A->nr;
  CG_UINT startRow = A->startRow;
  CG_UINT stopRow  = A->stopRow;

  for (int i = 0; i < numRows; i++) {
    for (int j = (int)rowPtr[i]; j < rowPtr[i + 1]; j++) {
      CG_UINT curIndex = entries[j].col;

      if (startRow <= curIndex && curIndex <= stopRow) {
        entries[j].col -= startRow;
      } else {
        entries[j].col = extLocalIndex[bstFind(extLookup, curIndex)];
      }
    }
  }
}

/**
 * @brief Build the mapping of which local elements to send to each destination rank.
 * 
 * This function exchanges external element lists between ranks to determine which local
 * elements each rank needs to send. The result is the elementsToSend array, which maps
 * to local row indices that will be packed into the send buffer during each exchange.
 *
 * @param[in,out] c Communication structure to populate with send information
 * @param startRow First global row index owned by this rank
 * @param extLocalToGlobalReordered Global column indices needed by this rank (reordered)
 *
 * Algorithm:
 * 1. Allocate send buffer and elementsToSend array
 * 2. Post non-blocking receives from all destination ranks (they will send the global
 *    indices they need from us)
 * 3. Send our extLocalToGlobalReordered array to all source ranks (these are the global
 *    indices we need from them)
 * 4. Wait for receives to complete
 * 5. Convert received global indices to local indices by subtracting startRow
 * 
 * Result: c->elementsToSend[totalSendCount] contains local row indices to pack for sending,
 * organized by destination rank using c->sdispls and c->sendCounts
 */
static void buildElementsToSend(CommType *c, int startRow, int *extLocalToGlobalReordered)
{
  c->totalSendCount = 0;
  for (int i = 0; i < c->outdegree; i++) {
    c->totalSendCount += c->sendCounts[i];
  }

  c->sendBuffer =
      (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, c->totalSendCount * sizeof(CG_FLOAT));
  MPI_Request request[c->outdegree];
  c->elementsToSend   = (int *)allocate(ARRAY_ALIGNMENT, c->totalSendCount * sizeof(int));
  int *elementsToSend = c->elementsToSend;

  int j               = 0;

  for (int i = 0; i < c->outdegree; i++) {
    c->sdispls[i] = j;
    MPI_Irecv(elementsToSend + j,
        c->sendCounts[i],
        MPI_INT,
        c->destinations[i],
        MPI_TAG_EXCHANGE,
        MPI_COMM_WORLD,
        request + i);

    j += c->sendCounts[i];
  }

  j = 0;

  for (int i = 0; i < c->indegree; i++) {
    c->rdispls[i] = j;
    MPI_Send(extLocalToGlobalReordered + j,
        c->recvCounts[i],
        MPI_INT,
        c->sources[i],
        MPI_TAG_EXCHANGE,
        MPI_COMM_WORLD);

    j += c->recvCounts[i];
  }

  MPI_Waitall(c->outdegree, request, MPI_STATUSES_IGNORE);

  for (int i = 0; i < c->totalSendCount; i++) {
    elementsToSend[i] -= startRow;
  }

#ifdef VERBOSE
  for (int i = 0; i < c->size; i++) {
    FPRINTF(c->logFile, "Rank %d: number of elements %d\n", c->rank, c->totalSendCount);
    for (int j = 0; j < c->totalSendCount; j++) {
      if (i == c->rank) {
        FPRINTF(c->logFile, "\t[%d]: %d\n", j, elementsToSend[j]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif //VERBOSE
}
#endif //MPI

static void scanMM(
    MMMatrix *m, int startRow, int stopRow, int *entryCount, int *entryOffset)
{
  MMEntry *e = m->entries;
  int in     = 0;

  for (size_t i = 0; i < m->count; i++) {
    if (e[i].row == startRow && in == 0) {
      *entryOffset = (int)i;
      in           = 1;
    }
    if (e[i].row == (stopRow + 1)) {
      *entryCount = (int)(i - *entryOffset);
      break;
    }
    if (i == m->count - 1) {
      *entryCount = (int)(i - *entryOffset + 1);
      break;
    }
  }
}

static void dumpMMMatrix(CommType *c, MMMatrix *mm)
{
  MMEntry *entries = mm->entries;

  for (int i = 0; i < mm->count; i++) {
    FPRINTF(c->logFile, "%d %d: %f\n", entries[i].row, entries[i].col, entries[i].val);
  }
}

#ifdef _MPI
static void createMMEntryDatatype(MPI_Datatype *entryType)
{
  MMEntry dummy;
  MPI_Aint baseAddress;
  MPI_Aint displ[3];
  MPI_Get_address(&dummy, &baseAddress);
  MPI_Get_address(&dummy.row, &displ[0]);
  MPI_Get_address(&dummy.col, &displ[1]);
  MPI_Get_address(&dummy.val, &displ[2]);

  displ[0]              = MPI_Aint_diff(displ[0], baseAddress);
  displ[1]              = MPI_Aint_diff(displ[1], baseAddress);
  displ[2]              = MPI_Aint_diff(displ[2], baseAddress);

  int blocklengths[3]   = { 1, 1, 1 };
  MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_DOUBLE };
  MPI_Type_create_struct(3, blocklengths, displ, types, entryType);
  MPI_Type_commit(entryType);
}

static void calculateMMSendCounts(
    MMMatrix *m, int size, int totalNr, int *sendcounts, int *senddispls)
{
  int cursor = 0;
  for (int i = 0; i < size; i++) {
    int numRows  = sizeOfRank(i, size, totalNr);
    int startRow = cursor;
    cursor += numRows;
    int stopRow = cursor - 1;
    scanMM(m, startRow, stopRow, &sendcounts[i], &senddispls[i]);
    printf("Rank %d count %d displ %d start %d stop %d\n",
        i,
        sendcounts[i],
        senddispls[i],
        startRow,
        stopRow);
  }
}

/**
 * @brief Scan the local matrix to identify all external column references.
 *
 * This function examines every matrix entry to find column indices that reference rows
 * owned by other ranks (external elements). Each unique external is recorded once in
 * the extLocalToGlobal array and indexed in the extLookup binary search tree.
 *
 * @param c Communication structure (for error handling)
 * @param A The local matrix partition to scan
 * @param[out] extLookup Binary search tree mapping global column index → external array index
 * @param[out] extLocalToGlobal Array mapping external index → global column index
 * @return Number of unique external elements found
 *
 * Algorithm:
 * For each matrix entry (row, col):
 * - If col is outside local range [startRow, stopRow]:
 *   - Check extLookup to see if this global column was already seen
 *   - If new: insert into extLookup, add to extLocalToGlobal, increment counter
 *   - If already seen: skip (we only need each external once)
 */
static int identifyExternals(
    CommType *c, GMatrix *A, Bstree *extLookup, int *extLocalToGlobal)
{
  CG_UINT *rowPtr  = A->rowPtr;
  Entry *entries   = A->entries;
  CG_UINT numRows  = A->nr;
  CG_UINT startRow = A->startRow;
  CG_UINT stopRow  = A->stopRow;
  int extCount     = 0;

  for (int i = 0; i < numRows; i++) {
    for (CG_UINT j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      CG_UINT curIndex = entries[j].col;

      if (curIndex < startRow || curIndex > stopRow) {
        if (!bstExists(extLookup, curIndex)) {
          bstInsert(extLookup, curIndex, extCount);

          if (extCount < MAX_EXTERNAL) {
            extLocalToGlobal[extCount] = (int)curIndex;
          } else {
            commAbort(c, "Must increase MAX_EXTERNAL");
          }
          extCount++;
        }
      }
    }
  }
#ifdef VERBOSE
  printf("Rank %d: %d externals\n", c->rank, extCount);
#endif
  return extCount;
}

/**
 * @brief Determine which rank owns each external element.
 *
 * This function uses an MPI_Allgather to obtain each rank's starting row offset, then
 * determines ownership for each external element by finding which rank's range contains
 * the global index. It also counts how many elements are needed from each source rank.
 *
 * @param c Communication structure
 * @param startRow First global row owned by this rank
 * @param extLocalToGlobal Array mapping external index → global column index
 * @param extCount Number of external elements
 * @param[out] recvFromNeighbors Array tracking how many elements needed from each rank
 *                                (-1 if no communication, else count)
 * @param[out] extOwningRank Array mapping external index → owning MPI rank
 * @return Number of distinct source ranks (in-degree of communication graph)
 *
 * Algorithm:
 * 1. Use MPI_Allgather to collect startRow from all ranks
 * 2. For each external element:
 *    - Binary search through globalIndexOffsets to find owning rank
 *    - Record owner in extOwningRank
 *    - Update recvFromNeighbors count for that rank
 * 3. Count distinct source ranks
 */
static int findExternalOwningRanks(CommType *c,
    const int startRow,
    const int *extLocalToGlobal,
    const int extCount,
    int *recvFromNeighbors,
    int *extOwningRank)
{
  int size = c->size;

  for (int i = 0; i < size; i++) {
    recvFromNeighbors[i] = -1;
  }

  int globalIndexOffsets[size];
  int sourceCount = 0;

  MPI_Allgather(&startRow, 1, MPI_INT, globalIndexOffsets, 1, MPI_INT, MPI_COMM_WORLD);

  for (int i = 0; i < extCount; i++) {
    int globalIndex = extLocalToGlobal[i];

    for (int j = size - 1; j >= 0; j--) {
      if (globalIndexOffsets[j] <= globalIndex) {
        extOwningRank[i] = j;
        if (recvFromNeighbors[j] < 0) {
          recvFromNeighbors[j] = 1;
          sourceCount++;
        } else {
          recvFromNeighbors[j]++;
        }
        break;
      }
    }
  }

  return sourceCount;
}

/**
 * @brief Create MPI distributed graph topology with known incoming edges.
 *
 * This function sets up the MPI communication topology by specifying which ranks this
 * rank needs to receive from (sources). The MPI topology will automatically determine
 * the reverse direction (which ranks need data from us). Edge weights communicate the
 * message sizes (number of elements to receive from each source).
 *
 * @param[in,out] c Communication structure; c->communicator will be set
 * @param sourceCount Number of source ranks (in-degree)
 * @param recvFromNeighbors Array tracking receive counts from each rank
 *
 * Algorithm:
 * 1. Build arrays of source ranks and weights from recvFromNeighbors
 * 2. Create incoming edge arrays (degrees=1, destinations=this rank)
 * 3. Call MPI_Dist_graph_create to establish the topology
 *
 * Result: c->communicator is initialized with the distributed graph topology
 */
static void setupTopology(
    CommType *c, const int sourceCount, const int *recvFromNeighbors)
{
  int sources[sourceCount];
  int degrees[sourceCount];
  int destinations[sourceCount];
  int weights[sourceCount];
  int cursor = 0;
  int size   = c->size;

  for (int i = 0; i < size; i++) {
    if (recvFromNeighbors[i] > 0) {
      sources[cursor]   = i;
      weights[cursor++] = recvFromNeighbors[i];
    }
  }

  for (int i = 0; i < sourceCount; i++) {
    degrees[i]      = 1;
    destinations[i] = c->rank;
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

/**
 * @brief Retrieve the complete communication topology from MPI.
 *
 * After the distributed graph topology is created, this function queries it to get
 * both the incoming and outgoing communication pattern. This includes source/destination
 * rank IDs and message counts in each direction.
 *
 * @param[in,out] c Communication structure to populate with topology information
 *
 * Algorithm:
 * 1. Call MPI_Dist_graph_neighbors_count to get in-degree and out-degree
 * 2. Allocate arrays for sources, destinations, and counts
 * 3. Call MPI_Dist_graph_neighbors to populate the arrays
 * 
 * Result: c is populated with:
 * - indegree, outdegree: Number of incoming/outgoing edges
 * - sources[indegree]: Ranks we receive from
 * - recvCounts[indegree]: Elements to receive from each source
 * - destinations[outdegree]: Ranks we send to
 * - sendCounts[outdegree]: Elements to send to each destination
 * - rdispls, sdispls: Displacement arrays (allocated but not yet set)
 */
static void retrieveTopology(CommType *c)
{
  int weighted;
  MPI_Dist_graph_neighbors_count(c->communicator, &c->indegree, &c->outdegree, &weighted);

#ifdef VERBOSE
  printf("Rank %d: In %d Out %d Weighted %d\n",
      c->rank,
      c->indegree,
      c->outdegree,
      weighted);
#endif

  c->sources      = (int *)allocate(ARRAY_ALIGNMENT, c->indegree * sizeof(int));
  c->recvCounts   = (int *)allocate(ARRAY_ALIGNMENT, c->indegree * sizeof(int));
  c->rdispls      = (int *)allocate(ARRAY_ALIGNMENT, c->indegree * sizeof(int));
  c->destinations = (int *)allocate(ARRAY_ALIGNMENT, c->outdegree * sizeof(int));
  c->sendCounts   = (int *)allocate(ARRAY_ALIGNMENT, c->outdegree * sizeof(int));
  c->sdispls      = (int *)allocate(ARRAY_ALIGNMENT, c->outdegree * sizeof(int));

  MPI_Dist_graph_neighbors(c->communicator,
      c->indegree,
      c->sources,
      c->recvCounts,
      c->outdegree,
      c->destinations,
      c->sendCounts);
}

#endif //MPI

/**
 * @brief Print application banner and configuration information.
 *
 * Prints the application banner along with matrix format, precision, and
 * integer type configuration. This is typically called once at startup.
 */
static void printConfigInfo(void)
{
  printf(BANNER "\n");
  printf("Using %s matrix format, %s precision floats and integer type %s\n\n",
      FMT,
      PRECISION_STRING,
      UINT_STRING);
}

#if defined(VERBOSE_AFFINITY) && defined(_OPENMP)
/**
 * @brief Print detailed thread affinity information.
 *
 * Prints detailed information about which CPU core each thread is running on,
 * along with process and thread IDs. This function should be called from within
 * an OpenMP parallel region with a critical section to avoid garbled output.
 * 
 * @param rank MPI rank of this process
 * @param host Hostname where the process is running
 * @param masterPid Process ID of the master thread
 */
static void printAffinityInfo(int rank, const char *host, pid_t masterPid)
{
  printf("Rank %d Thread %d running on Node %s core %d with pid %d and tid %d\n",
      rank,
      omp_get_thread_num(),
      host,
      sched_getcpu(),
      masterPid,
      gettid());
  affinity_getmask();
}
#endif

/**
 * @brief Print startup banner with system and configuration information.
 *
 * This function prints a comprehensive startup banner that includes:
 * - Application banner and compile-time configuration
 * - MPI rank information (if running with multiple processes)
 * - OpenMP thread count (if compiled with OpenMP support)
 * - Per-process hostname and PID information
 * - Detailed affinity information (if VERBOSE_AFFINITY is enabled)
 *
 * The output is synchronized across MPI ranks to ensure readable output.
 *
 * @param c Communication structure containing rank and size information
 */
void commPrintBanner(CommType *c)
{
  int rank = c->rank;
  int size = c->size;

  char host[_POSIX_HOST_NAME_MAX];
  pid_t masterPid = getpid();
  if (gethostname(host, _POSIX_HOST_NAME_MAX) != 0) {
    snprintf(host, sizeof(host), "unknown");
  }

  // Print banner and configuration (master only in MPI mode, or always in single-process mode)
  if (commIsMaster(c)) {
    printConfigInfo();

    if (size > 1) {
      printf("MPI parallel using %d ranks\n", size);
    } else {
      printf("Running with only one process!\n");
    }

#ifdef _OPENMP
    printf("OpenMP enabled using %d threads\n", omp_get_max_threads());
#endif
  }

  // In MPI mode, synchronize and print per-rank information
  if (size > 1) {
    commBarrier();

    for (int i = 0; i < size; i++) {
      if (i == rank) {
        printf("Process with rank %d running on Node %s with pid %d\n",
            rank,
            host,
            masterPid);
      }

#if defined(VERBOSE_AFFINITY) && defined(_OPENMP)
#pragma omp parallel
      {
#pragma omp critical
        {
          printAffinityInfo(rank, host, masterPid);
        }
      }
#endif

      commBarrier();
    }
  }
#if defined(VERBOSE_AFFINITY) && defined(_OPENMP)
  // In single-process mode, print affinity info if requested
  else {
#pragma omp parallel
    {
#pragma omp critical
      {
        printAffinityInfo(rank, host, masterPid);
      }
    }
  }
#endif
}

void commDistributeMatrix(CommType *c, MMMatrix *m, MMMatrix *mLocal)
{
#ifdef _MPI
  int rank = c->rank;
  int size = c->size;
  int totalCounts[2];

  if (rank == 0) {
    totalCounts[0] = m->nr;
    totalCounts[1] = m->nnz;
  }

  int result = MPI_Bcast(&totalCounts, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    commAbort(c, "MPI_Bcast failed during matrix distribution");
  }
  int totalNr  = totalCounts[0];
  int totalNnz = totalCounts[1];

  MPI_Datatype entryType;
  createMMEntryDatatype(&entryType);

  int sendcounts[size];
  int senddispls[size];

  if (commIsMaster(c)) {
    calculateMMSendCounts(m, size, totalNr, sendcounts, senddispls);
  }

  int count;
  result = MPI_Scatter(sendcounts, 1, MPI_INT, &count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    commAbort(c, "MPI_Scatter failed during matrix distribution");
  }

  mLocal->count    = count;
  mLocal->totalNr  = totalNr;
  mLocal->totalNnz = totalNnz;
  mLocal->entries  = (MMEntry *)allocate(ARRAY_ALIGNMENT, count * sizeof(MMEntry));

  result           = MPI_Scatterv(m->entries,
      sendcounts,
      senddispls,
      entryType,
      mLocal->entries,
      count,
      entryType,
      0,
      MPI_COMM_WORLD);
  if (result != MPI_SUCCESS) {
    commAbort(c, "MPI_Scatterv failed during matrix distribution");
  }

  mLocal->startRow = mLocal->entries[0].row;
  mLocal->stopRow  = mLocal->entries[count - 1].row;
  mLocal->nr       = mLocal->stopRow - mLocal->startRow + 1;
  mLocal->nnz      = count;

  if (count > 0) {
    printf("Rank %d count %zu start %d stop %d\n",
        rank,
        mLocal->count,
        mLocal->startRow,
        mLocal->stopRow);
  } else {
    commAbort(c, "No matrix entries received");
  }

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

/**
 * @brief Transform distributed matrix to enable efficient single-exchange communication.
 * 
 * This function is the core of the MPI communication setup. It reorganizes the local
 * matrix and establishes all necessary data structures to enable efficient communication
 * during iterative solvers (e.g., CG, SpMV operations). The key transformation is to
 * extend the local RHS vector to include space for external elements (values owned by
 * other ranks that are needed locally), then remap all matrix column indices to reference
 * this extended local vector using 0-based indexing.
 * 
 * **Pre-conditions:**
 * - Matrix m has been distributed across ranks (rows partitioned)
 * - m->startRow and m->stopRow define this rank's row ownership range
 * - Matrix entries use global column indexing
 * - CommType c has been initialized with MPI rank and size
 * 
 * **Post-conditions:**
 * - Matrix column indices are remapped to local 0-based indices
 * - m->nc is extended by extCount (now = original nc + extCount)
 * - CommType c is fully populated with:
 *   * MPI distributed graph communicator
 *   * Source/destination rank lists and message counts
 *   * elementsToSend array for packing send buffers
 *   * sendBuffer and displacement arrays for MPI_Neighbor_alltoallv
 * - The local RHS vector can be used with indices [0, nr+extCount-1]
 * 
 * **Four-Step Algorithm:**
 * 
 * 1. **Identify Externals**: Scan matrix to find all column indices referencing non-local
 *    rows. Build extLocalToGlobal mapping and extLookup binary search tree.
 * 
 * 2. **Build Communication Topology**: Determine which ranks own the external elements
 *    (sources we receive from). Use MPI_Dist_graph_create to establish topology, which
 *    also determines destinations (ranks we send to). This creates a sparse communication
 *    pattern optimized for the matrix structure.
 * 
 * 3. **Reorder and Localize**: Reorder externals so elements from the same source rank
 *    are consecutive in memory. Remap all matrix column indices from global to local
 *    (0-based) indexing, where indices [0, nr-1] are local and [nr, nr+extCount-1] are
 *    external.
 * 
 * 4. **Build Send Mapping**: Exchange external lists with neighbors to determine which
 *    local elements to send to each destination. Build elementsToSend array for efficient
 *    send buffer packing during the exchange phase.
 * 
 * **Result**: After this function, communication during each iteration reduces to:
 * ```c
 * // Pack send buffer
 * for (i = 0; i < totalSendCount; i++)
 *     sendBuffer[i] = x[elementsToSend[i]];
 * 
 * // Exchange with neighbors
 * MPI_Neighbor_alltoallv(sendBuffer, ..., x+numRows, ..., communicator);
 * 
 * // All matrix operations now use local indices only
 * y[i] += A->val[j] * x[A->colInd[j]];  // colInd[j] is now local!
 * ```
 * 
 * @param[in,out] c Communication structure to populate with topology and send/recv info
 * @param[in,out] m Distributed matrix to localize (column indices will be remapped)
 * 
 * @note This function is only active when compiled with _MPI defined
 * @note Uses MAX_EXTERNAL as a compile-time limit; will abort if exceeded
 * 
 * Complexity: O(nnz_local × log(extCount) + extCount × log(numRanks) + totalSendCount)
 */
void commLocalization(CommType *c, GMatrix *m)
{
#ifdef _MPI
  int rank = c->rank;
  int size = c->size;

#ifdef VERBOSE
  FPRINTF(c->logFile,
      "Rank %d of %d: num columns %d owns %d rows: %d to %d of total %d\n",
      rank,
      size,
      m->nc,
      m->nr,
      m->startRow,
      m->stopRow,
      m->totalNr);
#endif

  /***********************************************************************
   *    Step 1: Identify externals and create external lookup
   *    Scan matrix to find all unique column indices that reference non-local
   *    rows. Build a binary search tree for fast duplicate detection and an
   *    array mapping external index (0-based) to global column index.
   ************************************************************************/
  Bstree *extLookup     = bstNew();
  int *extLocalToGlobal = (int *)allocate(ARRAY_ALIGNMENT, MAX_EXTERNAL * sizeof(int));
  int extCount          = identifyExternals(c, m, extLookup, extLocalToGlobal);

  /***********************************************************************
   *    Step 2:  Build dist Graph topology and init incoming edges
   *    Determine which rank owns each external element, then create MPI
   *    distributed graph topology. The topology setup specifies incoming edges
   *    (ranks we receive from), and MPI automatically determines outgoing edges
   *    (ranks we send to) based on symmetry of the communication pattern.
   ************************************************************************/
  int *extOwningRank     = (int *)allocate(ARRAY_ALIGNMENT, extCount * sizeof(int));
  int *recvFromNeighbors = (int *)allocate(ARRAY_ALIGNMENT, size * sizeof(int));

  // Find which rank owns each external and count how many we need from each source
  int sourceCount = findExternalOwningRanks(
      c, (int)m->startRow, extLocalToGlobal, extCount, recvFromNeighbors, extOwningRank);

  // Create MPI distributed graph with incoming edges (sources + receive counts)
  setupTopology(c, sourceCount, recvFromNeighbors);

  // Query the topology to get both incoming and outgoing communication pattern
  retrieveTopology(c);

  free(recvFromNeighbors);

  /***********************************************************************
   *    Step 3:  Reorder externals and localize matrix
   *    Reorganize external elements so those from the same source rank are
   *    consecutive. This enables efficient MPI_Neighbor_alltoallv communication.
   *    Then remap all matrix column indices from global to local (0-based).
   ************************************************************************/
  int *extLocalToGlobalReordered =
      (int *)allocate(ARRAY_ALIGNMENT, extCount * sizeof(int));

  {
    int *extLocalIndex = (int *)allocate(ARRAY_ALIGNMENT, extCount * sizeof(int));
    int numRows        = m->nr;

    // Reorder externals: assign consecutive local RHS indices to externals from same rank
    // extLocalIndex[old_ext_idx] = new_local_rhs_idx (in range [numRows, numRows+extCount-1])
    reorderExternals(numRows, extCount, extLocalIndex, extOwningRank);

    // Build reordered mapping: extLocalToGlobalReordered[new_ext_idx] = global_col_idx
    // The new external index is (extLocalIndex[i] - numRows) for the i-th original external
    for (int i = 0; i < extCount; i++) {
      extLocalToGlobalReordered[extLocalIndex[i] - numRows] = extLocalToGlobal[i];
    }

    // Remap matrix column indices: global -> local (using extLocalIndex for externals)
    localizeMatrix(m, extLookup, extLocalIndex);

    // Clean up temporary structures
    free(extLocalIndex);
    free(extLocalToGlobal);
    bstFree(extLookup);
  }

#ifdef VERBOSE
  FPRINTF(c->logFile, "STEP 3 \n");
  FPRINTF(c->logFile, "Rank %d of %d: %d externals\n", rank, size, extCount);

  for (int i = 0; i < extCount; i++) {
    FPRINTF(c->logFile,
        "Rank %d of %d: external[%d] owned by %d\n",
        rank,
        size,
        i,
        extOwningRank[i]);
  }
#endif // VERBOSE

  // Extend the number of columns to include external elements
  // Local RHS vector is now [0..nr-1]: local, [nr..nr+extCount-1]: external
  m->nc = m->nc + extCount;
  free(extOwningRank);

  /***********************************************************************
   *    Step 4:  Build global index list for external communication
   *    Exchange external lists with neighbors to determine which local
   *    elements to send. Build elementsToSend array for efficient packing.
   ************************************************************************/
  buildElementsToSend(c, (int)m->startRow, extLocalToGlobalReordered);

  free(extLocalToGlobalReordered);
#endif
}

void commExchange(CommType *c, CG_UINT numRows, CG_FLOAT *x)
{
#ifdef _MPI
  CG_FLOAT *sendBuffer = c->sendBuffer;
  CG_FLOAT *externals  = x + numRows;
  int *elementsToSend  = c->elementsToSend;

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

void commReduction(CG_FLOAT *v, int op)
{
#ifdef _MPI
  if (op == MAX) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_FLOAT_TYPE, MPI_MAX, MPI_COMM_WORLD);
  } else if (op == SUM) {
    MPI_Allreduce(MPI_IN_PLACE, v, 1, MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
}

void commPrintConfig(
    CommType *c, CG_UINT nr, CG_UINT nnz, CG_UINT startRow, CG_UINT stopRow)
{
#ifdef _MPI
  FFLUSH(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if (commIsMaster(c)) {
    printf("Communication setup:\n");
  }

  for (int i = 0; i < c->size; i++) {
    if (i == c->rank) {
      printf("Rank %d has %u rows (%u to %u) and %u nnz\n",
          c->rank,
          nr,
          startRow,
          stopRow,
          nnz);

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
          FFLUSH(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      FFLUSH(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

void commMatrixDump(CommType *c, Matrix *m)
{
  int rank = c->rank;
  int size = c->size;

#ifdef CRS
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;
  CG_UINT *colInd = m->colInd;
  CG_FLOAT *val   = m->val;

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

        for (int rowEntry = (int)rowPtr[rowID]; rowEntry < rowPtr[rowID + 1];
            rowEntry++) {
          printf("[%d]:%.2f ", colInd[rowEntry], val[rowEntry]);
        }

        printf("\n");
      }
      FFLUSH(stdout);
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
#endif /* ifdef CRS */
#ifdef SCS
  printf("m->startRow = %d\n", m->startRow);
  printf("m->stopRow = %d\n", m->stopRow);
  printf("m->totalNr = %d\n", m->totalNr);
  printf("m->totalNnz = %d\n", m->totalNnz);
  printf("m->nr = %d\n", m->nr);
  printf("m->nc = %d\n", m->nc);
  printf("m->nnz = %d\n", m->nnz);
  printf("m->C = %d\n", m->C);
  printf("m->sigma = %d\n", m->sigma);
  printf("m->nChunks = %d\n", m->nChunks);
  printf("m->nrPadded = %d\n", m->nrPadded);

  // Dump permutation arrays
  printf("oldToNewPerm: ");
  for (int i = 0; i < m->nr; ++i) {
    printf("%d, ", m->oldToNewPerm[i]);
  }
  printf("\n");
  printf("newToOldPerm: ");
  for (int i = 0; i < m->nr; ++i) {
    printf("%d, ", m->newToOldPerm[i]);
  }
  printf("\n");

  // Dump chunk data
  printf("chunkLens: ");
  for (int i = 0; i < m->nChunks; ++i) {
    printf("%d, ", m->chunkLens[i]);
  }
  printf("\n");
  printf("chunkPtr: ");
  for (int i = 0; i < m->nChunks + 1; ++i) {
    printf("%d, ", m->chunkPtr[i]);
  }
  printf("\n");

  // Dump matrix data
  printf("colInd: ");
  for (int i = 0; i < m->nElems; ++i) {
    printf("%d, ", m->colInd[i]);
  }
  printf("\n");
  printf("val: ");
  for (int i = 0; i < m->nElems; ++i) {
    printf("%f, ", m->val[i]);
  }
  printf("\n");
#endif /* ifdef SCS */
}

void commVectorDump(CommType *c, CG_FLOAT *v, CG_UINT size, char *name)
{
  for (int i = 0; i < c->size; i++) {
    if (i == c->rank) {
      FPRINTF(c->logFile, "Vector %s Rank %d of %d\n", name, c->rank, c->size);
      for (int j = 0; j < size; j++) {
        FPRINTF(c->logFile, "\telement[%d] %f\n", j, v[j]);
      }
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

void commGMatrixDump(CommType *c, GMatrix *m)
{
  int rank        = c->rank;
  int size        = c->size;
  CG_UINT numRows = m->nr;
  CG_UINT *rowPtr = m->rowPtr;
  Entry *entries  = m->entries;

  FPRINTF(c->logFile,
      "Matrix: %d total non zeroes, total number of rows %d\n",
      m->totalNnz,
      m->totalNr);
  FPRINTF(c->logFile,
      "Matrix: %d local non zeroes, local number of rows %d\n",
      m->nnz,
      m->nr);

  for (int i = 0; i < size; i++) {
    if (i == rank) {
      FPRINTF(c->logFile, "Rank %d: number of rows %d\n", rank, numRows);

      for (int rowID = 0; rowID < numRows; rowID++) {
        FPRINTF(c->logFile, "Row [%d]: ", rowID);

        for (int rowEntry = (int)rowPtr[rowID]; rowEntry < rowPtr[rowID + 1];
            rowEntry++) {
          FPRINTF(c->logFile, "[%d]:%.2f ", entries[rowEntry].col, entries[rowEntry].val);
        }

        FPRINTF(c->logFile, "\n");
      }
      FFLUSH(stdout);
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

void commInit(CommType *c, int argc, char **argv)
{
#ifdef _MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &(c->rank));
  MPI_Comm_size(MPI_COMM_WORLD, &(c->size));

  // Initialize pointers to NULL to avoid issues in finalize if abort happens
  // early
  c->sources        = NULL;
  c->recvCounts     = NULL;
  c->rdispls        = NULL;
  c->destinations   = NULL;
  c->sendCounts     = NULL;
  c->sdispls        = NULL;
  c->elementsToSend = NULL;
  c->sendBuffer     = NULL;
#else
  c->rank = 0;
  c->size = 1;
#endif
#ifdef VERBOSE
  char filename[MAXSTRLEN];
  snprintf(filename, sizeof(filename), "out-%d.txt", c->rank);
  c->logFile = fopen(filename, "we");
  if (c->logFile == NULL) {
    printf("Warning: Could not open log file %s\n", filename);
  }
#endif
}

void commAbort(CommType *c, char *msg)
{
  printf("Abort: %s\n", msg);
#if defined(_MPI)
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
#ifdef VERBOSE
  if (c->logFile != NULL) {
    FCLOSE(c->logFile);
  }
#endif
  exit(EXIT_FAILURE);
}

void commFinalize(CommType *c)
{
#ifdef _MPI
  if (c->sources != NULL) {
    free(c->sources);
  }
  if (c->recvCounts != NULL) {
    free(c->recvCounts);
  }
  if (c->rdispls != NULL) {
    free(c->rdispls);
  }
  if (c->destinations != NULL) {
    free(c->destinations);
  }
  if (c->sendCounts != NULL) {
    free(c->sendCounts);
  }
  if (c->sdispls != NULL) {
    free(c->sdispls);
  }
  if (c->elementsToSend != NULL) {
    free(c->elementsToSend);
  }
  if (c->sendBuffer != NULL) {
    free(c->sendBuffer);
  }
  MPI_Finalize();
#endif

#ifdef VERBOSE
  FCLOSE(c->logFile);
#endif
}
