# Overall MPI algorithm

## Exchange

### Idea

Reduce communication to one exchange routine. All other steps are mostly local
without any MPI call. This is achieved by appending the column indices that are
not available on the current rank to the end of the local vector and rewrite the
column indices in the matrix accordingly.

## Communication data structure

- `totalSendCount`: Total number of elements to send to all receivers
- `elementsToSend[totalSendCount]`: Local element ids to send to all receivers
- `indegree`: Number of ranks we receive messages from
- `outdegree`: Number of ranks we send messages to
- `sources[indegree]`: List of ranks we receive messages from
- `recvCounts[indegree]`: Message counts for messages we receive from senders
- `rdispls[indegree]`: Displacements in receive buffer
- `destinations[outdegree]`: List of ranks we send messages to
- `sendCounts[outdegree]`: Message counts for messages we send to receivers
- `sdispls[outdegree]`: Displacements in send buffer

## Partitioning

The matrix has to be distributed. Every rank gets a consecutive number of rows.
Other options (not implemented):

- Take into account total non zeroes per rank
- Take into account total communication data volume

## Localization

### Overview and Purpose

The localization process transforms a distributed sparse matrix to enable efficient
communication during iterative solvers. The key idea is to **reorganize the local
data structures** such that:

1. **Communication is reduced to a single exchange operation** per iteration
2. **All column indices in the local matrix are converted to local indices** 
   (referencing either local or external elements)
3. **External elements are appended to the local vector** in a communication-friendly order
4. **Elements from the same source rank are stored consecutively** to optimize MPI operations

**Performance Benefits:**
- Eliminates scattered memory accesses during communication
- Enables efficient use of `MPI_Neighbor_alltoallv` with the distributed graph topology
- Reduces communication overhead by batching messages per neighbor
- Allows all SpMV operations to use purely local indexing

**Overall Strategy:**
After localization, the local RHS vector `x` is extended from size `nr` (number of local rows)
to size `nr + extCount` (local rows + external elements). The matrix column indices are
remapped so that indices [0, nr-1] reference local elements and indices [nr, nr+extCount-1]
reference external elements that will be received from neighboring ranks.

### Step 1: Identify Externals and Create External Lookup

**Purpose:** Scan the local matrix to discover all column indices that reference non-local
rows (i.e., rows owned by other ranks). These are the "external" elements we need to receive
during each communication phase.

**Algorithm:**
- Iterate through all matrix entries in the local row partition
- For each column index, determine if it references a local row (`startRow ≤ col ≤ stopRow`)
- If external (outside local row range):
  - Check if this global column index was already encountered using the binary search tree
  - If new: insert into `extLookup`, add to `extLocalToGlobal`, increment `extCount`
  - If already seen: skip (we only need each external element once)

**Complexity:** O(nnz_local × log(extCount)) where nnz_local is the number of non-zeros
in the local matrix partition.

**Variables:**

- `extCount` (int): Total number of unique external elements required by this rank
- `extLookup` (Bstree*): Binary search tree for O(log n) lookup to check if a global
  column index has already been identified as external. Maps global column index → 
  position in extLocalToGlobal array
- `extLocalToGlobal` (int[MAX_EXTERNAL]): Maps from zero-based external index → 
  global column index. This is the initial ordering as externals are encountered
  during the matrix scan.

### Step 2: Build Distributed Graph Topology

**Purpose:** Establish the communication pattern by determining which ranks need to send
data to this rank (sources) and which ranks this rank needs to send data to (destinations).
MPI's distributed graph topology is used to efficiently represent this sparse communication
pattern.

**Algorithm:**
1. **Gather global row offsets:** Use `MPI_Allgather` to collect each rank's starting
   row index, enabling determination of which rank owns each global row
2. **Find owners of externals:** For each external element, binary search through the
   gathered offsets to determine which rank owns that row
3. **Count receives per neighbor:** Track how many elements we need from each source rank
   in the `recvFromNeighbors` array
4. **Create MPI topology:** Call `MPI_Dist_graph_create` with incoming edges (sources),
   using edge weights to communicate the receive counts
5. **Retrieve full topology:** Use `MPI_Dist_graph_neighbors` to get both incoming
   (sources, recvCounts) and outgoing (destinations, sendCounts) communication pattern

**Complexity:** 
- O(extCount × log(numRanks)) for finding owners
- O(numRanks) for MPI collective operations

**Variables:**

- `extOwningRank` (int[extCount]): Rank that owns each external element (in original
  external ordering). Maps external index → owning MPI rank.
- `sourceCount` (int): Number of distinct ranks from which this rank receives data
  (in-degree of the communication graph)
- `recvFromNeighbors` (int[size]): Temporary array tracking how many elements are
  needed from each rank. Index is rank number, value is count (-1 if no communication
  with that rank).
- After topology retrieval, CommType structure is populated with:
  - `indegree`: Number of source ranks
  - `outdegree`: Number of destination ranks  
  - `sources[indegree]`: Array of source rank IDs
  - `recvCounts[indegree]`: Number of elements to receive from each source
  - `destinations[outdegree]`: Array of destination rank IDs
  - `sendCounts[outdegree]`: Number of elements to send to each destination

### Step 3: Reorder Externals and Localize Matrix

**Purpose:** Reorganize external elements so that all elements received from the same
source rank are stored consecutively in memory. This enables efficient MPI communication
and cache-friendly memory access patterns.

**Algorithm:**
1. **Reorder externals by owning rank:**
   - For each unique owning rank encountered in `extOwningRank`:
     - Assign consecutive local indices (starting from `numRows`) to all externals
       owned by that rank
     - Update `extLocalIndex` to map from original external index → new local RHS index
     - Build `extLocalToGlobalReordered` with the new ordering
2. **Remap matrix column indices:**
   - For each matrix entry:
     - If column references a local row: convert to zero-based local index (col - startRow)
     - If column references an external: use `extLookup` to find external index, then
       use `extLocalIndex[external_idx]` to get the new local RHS index

**Complexity:** O(extCount² / numRanks_avg) for reordering + O(nnz_local × log(extCount))
for matrix remapping

**Why reordering is necessary:** Without reordering, externals from different ranks would
be interleaved in memory. After reordering, the external portion of the RHS vector has
structure: [elements from rank A][elements from rank B][elements from rank C]..., which
aligns perfectly with how `MPI_Neighbor_alltoallv` will receive the data.

**Variables:**

- `extLocalToGlobalReordered` (int[extCount]): Maps from reordered zero-based external
  index → global column index. After reordering, externals from the same source rank
  are consecutive.
- `extLocalIndex` (int[extCount]): Temporary mapping from original external index →
  new local RHS vector index (in range [numRows, numRows+extCount-1])

### Step 4: Build Global Index List for External Communication

**Purpose:** Each rank needs to know which local elements to pack into the send buffer
for each destination rank. This step establishes the mapping from global column indices
(that destinations need) to local row indices (that this rank owns).

**Algorithm:**
1. **Exchange external lists:** Each rank sends its `extLocalToGlobalReordered` (the
   global indices it needs) to all source ranks it receives from
2. **Receive and convert to local indices:** Each rank receives global indices from
   its destination ranks and converts them to local row indices by subtracting `startRow`
3. **Store in elementsToSend:** The resulting array contains local indices to pack,
   organized by destination rank

**Complexity:** O(totalSendCount) where totalSendCount is the total number of elements
this rank sends to all neighbors

**Result:** The `CommType` structure now contains `elementsToSend[totalSendCount]`,
which is used during the exchange phase to efficiently pack the send buffer:
```c
for (i = 0; i < totalSendCount; i++)
    sendBuffer[i] = x[elementsToSend[i]];
```

**Variables:**

- `elementsToSend` (int[totalSendCount]): Local row indices to send, organized by
  destination rank. For each destination rank at offset `sdispls[i]`, the next
  `sendCounts[i]` elements specify which local rows to send.
- `totalSendCount` (int): Sum of all sendCounts - total number of elements to send
  to all destination ranks

### Final Data Structure Layout

After localization, the local RHS vector and matrix have the following structure:

**RHS Vector `x`:**
```
[0 ... nr-1]: Local elements (owned by this rank)
[nr ... nr+extCount-1]: External elements (grouped by source rank)
```

**Matrix Column Indices:**
- All column indices now reference the local RHS vector using 0-based indexing
- Local references: index ∈ [0, nr-1]
- External references: index ∈ [nr, nr+extCount-1]

**Communication Arrays:**
- `elementsToSend`: Which local elements to send (indices into local RHS)
- `recvCounts`, `rdispls`: How to receive external data
- `sendCounts`, `sdispls`: How to send local data to other ranks
