# Overall MPI algorithm

## Exchange

### Idea

Reduce communication to one exchange routine. All other steps are mostly local
without any MPI call. This is achieved by appending the column indices that are
not available on the current rank to the end of the local vector and rewrite the
column indices in the matrix accordingly.

### Required data

- `numNeighbors` The number of communication neighbors (same for send and
  receive)
- `neighbors` Array with communication neighbor ranks
- `recvCount` Array with number of external values we receive from each neighbor
- `sendCount` Array with number of values we need to send to each neighbor
- `sendBuffer` Send buffer holding externals for all ranks
- `elementsToSend` Index list with all values the current rank needs to send
- `totalSendCount` Total number of values the current rank needs to send

### Steps

- Post receives of externals. External are directly copied into the appended external
  portion of the target vector. No additional copying from a receive buffer is
  required. Non blocking receive calls are used.
- Copy all values the current rank needs to send to all neighbor ranks into the
  one send buffer.
- Send the requested values to every neighbor rank. Blocking Send calls are
  used.
- Wait for all non blocking receive calls to complete.

## Partitioning

The matrix has to be distributed. Every rank gets a consecutive number of rows.
Other options:

- Take into account total non zeroes per rank
- Take into account total communication data volume

### Step 1: Identify externals and create lookup maps

Initially the column indices are in global ordering. The indices need to be
converted to a local column ordering.

- Determine if an index targets a local or external value
- If local, subtract start row id from index to get local index
- If external, check if it was already taken care for:

  - If yes, do nothing
  - otherwise
    - add it to a list of external indices
    - find out which rank owns the value
    - set up communication for SpMVM

- `externalCount`: Number of externals for current rank
- `externals` SIZE `externalCount` temporary: Lookup if index was already seen before. Map from global index to local id in external lookups.
- `externalIndex` SIZE `externalCount` temporary: List of global ids for all
  external elements

- Mark external in local column index by negating
- Mark external in external lookup by setting to the local id
- Add global index to external index list

### Step 2 Identify owning rank for externals

Determine which processors are required to resolve all externals. Gather the
start row offsets for every ranks first.

- `externalRank` SIZE `externalCount` temporary: Store for every external
  the owning rank

### Step 3 Build and apply index mapping

Subroutine `buildIndexMapping`:
Generate a local ordering for externals, starting at the end of the local
elements. Give all indices belonging to same rank consecutive ids. Map all
external column ids in matrix to new local index.

- `externalLocalIndex` SIZE `externalCount` temporary: Local compacted index for
  every external. Consecutive ids for elements owned by same rank.
- `externalReordered` SIZE `externalCount` temporary: Map from external
  new local id to global id

### Step 4 Build list of neighbor ranks

Subroutine `buildNeighborlist`:
Generate a list that encodes how many ranks need values from a rank and the
total number of values that need to be sent.
Make a list of the neighbors that will send information to update our
external elements (in the order that we will receive this information).
Create one neighbor list for both send and receive.

- `neighborCount` persistent: Number of communication neighbors
- `neighbors` SIZE `neighborCount` persistent: All ranks that need to communicate
  with the current rank

### Step 5 Build message counts for all communication partners

Subroutine `buildMessageCounts`:
Setup send and receive counts for all neighbors of rank.

- `recvCount` SIZE `neighborCount` persistent: Receive message sizes for every
  communication partner
- `sendCount` SIZE `neighborCount` persistent: Send message sizes for every
  communication partner

### Step 6 Build global index list for external communication

Subroutine `buildElementsToSend`:
Send the global external indices in the order the current rank requires them.
On the sending rank there is one index list for all neighboring ranks.

- `elementsToSend` SIZE `totalSendCount` persistent: List with all global
  indices the current rank needs to send to other ranks in the exact order they
  need it in their external part of the vector
