/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifdef _MPI
#include <stdio.h>

#include "mpi.h"

#include "allocate.h"
#include "matrixBinfile.h"
#include "util.h"

#define HEADERSIZE 24

static int sizeOfRank(int rank, int size, int N)
{
  return N / size + ((N % size > rank) ? 1 : 0);
}

static void createEntrytype(MPI_Datatype *entryType)
{
  MPI_Aint displ[2];
  FEntryType dummy;
  MPI_Aint baseAddress;
  MPI_Get_address(&dummy, &baseAddress);
  MPI_Get_address(&dummy.col, &displ[0]);
  MPI_Get_address(&dummy.val, &displ[1]);
  displ[0]              = MPI_Aint_diff(displ[0], baseAddress);
  displ[1]              = MPI_Aint_diff(displ[1], baseAddress);

  int lengths[2]        = { 1, 1 };
  MPI_Datatype types[2] = { MPI_UNSIGNED, MPI_FLOAT };
  MPI_Type_create_struct(2, lengths, displ, types, entryType);
  MPI_Type_commit(entryType);
}

void matrixBinWrite(GMatrix *m, CommType *c, char *filename)
{
  MPI_File fh;

  if (c->size > 1) {
    fprintf(stderr, "ERROR: Matrix writing only supported for single rank\n");
    return;
  }

  MPI_File_open(
      MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

  if (commIsMaster(c)) {
    printf("Writing matrix to %s\n", filename);
  }

  char header[HEADERSIZE] = "# SparseBench DataFile";
  MPI_File_set_view(fh, 0, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);

  if (commIsMaster(c)) {
    MPI_File_write(fh, header, HEADERSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_Offset disp;
  MPI_File_sync(fh);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_get_size(fh, &disp);
  MPI_File_set_view(fh, disp, MPI_UNSIGNED, MPI_UNSIGNED, "native", MPI_INFO_NULL);
  if (commIsMaster(c)) {
    // FIXME: convert to unsigned int in case CG_UINT is different
    MPI_File_write(fh, &m->totalNr, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
    MPI_File_write(fh, &m->totalNnz, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
    MPI_File_write(fh, m->rowPtr, m->totalNr + 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  }
  MPI_Datatype entryType;
  createEntrytype(&entryType);
  MPI_File_sync(fh);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_get_size(fh, &disp);
  MPI_File_set_view(fh, disp, entryType, entryType, "native", MPI_INFO_NULL);

  FEntryType *entries =
      (FEntryType *)allocate(ARRAY_ALIGNMENT, m->totalNnz * sizeof(FEntryType));

  for (int i = 0; i < m->nnz; i++) {
    entries[i].col = (unsigned int)m->entries[i].col;
    entries[i].val = (float)m->entries[i].val;
  }

  if (commIsMaster(c)) {
    MPI_File_write(fh, entries, m->totalNnz, entryType, MPI_STATUS_IGNORE);
  }

  free(entries);
  MPI_Type_free(&entryType);
  MPI_File_close(&fh);
}

void matrixBinRead(GMatrix *m, CommType *c, char *filename)
{
  MPI_File fh;
  MPI_Status status;
  MPI_Offset offset, disp;
  int count = 0;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  if (commIsMaster(c)) {
    printf("Reading matrix from %s\n", filename);
  }

  // read file header
  char header[HEADERSIZE];
  MPI_File_set_view(fh, 0, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
  MPI_File_read(fh, header, HEADERSIZE, MPI_CHAR, &status);
  MPI_Get_count(&status, MPI_CHAR, &count);
  if (count != HEADERSIZE) {
    printf("ERROR reading header!\n");
  }

  // read total matrix size
  MPI_File_get_position(fh, &offset);
  MPI_File_get_byte_offset(fh, offset, &disp);
  MPI_File_set_view(fh, disp, MPI_UNSIGNED, MPI_UNSIGNED, "native", MPI_INFO_NULL);
  unsigned int totalNr, totalNnz;
  MPI_File_read(fh, &totalNr, 1, MPI_UNSIGNED, &status);
  MPI_Get_count(&status, MPI_UNSIGNED, &count);
  if (count != 1) {
    printf("ERROR reading unsigned!\n");
  }
  MPI_File_read(fh, &totalNnz, 1, MPI_UNSIGNED, &status);
  MPI_Get_count(&status, MPI_UNSIGNED, &count);
  if (count != 1) {
    printf("ERROR reading unsigned!\n");
  }

  m->totalNr  = (CG_UINT)totalNr;
  m->totalNnz = (CG_UINT)totalNnz;
  printf("Rank %d: totalNr %u totalNnz %u\n", c->rank, m->totalNr, m->totalNnz);

  // partition matrix row wise
  int rank = c->rank;
  int size = c->size;
  int numRows, startRow, stopRow;

  int cursor = 0;
  for (int i = 0; i < rank + 1; i++) {
    numRows  = sizeOfRank(i, size, totalNr);
    startRow = cursor;
    cursor += numRows;
    stopRow = cursor - 1;
  }

  printf("Rank %d: numRows %d startRow %d stopRow %d\n",
      c->rank,
      numRows,
      startRow,
      stopRow);

  m->nr       = numRows;
  m->nc       = numRows;
  m->startRow = startRow;
  m->stopRow  = stopRow;

  // read row pointers
  m->rowPtr = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (numRows + 1) * sizeof(CG_UINT));

  MPI_File_get_position(fh, &offset);
  MPI_File_seek(fh, startRow, MPI_SEEK_CUR);
  MPI_File_read(fh, m->rowPtr, numRows + 1, MPI_UNSIGNED, &status);
  MPI_Get_count(&status, MPI_UNSIGNED, &count);
  if (count != numRows + 1) {
    printf("ERROR reading rowptr!\n");
  }

  // localize row pointer and determine local nnz
  int nnz = 0;
  for (int i = 0; i < numRows; i++) {
    nnz += m->rowPtr[i + 1] - m->rowPtr[i];
  }

  m->nnz = (CG_UINT)nnz;

  // determine entry offset
  int allnnz[size];
  MPI_Allgather(&nnz, 1, MPI_INT, allnnz, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Offset entryOffset = 0;

  for (int i = 0; i < rank; i++) {
    entryOffset += allnnz[i];
  }

  for (int i = 0; i <= numRows; i++) {
    m->rowPtr[i] -= entryOffset;
  }

  MPI_File_seek(fh, offset + m->totalNr + 1, MPI_SEEK_SET);
  MPI_File_get_position(fh, &offset);
  MPI_File_get_byte_offset(fh, offset, &disp);

  FEntryType *entries =
      (FEntryType *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(FEntryType));
  MPI_Datatype entryType;
  createEntrytype(&entryType);
  MPI_File_set_view(fh, disp, entryType, entryType, "native", MPI_INFO_NULL);
  MPI_File_seek(fh, entryOffset, MPI_SEEK_SET);
  MPI_File_read(fh, entries, m->nnz, entryType, &status);
  MPI_Get_count(&status, entryType, &count);
  if (count != m->nnz) {
    printf("ERROR reading rowptr!\n");
  }
  MPI_Type_free(&entryType);
  MPI_File_close(&fh);

  m->entries = (Entry *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(Entry));

  for (int i = 0; i < m->nnz; i++) {
    m->entries[i].col = (CG_UINT)entries[i].col;
    m->entries[i].val = (CG_FLOAT)entries[i].val;
  }

  free(entries);
}
#endif
