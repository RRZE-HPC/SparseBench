/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#include "matrixBinfile.h"
#include "mpi.h"

#include "allocate.h"
#include "util.h"
#include <stdio.h>

#define HEADERSIZE 24

static int sizeOfRank(int rank, int size, int N) {
  return N / size + ((N % size > rank) ? 1 : 0);
}

static void createEntrytype(MPI_Datatype *entryType) {
  MPI_Aint displ[2];
  FEntry dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.col, &displ[0]);
  MPI_Get_address(&dummy.val, &displ[1]);
  displ[0] = MPI_Aint_diff(displ[0], base_address);
  displ[1] = MPI_Aint_diff(displ[1], base_address);

  int lengths[2] = {1, 1};
  MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_FLOAT};
  MPI_Type_create_struct(2, lengths, displ, types, entryType);
  MPI_Type_commit(entryType);
}

void matrixBinWrite(GMatrix *m, Comm *c, char *filename) {
  MPI_File fh;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);

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
  MPI_File_set_view(fh, disp, MPI_UNSIGNED, MPI_UNSIGNED, "native",
                    MPI_INFO_NULL);
  if (commIsMaster(c)) {
    printf("Writing matrix (nr=%u, nnz=%u) at offset %lld to %s\n", m->totalNr,
           m->totalNnz, disp, filename);
    // FIXME: convert to unsigned int in case CG_UINT is different
    MPI_File_write(fh, &m->totalNr, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
    MPI_File_write(fh, &m->totalNnz, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
    MPI_File_write(fh, m->rowPtr, m->totalNr + 1, MPI_UNSIGNED,
                   MPI_STATUS_IGNORE);
  }
  MPI_Datatype entryType;
  createEntrytype(&entryType);
  MPI_File_sync(fh);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_get_size(fh, &disp);
  MPI_File_set_view(fh, disp, entryType, entryType, "native", MPI_INFO_NULL);
  if (commIsMaster(c)) {
    MPI_File_write(fh, m->entries, m->totalNnz, entryType, MPI_STATUS_IGNORE);
  }

  MPI_Type_free(&entryType);
  MPI_File_close(&fh);
}

void matrixBinRead(GMatrix *m, Comm *c, char *filename) {
  MPI_File fh;
  MPI_Status status;
  MPI_Offset offset, disp;
  int count = 0;

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  if (commIsMaster(c)) {
    printf("Reading matrix from %s\n", filename);
  }

  char header[HEADERSIZE];
  MPI_File_set_view(fh, 0, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);

  if (commIsMaster(c)) {
    MPI_File_read(fh, header, HEADERSIZE, MPI_CHAR, &status);
    MPI_Get_count(&status, MPI_CHAR, &count);
    printf("Read %d elements from file\n", count);
    printf("File header %s\n", header);
  }
  MPI_File_get_position(fh, &offset);
  MPI_File_get_byte_offset(fh, offset, &disp);
  MPI_File_set_view(fh, disp, MPI_UNSIGNED, MPI_UNSIGNED, "native",
                    MPI_INFO_NULL);
  unsigned int totalNr, totalNnz;
  MPI_File_read(fh, &totalNr, 1, MPI_UNSIGNED, &status);
  MPI_Get_count(&status, MPI_UNSIGNED, &count);
  printf("Read %d elements from file at offset %lld\n", count, disp);
  MPI_File_read(fh, &totalNnz, 1, MPI_UNSIGNED, &status);
  MPI_Get_count(&status, MPI_UNSIGNED, &count);
  printf("Read %d elements from file\n", count);

  m->totalNr = (CG_UINT)totalNr;
  m->totalNnz = (CG_UINT)totalNnz;
  printf("Matrix: %u total non zeroes, total number of rows %u\n", m->totalNnz,
         m->totalNr);

  int rank = c->rank;
  int size = c->size;
  int numRows, startRow, stopRow;

  int cursor = 0;
  for (int i = 0; i < rank + 1; i++) {
    numRows = sizeOfRank(i, size, totalNr);
    startRow = cursor;
    cursor += numRows;
    stopRow = cursor - 1;
  }

  m->nr = numRows;
  m->startRow = startRow;
  m->stopRow = stopRow;
  m->rowPtr =
      (CG_UINT *)allocate(ARRAY_ALIGNMENT, (numRows + 1) * sizeof(CG_UINT));

  MPI_File_get_position(fh, &offset);
  MPI_File_get_byte_offset(fh, offset, &disp);
  MPI_File_set_view(fh, disp + (startRow * sizeof(unsigned int)), MPI_UNSIGNED,
                    MPI_UNSIGNED, "native", MPI_INFO_NULL);
  MPI_File_read(fh, m->rowPtr, numRows + 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
  int nnz = 0;
  for (int i = 0; i < numRows; i++) {
    nnz += m->rowPtr[i + 1] - m->rowPtr[i];
  }

  m->nnz = nnz;
  m->entries = (Entry *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(Entry));

  int allnnz[size];
  MPI_Allgather(&nnz, 1, MPI_INT, allnnz, 1, MPI_INT, MPI_COMM_WORLD);
  int entryOffset = 0;

  for (int i = 0; i < rank; i++) {
    entryOffset += allnnz[i];
  }

  MPI_Datatype entryType;
  createEntrytype(&entryType);
  disp += (totalNr * sizeof(unsigned int));
  MPI_Aint extent;
  MPI_File_get_type_extent(fh, entryType, &extent);
  MPI_File_set_view(fh, disp + entryOffset * extent, entryType, entryType,
                    "native", MPI_INFO_NULL);

  MPI_File_read(fh, m->entries, nnz, entryType, MPI_STATUS_IGNORE);
  MPI_Type_free(&entryType);
  MPI_File_close(&fh);
}
