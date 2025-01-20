/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"
#include "mmio.h"
#include "util.h"

static inline int compareColumn(const void* a, const void* b)
{
  const Entry* a_ = (const Entry*)a;
  const Entry* b_ = (const Entry*)b;

  return (a_->col > b_->col) - (a_->col < b_->col);
}

static inline int compareRow(const void* a, const void* b)
{
  const Entry* a_ = (const Entry*)a;
  const Entry* b_ = (const Entry*)b;

  return (a_->row > b_->row) - (a_->row < b_->row);
}

static void dumpMMMatrix(Entry* mm, int nz)
{
  for (int i = 0; i < nz; i++) {
    printf("%d %d: %f\n", mm[i].row, mm[i].col, mm[i].val);
  }
}

void matrixRead(Matrix* m, char* filename)
{
  MM_typecode matcode;
  FILE* f;
  int M, N, nz;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("Unable to open file");
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(EXIT_FAILURE);
  }

  if (!((mm_is_real(matcode) || mm_is_pattern(matcode) ||
            mm_is_integer(matcode)) &&
          mm_is_matrix(matcode) && mm_is_sparse(matcode))) {
    fprintf(stderr, "Sorry, this application does not support ");
    fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(EXIT_FAILURE);
  }

  bool compatible_flag = (mm_is_sparse(matcode) &&
                             (mm_is_real(matcode) || mm_is_pattern(matcode) ||
                                 mm_is_integer(matcode))) &&
                         (mm_is_symmetric(matcode) || mm_is_general(matcode));
  bool sym_flag     = mm_is_symmetric(matcode);
  bool pattern_flag = mm_is_pattern(matcode);
  bool complex_flag = mm_is_complex(matcode);

  if (!compatible_flag) {
    printf("The matrix market file provided is not supported.\n Reason :\n");
    if (!mm_is_sparse(matcode)) {
      printf(" * matrix has to be sparse\n");
    }

    if (!mm_is_real(matcode) && !(mm_is_pattern(matcode))) {
      printf(" * matrix has to be real or pattern\n");
    }

    if (!mm_is_symmetric(matcode) && !mm_is_general(matcode)) {
      printf(" * matrix has to be either general or symmetric\n");
    }

    exit(EXIT_FAILURE);
  }

  if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
    exit(EXIT_FAILURE);
  }

  printf("Read matrix %s with %d non zeroes and %d rows\n", filename, nz, M);

  m->nr  = M;
  m->nnz = nz;
  Entry* mm;

  if (sym_flag) {
    mm = (Entry*)allocate(ARRAY_ALIGNMENT, nz * 2 * sizeof(Entry));
  } else {
    mm = (Entry*)allocate(ARRAY_ALIGNMENT, nz * sizeof(Entry));
  }

  size_t cursor = 0;
  int row, col;
  double v;

  for (size_t i = 0; i < nz; i++) {

    if (pattern_flag) {
      fscanf(f, "%d %d\n", &row, &col);
      v = 1.;
    } else if (complex_flag) {
      fscanf(f, "%d %d %lg %*g\n", &row, &col, &v);
    } else {
      fscanf(f, "%d %d %lg\n", &row, &col, &v);
    }

    row--; /* adjust from 1-based to 0-based */
    col--;

    mm[cursor].row   = row;
    mm[cursor].col   = col;
    mm[cursor++].val = v;

    if (sym_flag && (row != col)) {
      mm[cursor].row   = col;
      mm[cursor].col   = row;
      mm[cursor++].val = v;
    }
  }

  fclose(f);
  size_t mms = cursor;

  // sort by column
  qsort(mm, mms, sizeof(Entry), compareColumn);
  // dumpMMMatrix(mm, nz);
  // sort by row
  mergesort(mm, mms, sizeof(Entry), compareRow);
  // dumpMMMatrix(mm, nz);

  m->rowPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      (m->nr + 1) * sizeof(CG_UINT));
  m->colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  m->val    = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_FLOAT));

  int* valsPerRow = (int*)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(int));

  for (int i = 0; i < m->nr; i++) {
    valsPerRow[i] = 0;
  }

  for (int i = 0; i < mms; i++) {
    valsPerRow[mm[i].row]++;
  }

  m->rowPtr[0] = 0;

  // convert to CRS format
  for (int rowID = 0; rowID < m->nr; rowID++) {

    m->rowPtr[rowID + 1] = m->rowPtr[rowID] + valsPerRow[rowID];

    // loop over all elements in Row
    for (int id = m->rowPtr[rowID]; id < m->rowPtr[rowID + 1]; id++) {
      m->val[id]    = (CG_FLOAT)mm[id].val;
      m->colInd[id] = (CG_UINT)mm[id].col;
    }
  }
}
