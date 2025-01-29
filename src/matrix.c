/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"
#include "mmio.h"

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

static void dumpMMMatrix(MmMatrix* mm)
{
  Entry* entries = mm->entries;

  for (int i = 0; i < mm->count; i++) {
    printf("%d %d: %f\n", entries[i].row, entries[i].col, entries[i].val);
  }
}

void matrixGenerate(
    Matrix* m, Parameter* p, int rank, int size, bool use_7pt_stencil)
{

  CG_UINT local_nrow = p->nx * p->ny * p->nz;
  CG_UINT local_nnz  = 27 * local_nrow;

  CG_UINT total_nrow = local_nrow * size;
  CG_UINT total_nnz  = 27 * total_nrow;

  int start_row = local_nrow * rank;
  int stop_row  = start_row + local_nrow - 1;

  if (!rank) {
    if (use_7pt_stencil) {
      printf("Generate 7pt matrix with ");
    } else {
      printf("Generate 27pt matrix with ");
    }
    printf("%d total rows and %d nonzeros\n", (int)total_nrow, (int)local_nnz);
  }

  m->val = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, local_nnz * sizeof(CG_FLOAT));
  m->colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT, local_nnz * sizeof(CG_UINT));
  m->rowPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      (local_nrow + 1) * sizeof(CG_UINT));

  CG_FLOAT* curvalptr = m->val;
  CG_UINT* curindptr  = m->colInd;
  CG_UINT* currowptr  = m->rowPtr;

  CG_UINT nnzglobal = 0;
  int nx = p->nx, ny = p->ny, nz = p->nz;
  CG_UINT cursor = 0;

  *currowptr++ = 0;

  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {

        int curlocalrow = iz * nx * ny + iy * nx + ix;
        int currow      = start_row + iz * nx * ny + iy * nx + ix;
        int nnzrow      = 0;

        for (int sz = -1; sz <= 1; sz++) {
          for (int sy = -1; sy <= 1; sy++) {
            for (int sx = -1; sx <= 1; sx++) {

              int curcol = currow + sz * nx * ny + sy * nx + sx;
              // Since we have a stack of nx by ny by nz domains
              //, stacking in the z direction, we check to see
              // if sx and sy are reaching outside of the domain,
              // while the check for the curcol being valid is
              // sufficient to check the z values
              if ((ix + sx >= 0) && (ix + sx < nx) && (iy + sy >= 0) &&
                  (iy + sy < ny) && (curcol >= 0 && curcol < total_nrow)) {
                // This logic will skip over point that are not part of a
                // 7-pt stencil
                if (!use_7pt_stencil || (sz * sz + sy * sy + sx * sx <= 1)) {
                  if (curcol == currow) {
                    *curvalptr++ = 27.0;
                  } else {
                    *curvalptr++ = -1.0;
                  }
                  *curindptr++ = curcol;
                  nnzrow++;
                }
              }
            } // end sx loop
          } // end sy loop
        } // end sz loop

        *currowptr = *(currowptr - 1) + nnzrow;
        currowptr++;
        nnzglobal += nnzrow;
      } // end ix loop
    } // end iy loop
  } // end iz loop

#ifdef VERBOSE
  printf("Process %d of %d has %d rows\n", rank, size, local_nrow);
  printf("Global rows %d through %d\n", start_row, stop_row);
  printf("%d nonzeros\n", local_nnz);
#endif

  m->startRow = start_row;
  m->stopRow  = stop_row;
  m->totalNr  = total_nrow;
  m->totalNnz = total_nnz;
  m->nr       = local_nrow;
  m->nc       = local_nrow;
  m->nnz      = local_nnz;
}

void matrixRead(MmMatrix* m, char* filename)
{
  MM_typecode matcode;
  FILE* f = NULL;
  int M, N, nz;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("Unable to open file");
    exit(EXIT_FAILURE);
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

  if (sym_flag) {
    m->entries = (Entry*)allocate(ARRAY_ALIGNMENT, nz * 2 * sizeof(Entry));
  } else {
    m->entries = (Entry*)allocate(ARRAY_ALIGNMENT, nz * sizeof(Entry));
  }

  size_t cursor = 0;
  int row, col;
  double v;
  Entry* entries = m->entries;

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

    entries[cursor].row   = row;
    entries[cursor].col   = col;
    entries[cursor++].val = v;

    if (sym_flag && (row != col)) {
      entries[cursor].row   = col;
      entries[cursor].col   = row;
      entries[cursor++].val = v;
    }
  }

  fclose(f);
  m->nr    = M;
  m->nnz   = nz;
  m->count = cursor;

  // sort by column
  qsort(m->entries, m->count, sizeof(Entry), compareColumn);
// dumpMMMatrix(m);
// sort by row requires a stable sort. As glibc qsort is mergesort this
// hopefully works.
#ifdef __linux__
  qsort(m->entries, m->count, sizeof(Entry), compareColumn);
#else
  // BSD has a dedicated mergesort available in its libc
  mergesort(m->entries, m->count, sizeof(Entry), compareRow);
#endif
  // dumpMMMatrix(m);
}

void matrixConvertMMtoCRS(MmMatrix* mm, Matrix* m, int rank, int size)
{
  m->nr  = mm->nr;
  m->nnz = mm->nnz;

  m->rowPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      (m->nr + 1) * sizeof(CG_UINT));
  m->colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  m->val    = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_FLOAT));

  int* valsPerRow = (int*)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(int));

  for (int i = 0; i < m->nr; i++) {
    valsPerRow[i] = 0;
  }

  Entry* entries = mm->entries;

  for (int i = 0; i < mm->count; i++) {
    valsPerRow[entries[i].row]++;
  }

  m->rowPtr[0] = 0;

  // convert to CRS format
  for (int rowID = 0; rowID < m->nr; rowID++) {

    m->rowPtr[rowID + 1] = m->rowPtr[rowID] + valsPerRow[rowID];

    // loop over all elements in Row
    for (int id = m->rowPtr[rowID]; id < m->rowPtr[rowID + 1]; id++) {
      m->val[id]    = (CG_FLOAT)entries[id].val;
      m->colInd[id] = (CG_UINT)entries[id].col;
    }
  }
}
