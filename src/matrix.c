/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"
#include "mmio.h"
#include "util.h"

static inline int compareColumn(const void *a, const void *b)
{
  const MMEntry *a_ = (const MMEntry *)a;
  const MMEntry *b_ = (const MMEntry *)b;

  return (a_->col > b_->col) - (a_->col < b_->col);
}

static inline int compareRow(const void *a, const void *b)
{
  const MMEntry *a_ = (const MMEntry *)a;
  const MMEntry *b_ = (const MMEntry *)b;

  return (a_->row > b_->row) - (a_->row < b_->row);
}

void matrixGenerate(GMatrix *m, Parameter *p, int rank, int size, bool use_7pt_stencil)
{

  CG_UINT local_nrow = p->nx * p->ny * p->nz;
  CG_UINT local_nnz  = 27 * local_nrow;

  CG_UINT total_nrow = local_nrow * size;
  CG_UINT total_nnz  = 27 * total_nrow;

  int start_row      = local_nrow * rank;
  int stop_row       = start_row + local_nrow - 1;

  if (!rank) {
    if (use_7pt_stencil) {
      printf("Generate 7pt matrix with ");
    } else {
      printf("Generate 27pt matrix with ");
    }
    printf("%.2e total rows and %.2e nonzeros\n", (double)total_nrow, (double)total_nnz);
  }

  m->entries = (Entry *)allocate(ARRAY_ALIGNMENT, local_nnz * sizeof(Entry));
  m->rowPtr  = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (local_nrow + 1) * sizeof(CG_UINT));

  CG_UINT *currowptr = m->rowPtr;
  CG_UINT nnzglobal  = 0;
  int nx = p->nx, ny = p->ny, nz = p->nz;
  CG_UINT cursor = 0;

  *currowptr++   = 0;

  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {

        int currow = start_row + iz * nx * ny + iy * nx + ix;
        int nnzrow = 0;

        for (int sz = -1; sz <= 1; sz++) {
          for (int sy = -1; sy <= 1; sy++) {
            for (int sx = -1; sx <= 1; sx++) {

              int curcol = currow + sz * nx * ny + sy * nx + sx;
              // Since we have a stack of nx by ny by nz domains
              //, stacking in the z direction, we check to see
              // if sx and sy are reaching outside of the domain,
              // while the check for the curcol being valid is
              // sufficient to check the z values
              if ((ix + sx >= 0) && (ix + sx < nx) && (iy + sy >= 0) && (iy + sy < ny) &&
                  (curcol >= 0 && curcol < total_nrow)) {
                // This logic will skip over point that are not part of a
                // 7-pt stencil
                if (!use_7pt_stencil || (sz * sz + sy * sy + sx * sx <= 1)) {
                  if (curcol == currow) {
                    m->entries[cursor].val = 27.0;
                  } else {
                    m->entries[cursor].val = -1.0;
                  }
                  m->entries[cursor].col = curcol;
                  cursor++;
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

void MMMatrixRead(MMMatrix *m, char *filename)
{
  MM_typecode matcode;
  FILE *f = NULL;
  int M, N, nz;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("Unable to open file.\n");
    exit(EXIT_FAILURE);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(EXIT_FAILURE);
  }

  if (!((mm_is_real(matcode) || mm_is_pattern(matcode) || mm_is_integer(matcode)) &&
          mm_is_matrix(matcode) && mm_is_sparse(matcode))) {
    fprintf(stderr, "Sorry, this application does not support ");
    fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(EXIT_FAILURE);
  }

  bool compatible_flag =
      (mm_is_sparse(matcode) &&
          (mm_is_real(matcode) || mm_is_pattern(matcode) || mm_is_integer(matcode))) &&
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

    if (!mm_is_symmetric(matcode)) {
      printf(" * matrix has to be symmetric\n");
    }

    exit(EXIT_FAILURE);
  }

  if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
    exit(EXIT_FAILURE);
  }

  printf("Read matrix %s with %d non zeroes and %d rows\n", filename, nz, M);

  if (sym_flag) {
    m->entries = (MMEntry *)allocate(ARRAY_ALIGNMENT, nz * 2 * sizeof(MMEntry));
  } else {
    m->entries = (MMEntry *)allocate(ARRAY_ALIGNMENT, nz * sizeof(MMEntry));
  }

  size_t cursor = 0;
  int row, col;
  double v;
  MMEntry *entries = m->entries;

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
  m->nnz   = cursor;
  m->count = cursor;

  // sort by column
  qsort(m->entries, m->count, sizeof(MMEntry), compareColumn);
// sort by row requires a stable sort. As glibc qsort is mergesort this
// hopefully works.
#ifdef __linux__
  qsort(m->entries, m->count, sizeof(MMEntry), compareRow);
#else
  // BSD has a dedicated mergesort available in its libc
  mergesort(m->entries, m->count, sizeof(MMEntry), compareRow);
#endif
}

void matrixConvertfromMM(MMMatrix *mm, GMatrix *m)
{
  m->startRow     = mm->startRow;
  m->stopRow      = mm->stopRow;
  m->totalNr      = mm->totalNr;
  m->totalNnz     = mm->totalNnz;
  m->nr           = mm->nr;
  m->nc           = mm->nr;
  m->nnz          = mm->nnz;
  m->entries      = (Entry *)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(Entry));
  m->rowPtr       = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (m->nr + 1) * sizeof(CG_UINT));

  int *valsPerRow = (int *)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(int));

  for (int i = 0; i < m->nr; i++) {
    valsPerRow[i] = 0;
  }

  MMEntry *entries = mm->entries;
  int startRow     = mm->startRow;

  for (int i = 0; i < mm->count; i++) {
    valsPerRow[entries[i].row - startRow]++;
  }

  m->rowPtr[0] = 0;

  // convert to CCRS format
  for (int rowID = 0; rowID < m->nr; rowID++) {
    m->rowPtr[rowID + 1] = m->rowPtr[rowID] + valsPerRow[rowID];

    // loop over all elements in Row
    for (int id = m->rowPtr[rowID]; id < m->rowPtr[rowID + 1]; id++) {
      m->entries[id].val = (CG_FLOAT)entries[id].val;
      m->entries[id].col = (CG_UINT)entries[id].col;
    }
  }
}
