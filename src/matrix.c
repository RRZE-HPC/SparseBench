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
#include "vtype.h"

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

  if (!(mm_is_matrix(matcode) && mm_is_sparse(matcode) &&
          (mm_is_real(matcode) || mm_is_pattern(matcode) || mm_is_integer(matcode) ||
              mm_is_complex(matcode)))) {
    fprintf(stderr, "Sorry, this application does not support ");
    fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(EXIT_FAILURE);
  }

  bool compatible_flag =
      mm_is_sparse(matcode) &&
      (mm_is_real(matcode) || mm_is_pattern(matcode) || mm_is_integer(matcode) ||
          mm_is_complex(matcode)) &&
      (mm_is_symmetric(matcode) || mm_is_general(matcode) || mm_is_hermitian(matcode));
  bool sym_flag       = mm_is_symmetric(matcode) || mm_is_hermitian(matcode);
  bool hermitian_flag = mm_is_hermitian(matcode);
  bool pattern_flag   = mm_is_pattern(matcode);
  bool complex_flag   = mm_is_complex(matcode);

  if (!compatible_flag) {
    printf("The matrix market file provided is not supported.\n Reason :\n");
    if (!mm_is_sparse(matcode)) {
      printf(" * matrix has to be sparse\n");
    }

    if (!mm_is_real(matcode) && !mm_is_pattern(matcode) && !mm_is_complex(matcode)) {
      printf(" * matrix has to be real, complex, or pattern\n");
    }

    if (!mm_is_symmetric(matcode) && !mm_is_general(matcode) &&
        !mm_is_hermitian(matcode)) {
      printf(" * matrix has to be symmetric, hermitian, or general\n");
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
  double v, v_imag;
  MMEntry *entries = m->entries;

  for (size_t i = 0; i < nz; i++) {
    v_imag = 0.0;

    if (pattern_flag) {
      fscanf(f, "%d %d\n", &row, &col);
      v = 1.;
    } else if (complex_flag) {
      fscanf(f, "%d %d %lg %lg\n", &row, &col, &v, &v_imag);
    } else {
      fscanf(f, "%d %d %lg\n", &row, &col, &v);
    }

    row--; /* adjust from 1-based to 0-based */
    col--;

    entries[cursor].row        = row;
    entries[cursor].col        = col;
    entries[cursor].val        = v;
    entries[cursor++].val_imag = v_imag;

    if (sym_flag && (row != col)) {
      entries[cursor].row        = col;
      entries[cursor].col        = row;
      entries[cursor].val        = v;
      entries[cursor++].val_imag = hermitian_flag ? -v_imag : v_imag;
    }
  }

  fclose(f);
  m->nr       = M;
  m->nnz      = cursor;
  m->count    = cursor;
  m->totalNr  = M;
  m->totalNnz = cursor;
  m->startRow = 0;
  m->stopRow  = M;

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
#ifdef USE_COMPLEX
      m->entries[id].val = VCONST(entries[id].val, entries[id].val_imag);
#else
      m->entries[id].val = (V_ELE)entries[id].val;
#endif
      m->entries[id].col = (CG_UINT)entries[id].col;
    }
  }
}

void MMMatrixPrintToFile(MMMatrix *m, char *filename)
{
  FILE *fptr;
  fptr = fopen(filename, "w");
  MMMatrixPrint_impl(m, fptr);
  fclose(fptr);
}

void MMMatrixPrint(MMMatrix *m)
{
  MMMatrixPrint_impl(m, stdout);
}

void MMMatrixPrint_impl(MMMatrix *m, FILE *fptr)
{
  fprintf(fptr, "Matrix dimensions: %d x %d\n", m->nr, m->nr);
  fprintf(fptr, "Number of non-zeros: %d\n", m->nnz);
  fprintf(fptr, "\nMatrix entries (row, col, value):\n");

  for (size_t i = 0; i < m->count; i++) {
    if (m->entries[i].val_imag != 0.0) {
      fprintf(fptr,
          "%d\t%d\t%g\t%g\n",
          m->entries[i].row,
          m->entries[i].col,
          m->entries[i].val,
          m->entries[i].val_imag);
    } else {
      fprintf(
          fptr, "%d\t%d\t%g\n", m->entries[i].row, m->entries[i].col, m->entries[i].val);
    }
  }
}

void GMatrixPrintTofile(GMatrix *m, char *filename)
{
  FILE *fptr;
  fptr = fopen(filename, "w");
  GMatrixPrint_impl(m, fptr);
  fclose(fptr);
}

void GMatrixPrint(GMatrix *m)
{
  GMatrixPrint_impl(m, stdout);
}

void GMatrixPrint_impl(GMatrix *m, FILE *fptr)
{
  fprintf(fptr, "Matrix dimensions: %u x %u\n", m->nr, m->nc);
  fprintf(fptr, "Number of non-zeros: %u\n", m->nnz);
  fprintf(fptr, "\nMatrix entries (row, col, value):\n");
  for (CG_UINT row = 0; row < m->nr; row++) {
    for (CG_UINT idx = m->rowPtr[row]; idx < m->rowPtr[row + 1]; idx++) {
#ifdef USE_COMPLEX
      fprintf(fptr,
          "%u\t%u\t%g+%gi\n",
          row + m->startRow,
          m->entries[idx].col,
          VREAL(m->entries[idx].val),
          VIMAG(m->entries[idx].val));
#else
      fprintf(fptr,
          "%u\t%u\t%g\n",
          row + m->startRow,
          m->entries[idx].col,
          m->entries[idx].val);
#endif
    }
  }
}

void dumpVectorPrint(V_ELE *restrict y, CG_UINT numRows)
{
  dumpVectorToFile(y, numRows, stdout);
  printf("\n");
}

void dumpVectorToFile(V_ELE *restrict y, CG_UINT numRows, FILE *reportedData)
{
  fprintf(reportedData, "vec = ");
  for (CG_UINT i = 0; i < numRows; i++) {
#ifdef USE_COMPLEX
    fprintf(reportedData, "(%lf+%lfi), ", VREAL(y[i]), VIMAG(y[i]));
#else
    fprintf(reportedData, "%lf, ", y[i]);
#endif
  }
}

extern void dumpDMatrixToFile(DMatrix *m, char *filename)
{
  FILE *fptr = fopen(filename, "w");
  dumpDMatrix_impl(m, fptr);
  fclose(fptr);
}
extern void dumpDMatrix(DMatrix *m)
{
  dumpDMatrix_impl(m, stdout);
}
extern void dumpDMatrix_impl(DMatrix *m, FILE *reportedData)
{
  fprintf(reportedData, "row order matrix = ");
  for (CG_UINT i = 0; i < m->nr * m->nc; i++) {
#ifdef USE_COMPLEX
    fprintf(reportedData, "(%lf+%lfi), ", VREAL(m->entries[i]), VIMAG(m->entries[i]));
#else
    fprintf(reportedData, "%lf, ", m->entries[i]);
#endif
  }
}

void permute_DMatrix(const CG_UINT *perm, const DMatrix *src, DMatrix *dst)
{
  CG_UINT nc = src->nc;
  CG_UINT nr = MIN(src->nr, dst->nr);
  for (CG_UINT i = 0; i < nr; i++) {
    CG_UINT newRow         = perm[i];
    V_ELE *dst_start       = &dst->entries[newRow * nc];
    const V_ELE *src_start = &src->entries[i * nc];
    for (CG_UINT j = 0; j < nc; j++) {
      dst_start[j] = src_start[j];
    }
  }
}

void permute_vector(
    const CG_UINT *permute, const V_ELE *vec_src, V_ELE *vec_dst, CG_UINT nr)
{
  for (CG_UINT i = 0; i < nr; i++) {
    CG_UINT alt  = permute[i];
    vec_dst[alt] = vec_src[i];
  }
}
