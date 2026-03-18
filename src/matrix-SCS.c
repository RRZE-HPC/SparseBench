/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "matrix.h"

static inline int compareDesc(const void *a, const void *b)
{
  const int val_a = *(const int *)a;
  const int val_b = *(const int *)b;

  return val_b - val_a;
}

static inline int compareDescSCS(const void *a, const void *b)
{

  const SellCSigmaPair *pa = (const SellCSigmaPair *)a;
  const SellCSigmaPair *pb = (const SellCSigmaPair *)b;

  if (pa->count < pb->count)
    return 1; // Descending order
  if (pa->count > pb->count)
    return -1;
  return 0; // Stable if equal
}

void convertMatrix(Matrix *m, GMatrix *im)
{
  // m->C        = (CG_UINT)SELL_CHUNK; // set this before to maintain API
  // m->sigma    = (CG_UINT)SELL_SIGMA; // set this before to maintain API
  m->startRow = im->startRow;
  m->stopRow  = im->stopRow;
  m->totalNr  = im->totalNr;
  m->totalNnz = im->totalNnz;
  m->nr       = im->nr;
  m->nc       = im->nc;
  m->nnz      = im->nnz;
  m->nChunks  = (m->nr + m->C - 1) / m->C;
  m->nrPadded = m->nChunks * m->C;

  // printf("info : ")

  // (Temporary array) Assign an index to each row to use for row sorting
  SellCSigmaPair *elemsPerRow =
      (SellCSigmaPair *)allocate(ARRAY_ALIGNMENT, m->nrPadded * sizeof(SellCSigmaPair));

  for (int i = 0; i < m->nrPadded; ++i) {
    elemsPerRow[i].index = i;
    elemsPerRow[i].count = 0;
  }

  // Collect the number of elements in each row
  CG_UINT *rowPtr = im->rowPtr;
  for (int i = 0; i < m->nr; i++) {
    elemsPerRow[i].count = rowPtr[i + 1] - rowPtr[i];
  }

  // Sort rows over a scope of sigma
  for (int i = 0; i < m->nrPadded; i += m->sigma) {
    int chunkStart = i;
    int chunkStop  = MIN(i + m->sigma, m->nrPadded);
    int size       = chunkStop - chunkStart;

    // Sorting rows by element count using struct keeps index/count together
#ifdef __linux__
    qsort(&elemsPerRow[chunkStart], size, sizeof(SellCSigmaPair), compareDescSCS);
#else
    // BSD has a dedicated mergesort available in its libc
    mergesort(&elemsPerRow[chunkStart], size, sizeof(SellCSigmaPair), compareDescSCS);
#endif
  }

  m->chunkLens = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nChunks * sizeof(CG_UINT));
  m->chunkPtr  = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (m->nChunks + 1) * sizeof(CG_UINT));

  CG_UINT currentChunkPtr = 0;

  for (int i = 0; i < m->nChunks; ++i) {
    // Note sure about this yet
    // int chunkStart = elemsPerRow[i * m->C].count;
    // int chunkStop = ((i * m->C + m->C) < m->nrPadded)
    //               ? elemsPerRow[i * m->C + m->C].count
    //               : elemsPerRow[m->nrPadded - 1].count;
    // SellCSigmaPair chunkStart = elemsPerRow[i * m->C];
    // SellCSigmaPair chunkStop  = elemsPerRow[MIN((i+1) * m->C, m->nrPadded - 1)];

    // int size                  = chunkStop.index - chunkStart.index;

    // Collect longest row in chunk as chunk length
    CG_UINT maxLength = 0;
    for (int j = 0; j < m->C; ++j) {
      CG_UINT rowLenth = elemsPerRow[i * m->C + j].count;
      if (rowLenth > maxLength)
        maxLength = rowLenth;
    }

    // Collect chunk data to arrays
    m->chunkLens[i] = (CG_UINT)maxLength;
    m->chunkPtr[i]  = (CG_UINT)currentChunkPtr;
    currentChunkPtr += m->chunkLens[i] * m->C;
  }

  // Account for final chunk
  m->nElems               = currentChunkPtr;
  m->beta                 = (double)m->nnz / (double)m->nElems;
  m->chunkPtr[m->nChunks] = (CG_UINT)m->nElems;

  // Construct permutation vector
  m->oldToNewPerm = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(CG_UINT));
  for (int i = 0; i < m->nrPadded; ++i) {
    CG_UINT oldRow = elemsPerRow[i].index;
    if (oldRow < m->nr)
      m->oldToNewPerm[oldRow] = (CG_UINT)i;
  }

  // Construct inverse permutation vector
  m->newToOldPerm = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(CG_UINT));
  for (int i = 0; i < m->nr; ++i) {
#ifdef VERBOSE
    // Sanity check for common error
    if (m->oldToNewPerm[i] >= m->nr) {
      fprintf(stderr,
          "ERROR matrixConvertMMtoSCS: m->oldToNewPerm[%d]=%d"
          " is out of bounds (>%d).\n",
          i,
          m->oldToNewPerm[i],
          m->nr);
    }
#endif
    m->newToOldPerm[m->oldToNewPerm[i]] = (CG_UINT)i;
  }

  // Now that chunk data is collected, fill with matrix data
  m->colInd = (CG_UINT *)allocate(ARRAY_ALIGNMENT, m->nElems * sizeof(CG_UINT));
  m->val    = (V_ELE *)allocate(ARRAY_ALIGNMENT, m->nElems * sizeof(V_ELE));

// Initialize defaults (essential for padded elements)
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int c = 0; c < m->nChunks; ++c) {
    CG_UINT start = m->chunkPtr[c];
    CG_UINT end   = m->chunkPtr[c + 1];
    for (CG_UINT j = start; j < end; ++j) {
      m->val[j]    = 0.0;
      m->colInd[j] = (CG_UINT)0;
    }
  }

  // (Temporary array) Keep track of how many elements we've seen in each row
  int *rowLocalElemCount = (int *)allocate(ARRAY_ALIGNMENT, m->nrPadded * sizeof(int));
  for (int i = 0; i < m->nrPadded; ++i) {
    rowLocalElemCount[i] = 0;
  }

  for (int i = 0; i < m->nr; i++) {

    int rowOld = i;

    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      Entry e        = im->entries[j];

      int row        = m->oldToNewPerm[rowOld];
      int chunkIdx   = row / m->C;
      int chunkStart = m->chunkPtr[chunkIdx];
      int chunkRow   = row % m->C;
      int idx        = chunkStart + rowLocalElemCount[row] * m->C + chunkRow;

      // Only permute local column indices; external columns (from MPI
      // localization) live beyond nr and must not be permuted.
      if (e.col < m->nr) {
        m->colInd[idx] = (CG_UINT)m->oldToNewPerm[e.col];
      } else {
        m->colInd[idx] = (CG_UINT)e.col;
      }
#ifdef VERBOSE
      // Sanity check for common error
      if (m->colInd[idx] >= m->nc) {
        fprintf(stderr,
            "ERROR matrixConvertMMtoSCS: m->colInd[%d]=%d"
            " is out of bounds (>%d).\n",
            idx,
            m->colInd[idx],
            m->nc);
      }
#endif
      m->val[idx] = e.val;
      ++rowLocalElemCount[row];
    }
  }

  deallocate(elemsPerRow);
  deallocate(rowLocalElemCount);
}

void MatrixPrintTofile(Matrix *m, char *filename)
{
  FILE *fptr;
  fptr = fopen(filename, "w");
  MatrixPrint_impl(m, fptr);
  fclose(fptr);
}

void MatrixPrint(Matrix *m)
{
  MatrixPrint_impl(m, stdout);
}

void MatrixPrint_impl(Matrix *m, FILE *fptr)
{
  // Print SCS matrix in strict row order (original row order, skipping padded rows)
  if (!m || !fptr)
    return;

  CG_UINT nr            = m->nr;
  CG_UINT nc            = m->nc;
  CG_UINT C             = m->C;
  CG_UINT nChunks       = m->nChunks;
  CG_UINT *chunkPtr     = m->chunkPtr;
  CG_UINT *chunkLens    = m->chunkLens;
  CG_UINT *colInd       = m->colInd;
  V_ELE *val            = m->val;
  CG_UINT *oldToNewPerm = m->oldToNewPerm;
  CG_UINT *newToOldPerm = m->newToOldPerm;

  // For each original row (strict order)
  for (CG_UINT row = 0; row < nr; ++row) {
    CG_UINT newRow     = oldToNewPerm[row];
    CG_UINT chunkIdx   = newRow / C;
    CG_UINT chunkRow   = newRow % C;
    CG_UINT chunkStart = chunkPtr[chunkIdx];
    CG_UINT rowLen     = chunkLens[chunkIdx];

    fprintf(fptr, "row %u:", row);
    for (CG_UINT j = 0; j < rowLen; ++j) {
      CG_UINT idx = chunkStart + j * C + chunkRow;
      CG_UINT col = colInd[idx];
      V_ELE v     = val[idx];
#ifdef USE_COMPLEX
      if (VREAL(v) != 0.0 || VIMAG(v) != 0.0) {
        fprintf(fptr, " (%u, %.12g+%.12gi)", col, VREAL(v), VIMAG(v));
      }
#else
      if (v != 0.0) {
        fprintf(fptr, " (%u, %.12g)", col, v);
      }
#endif
    }
    fprintf(fptr, "\n");
  }
}

void dumpMatrixToFile(Matrix *m, char *filename)
{
  FILE *fptr;
  fptr = fopen(filename, "w");
  dumpMatrix_impl(m, fptr);
  fclose(fptr);
}

void dumpMatrix(Matrix *m)
{
  dumpMatrix_impl(m, stdout);
}

#define PRINT_FIELD(fp, obj, field)                                                      \
  fprintf((fp), #obj "->" #field " = %lld\n", (long long)((obj)->field));

#define PRINT_INT_ARRAY(fp, obj, field, n)                                               \
  do {                                                                                   \
    fprintf((fp), #field ": ");                                                          \
    for (size_t i = 0; i < (n); ++i) {                                                   \
      fprintf((fp), "%d, ", (obj)->field[i]);                                            \
    }                                                                                    \
    fprintf((fp), "\n");                                                                 \
  } while (0)

#ifdef USE_COMPLEX
#define PRINT_V_ELE_ARRAY(fp, obj, field, n)                                             \
  do {                                                                                   \
    fprintf((fp), #field ": ");                                                          \
    for (size_t i = 0; i < (n); ++i) {                                                   \
      fprintf((fp), "(%f+%fi), ", VREAL((obj)->field[i]), VIMAG((obj)->field[i]));       \
    }                                                                                    \
    fprintf((fp), "\n");                                                                 \
  } while (0)
#else
#define PRINT_V_ELE_ARRAY(fp, obj, field, n)                                             \
  do {                                                                                   \
    fprintf((fp), #field ": ");                                                          \
    for (size_t i = 0; i < (n); ++i) {                                                   \
      fprintf((fp), "%f, ", (obj)->field[i]);                                            \
    }                                                                                    \
    fprintf((fp), "\n");                                                                 \
  } while (0)
#endif

void dumpMatrix_impl(Matrix *m, FILE *fptr)
{
  PRINT_FIELD(fptr, m, startRow);
  PRINT_FIELD(fptr, m, stopRow);
  PRINT_FIELD(fptr, m, totalNr);
  PRINT_FIELD(fptr, m, totalNnz);
  PRINT_FIELD(fptr, m, nr);
  PRINT_FIELD(fptr, m, nc);
  PRINT_FIELD(fptr, m, nnz);
  PRINT_FIELD(fptr, m, C);
  PRINT_FIELD(fptr, m, sigma);
  PRINT_FIELD(fptr, m, nChunks);
  PRINT_FIELD(fptr, m, nrPadded);
  PRINT_FIELD(fptr, m, nElems);
  PRINT_INT_ARRAY(fptr, m, oldToNewPerm, m->nr);
  PRINT_INT_ARRAY(fptr, m, newToOldPerm, m->nr);
  PRINT_INT_ARRAY(fptr, m, chunkLens, m->nChunks);
  PRINT_INT_ARRAY(fptr, m, chunkPtr, m->nChunks + 1);
  PRINT_INT_ARRAY(fptr, m, colInd, m->nElems);
  PRINT_V_ELE_ARRAY(fptr, m, val, m->nElems);
}

void spMVM(Matrix *m, const V_ELE *restrict x, V_ELE *restrict y)
{
  CG_UINT *colInd    = m->colInd;
  V_ELE *val         = m->val;

  CG_UINT numChunks  = m->nChunks;
  CG_UINT C          = m->C;
  CG_UINT *chunkPtr  = m->chunkPtr;
  CG_UINT *chunkLens = m->chunkLens;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numChunks; ++i) {
    V_ELE tmp[C];
    for (int j = 0; j < C; ++j) {
      tmp[j] = 0.0;
    }

    int chunkOffset = chunkPtr[i];
    for (int j = 0; j < chunkLens[i]; ++j) {
      // NOTE: SIMD should be applied here
      for (int k = 0; k < C; ++k) {
        tmp[k] += val[chunkOffset + j * C + k] * x[colInd[chunkOffset + j * C + k]];
      }
    }

    for (int j = 0; j < C; ++j) {
      y[i * C + j] = tmp[j];
    }
  }
}

void spMMVM(Matrix *m, const DMatrix *x, DMatrix *y)
{
  CG_UINT *colInd    = m->colInd;
  V_ELE *val         = m->val;

  CG_UINT numChunks  = m->nChunks;
  CG_UINT C          = m->C;
  CG_UINT *chunkPtr  = m->chunkPtr;
  CG_UINT *chunkLens = m->chunkLens;

  CG_UINT numVecs    = x->nc; // number of vectors in the block

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numChunks; ++i) {
    V_ELE tmp[C * numVecs];
    for (int j = 0; j < C * numVecs; ++j) {
      tmp[j] = 0.0;
    }

    int chunkOffset = chunkPtr[i];
    for (int j = 0; j < chunkLens[i]; ++j) {
      // NOTE: SIMD should be applied here
      for (int k = 0; k < C; ++k) {
        CG_UINT col = colInd[chunkOffset + j * C + k];
        V_ELE a     = val[chunkOffset + j * C + k];
        for (int v = 0; v < numVecs; ++v) {
          tmp[k * numVecs + v] += a * x->entries[col * numVecs + v];
        }
      }
    }

    for (int j = 0; j < C; ++j) {
      for (int v = 0; v < numVecs; ++v) {
        y->entries[(i * C + j) * numVecs + v] = tmp[j * numVecs + v];
      }
    }
  }
}
