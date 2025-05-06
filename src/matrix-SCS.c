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

static inline int compareDesc(const void *a, const void *b)
{
  const int val_a = *(const int *)a;
  const int val_b = *(const int *)b;

  return (val_b - val_a);
}

static inline int compareDescSCS(const void* a, const void* b) {
  
  const SellCSigmaPair* pa = (const SellCSigmaPair*)a;
  const SellCSigmaPair* pb = (const SellCSigmaPair*)b;

  if (pa->count < pb->count) return 1;  // Descending order
  if (pa->count > pb->count) return -1;
  return 0;  // Stable if equal
}

void dumpVectorToFile(CG_FLOAT* vec, int size, FILE* file){
  fprintf(file, "vec = ");
  for(int i = 0; i < size; ++i){
    fprintf(file, "%f, ", vec[i]);
  }
}

void dumpSCSMatrixToFile(Matrix* m, FILE* file){
  fprintf(file, "m->startRow = %d\n", m->startRow);
  fprintf(file, "m->stopRow = %d\n", m->stopRow);
  fprintf(file, "m->totalNr = %d\n", m->totalNr);
  fprintf(file, "m->totalNnz = %d\n", m->totalNnz);
  fprintf(file, "m->nr = %d\n", m->nr);      
  fprintf(file, "m->nc = %d\n", m->nc);      
  fprintf(file, "m->nnz = %d\n", m->nnz);    
  fprintf(file, "m->C = %d\n", m->C);       
  fprintf(file, "m->sigma = %d\n", m->sigma);   
  fprintf(file, "m->nChunks = %d\n", m->nChunks); 
  fprintf(file, "m->nrPadded = %d\n", m->nrPadded);
  fprintf(file, "m->nElems = %d\n", m->nElems);

  // Dump permutation arrays
  fprintf(file, "oldToNewPerm: ");
  for(int i = 0; i < m->nr; ++i){
    fprintf(file, "%d, ", m->oldToNewPerm[i]);
  }
  fprintf(file, "\n");
  fprintf(file, "newToOldPerm: ");
  for(int i = 0; i < m->nr; ++i){
    fprintf(file, "%d, ", m->newToOldPerm[i]);
  }
  fprintf(file, "\n");

  // Dump chunk data
  fprintf(file, "chunkLens: ");
  for(int i = 0; i < m->nChunks; ++i){
    fprintf(file, "%d, ", m->chunkLens[i]);
  }
  fprintf(file, "\n");
  fprintf(file, "chunkPtr: ");
  for(int i = 0; i < m->nChunks+1; ++i){
    fprintf(file, "%d, ", m->chunkPtr[i]);
  }
  fprintf(file, "\n");

  // Dump matrix data
  fprintf(file, "colInd: ");
  for(int i = 0; i < m->nElems; ++i){
    fprintf(file, "%d, ", m->colInd[i]);
  }
  fprintf(file, "\n");
  fprintf(file, "val: ");
  for(int i = 0; i < m->nElems; ++i){
    fprintf(file, "%f, ", m->val[i]);
  }
  fprintf(file, "\n");
}

// This version just goes to stdout
void dumpSCSMatrix(Matrix* m)
{
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
  for(int i = 0; i < m->nr; ++i){
    printf("%d, ", m->oldToNewPerm[i]);
  }
  printf("\n");
  printf("newToOldPerm: ");
  for(int i = 0; i < m->nr; ++i){
    printf("%d, ", m->newToOldPerm[i]);
  }
  printf("\n");

  // Dump chunk data
  printf("chunkLens: ");
  for(int i = 0; i < m->nChunks; ++i){
    printf("%d, ", m->chunkLens[i]);
  }
  printf("\n");
  printf("chunkPtr: ");
  for(int i = 0; i < m->nChunks+1; ++i){
    printf("%d, ", m->chunkPtr[i]);
  }
  printf("\n");

  // Dump matrix data
  printf("colInd: ");
  for(int i = 0; i < m->nElems; ++i){
    printf("%d, ", m->colInd[i]);
  }
  printf("\n");
  printf("val: ");
  for(int i = 0; i < m->nElems; ++i){
    printf("%f, ", m->val[i]);
  }
  printf("\n");
}

void dumpMMMatrix(MmMatrix* mm)
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
    printf("%.2e total rows and %.2e nonzeros\n",
        (double)total_nrow,
        (double)local_nnz);
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
    printf("Unable to open file.\n");
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
  m->nnz   = cursor;
  m->count = cursor;

  // sort by column
  qsort(m->entries, m->count, sizeof(Entry), compareColumn);
// sort by row requires a stable sort. As glibc qsort is mergesort this
// hopefully works.
#ifdef __linux__
  qsort(m->entries, m->count, sizeof(Entry), compareRow);
#else
  // BSD has a dedicated mergesort available in its libc
  mergesort(m->entries, m->count, sizeof(Entry), compareRow);
#endif
}

void matrixConvertMMtoCRS(MmMatrix* mm, Matrix* m, int rank, int size)
{
  m->startRow = mm->startRow;
  m->stopRow  = mm->stopRow;
  m->totalNr  = mm->totalNr;
  m->totalNnz = mm->totalNnz;
  m->nr       = mm->nr;
  m->nc       = mm->nr;
  m->nnz      = mm->nnz;

  m->rowPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      (m->nr + 1) * sizeof(CG_UINT));
  m->colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_UINT));
  m->val    = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m->nnz * sizeof(CG_FLOAT));

  int* valsPerRow = (int*)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(int));

  for (int i = 0; i < m->nr; i++) {
    valsPerRow[i] = 0;
  }

  Entry* entries = mm->entries;
  int startRow   = m->startRow;

  for (int i = 0; i < mm->count; i++) {
    valsPerRow[entries[i].row - startRow]++;
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

void matrixConvertMMtoSCS(MmMatrix* mm, Matrix* m, int rank, int size)
{
  m->startRow = mm->startRow;
  m->stopRow  = mm->stopRow;
  m->totalNr  = mm->totalNr;
  m->totalNnz = mm->totalNnz;
  m->nr       = mm->nr;
  m->nc       = mm->nr;
  m->nnz      = mm->nnz;
  m->nChunks  = (m->nr + m->C - 1) / m->C;
  m->nrPadded = m->nChunks * m->C;

  // (Temporary array) Assign an index to each row to use for row sorting
  SellCSigmaPair* elemsPerRow = (SellCSigmaPair*)allocate(ARRAY_ALIGNMENT, m->nrPadded * sizeof(SellCSigmaPair));

  for(int i = 0; i < m->nrPadded; ++i){
    elemsPerRow[i].index = i;
    elemsPerRow[i].count = 0;
  }

  // Collect the number of elements in each row
  for(int i = 0; i < m->nnz; ++i){
    Entry e = mm->entries[i];
    ++(elemsPerRow[e.row].count);
  }

  // Sort rows over a scope of sigma
  for(int i = 0; i < m->nrPadded; i += m->sigma){
    int chunkStart = i;
    int chunkStop = ((i + m->sigma) < m->nrPadded) 
                  ? i + m->sigma : m->nrPadded;
    int size = chunkStop - chunkStart;

  // Sorting rows by element count using struct keeps index/count together
#ifdef __linux__
    qsort(&elemsPerRow[chunkStart], size, sizeof(SellCSigmaPair), compareDescSCS);
#else
    // BSD has a dedicated mergesort available in its libc
    mergesort(&elemsPerRow[chunkStart], size, sizeof(SellCSigmaPair), compareDescSCS);
#endif
  }

  m->chunkLens = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nChunks * sizeof(CG_UINT));
  m->chunkPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT, (m->nChunks + 1) * sizeof(CG_UINT));
  
  CG_UINT currentChunkPtr = 0;

  for(int i = 0; i < m->nChunks; ++i){
    // Note sure about this yet
    // int chunkStart = elemsPerRow[i * m->C].count;
    // int chunkStop = ((i * m->C + m->C) < m->nrPadded) 
    //               ? elemsPerRow[i * m->C + m->C].count
    //               : elemsPerRow[m->nrPadded - 1].count;
    SellCSigmaPair chunkStart = elemsPerRow[i * m->C];
    SellCSigmaPair chunkStop = (i * m->C + m->C) < (m->nrPadded - 1) 
                             ? elemsPerRow[i * m->C + m->C]
                             : elemsPerRow[m->nrPadded - 1];

    int size = chunkStop.index - chunkStart.index; 

    // Collect longest row in chunk as chunk length
    CG_UINT maxLength = 0;
    for(int j = 0; j < m->C; ++j){
      CG_UINT rowLenth = elemsPerRow[i * m->C + j].count;
      if(rowLenth > maxLength) maxLength = rowLenth;
    }

    // Collect chunk data to arrays
    m->chunkLens[i] = (CG_UINT)maxLength;
    m->chunkPtr[i] = (CG_UINT)currentChunkPtr;
    currentChunkPtr += m->chunkLens[i] * m->C;
  }

  // Account for final chunk
  m->nElems = m->chunkPtr[m->nChunks - 1] + \
    m->chunkLens[m->nChunks - 1] * m->C;

  m->chunkPtr[m->nChunks] = (CG_UINT)m->nElems;

  // Construct permutation vector
  m->oldToNewPerm = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(CG_UINT));
  for(int i = 0; i < m->nrPadded; ++i){
    CG_UINT oldRow = elemsPerRow[i].index;
    if(oldRow < m->nr) m->oldToNewPerm[oldRow] = (CG_UINT)i;
  }

  // Construct inverse permutation vector
  m->newToOldPerm = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nr * sizeof(CG_UINT));
  for(int i = 0; i < m->nr; ++i){
#ifdef VERBOSE
    // Sanity check for common error
    if(m->oldToNewPerm[i] >= m->nr){
      fprintf(stderr, "ERROR matrixConvertMMtoSCS: m->oldToNewPerm[%d]=%d" \
        " is out of bounds (>%d).\n", i, m->oldToNewPerm[i], m->nr);
    }
#endif
    m->newToOldPerm[m->oldToNewPerm[i]] = (CG_UINT)i;
  }

  // Now that chunk data is collected, fill with matrix data
  m->colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT, m->nElems * sizeof(CG_UINT));
  m->val = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, m->nElems * sizeof(CG_FLOAT));

  // Initialize defaults (essential for padded elements)
  for(int i = 0; i < m->nElems; ++i){
    m->val[i] = (CG_FLOAT)0.0;
    m->colInd[i] = (CG_UINT)0;
    // TODO: may need to offset when used with MPI
    // m->colInd[i] = padded_val; 
  }

  // (Temporary array) Keep track of how many elements we've seen in each row
  int* rowLocalElemCount = (int*)allocate(ARRAY_ALIGNMENT, m->nrPadded * sizeof(int));
  for(int i = 0; i < m->nrPadded; ++i){
    rowLocalElemCount[i] = 0;
  }

  for(int i = 0; i < m->nnz; ++i){
    Entry e = mm->entries[i];

    int rowOld = e.row;
    int row = m->oldToNewPerm[rowOld];
    int chunkIdx = row / m->C;
    int chunkStart = m->chunkPtr[chunkIdx];
    int chunkRow = row % m->C;
    int idx = chunkStart + rowLocalElemCount[row] * m->C + chunkRow;
    m->colInd[idx] = (CG_UINT)e.col;
#ifdef VERBOSE
    // Sanity check for common error
    if(m->colInd[idx] >= m->nc){
      fprintf(stderr, "ERROR matrixConvertMMtoSCS: m->colInd[%d]=%d" \
        " is out of bounds (>%d).\n", idx, m->colInd[idx], m->nc);
    }
#endif
    m->val[idx] = (CG_FLOAT)e.val;
    ++rowLocalElemCount[row];
  }

  free(elemsPerRow);
  free(rowLocalElemCount);
}
