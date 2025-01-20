/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "matrix.h"
#include "solver.h"
#include "util.h"

static void matrixGenerate(
    Parameter* p, Solver* s, Comm* c, bool use_7pt_stencil)
{
  int size = c->size;
  int rank = c->rank;

  CG_UINT local_nrow = p->nx * p->ny * p->nz;
  CG_UINT local_nnz  = 27 * local_nrow;

  CG_UINT total_nrow = local_nrow * size;
  CG_UINT total_nnz  = 27 * total_nrow;

  int start_row = local_nrow * rank;
  int stop_row  = start_row + local_nrow - 1;

  if (commIsMaster(c)) {
    if (use_7pt_stencil) {
      printf("Generate 7pt matrix with ");
    } else {
      printf("Generate 27pt matrix with ");
    }
    printf("%d total rows and %d nonzeros\n", (int)total_nrow, (int)local_nnz);
  }

  s->A.val = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, local_nnz * sizeof(CG_FLOAT));
  s->A.colInd = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      local_nnz * sizeof(CG_UINT));
  s->A.rowPtr = (CG_UINT*)allocate(ARRAY_ALIGNMENT,
      (local_nrow + 1) * sizeof(CG_UINT));
  s->x = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, local_nrow * sizeof(CG_FLOAT));
  s->b = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT, local_nrow * sizeof(CG_FLOAT));
  s->xexact = (CG_FLOAT*)allocate(ARRAY_ALIGNMENT,
      local_nrow * sizeof(CG_FLOAT));

  CG_FLOAT* curvalptr = s->A.val;
  CG_UINT* curindptr  = s->A.colInd;
  CG_UINT* currowptr  = s->A.rowPtr;
  CG_FLOAT* x         = s->x;
  CG_FLOAT* b         = s->b;
  CG_FLOAT* xexact    = s->xexact;

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
              //            Since we have a stack of nx by ny by nz domains
              //            , stacking in the z direction, we check to see
              //            if sx and sy are reaching outside of the domain,
              //            while the check for the curcol being valid is
              //            sufficient to check the z values
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
        x[curlocalrow]      = 0.0;
        b[curlocalrow]      = 27.0 - ((CG_FLOAT)(nnzrow - 1));
        xexact[curlocalrow] = 1.0;
      } // end ix loop
    } // end iy loop
  } // end iz loop

#ifdef VERBOSE
  printf("Process %d of %d has %d rows\n", rank, size, local_nrow);
  printf("Global rows %d through %d\n", start_row, stop_row);
  printf("%d nonzeros\n", local_nnz);
#endif /* ifdef VERBOSE */

  s->A.startRow = start_row;
  s->A.stopRow  = stop_row;
  s->A.totalNr  = total_nrow;
  s->A.totalNnz = total_nnz;
  s->A.nr       = local_nrow;
  s->A.nc       = local_nrow;
  s->A.nnz      = local_nnz;
}

void initSolver(Solver* s, Comm* c, Parameter* p)
{
  if (!strcmp(p->filename, "generate")) {
    matrixGenerate(p, s, c, false);
  } else if (!strcmp(p->filename, "generate7P")) {
    matrixGenerate(p, s, c, true);
  } else {
    matrixRead(&s->A, p->filename);
  }
}

void spMVM(Matrix* m, const CG_FLOAT* restrict x, CG_FLOAT* restrict y)
{
  CG_UINT numRows = m->nr;
  CG_UINT* rowPtr = m->rowPtr;
  CG_UINT* colInd = m->colInd;
  CG_FLOAT* val   = m->val;

  for (int rowID = 0; rowID < numRows; rowID++) {
    CG_FLOAT tmp = y[rowID];

    // loop over all elements in row
    for (int rowEntry = rowPtr[rowID]; rowEntry < rowPtr[rowID + 1];
        rowEntry++) {
      tmp += val[rowEntry] * x[colInd[rowEntry]];
    }

    y[rowID] = tmp;
  }
}

void waxpby(const CG_UINT n,
    const CG_FLOAT alpha,
    const CG_FLOAT* restrict x,
    const CG_FLOAT beta,
    const CG_FLOAT* restrict y,
    CG_FLOAT* const w)
{
  if (alpha == 1.0) {
    for (int i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
}

void ddot(const CG_UINT n,
    const CG_FLOAT* restrict x,
    const CG_FLOAT* restrict y,
    CG_FLOAT* restrict result)
{
  CG_FLOAT sum = 0.0;

  if (y == x) {
    for (int i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }

  commReduction(&sum, SUM);
  *result = sum;
}
