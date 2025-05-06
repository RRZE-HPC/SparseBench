/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SCSmatrix.h"

void spMVM(Matrix *m, const CG_FLOAT *restrict x, CG_FLOAT *restrict y) {
  CG_UINT *colInd = m->colInd;
  CG_FLOAT *val = m->val;

  CG_UINT numChunks = m->nChunks;
  CG_UINT C = m->C;
  CG_UINT *chunkPtr = m->chunkPtr;
  CG_UINT *chunkLens = m->chunkLens;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < numChunks; ++i) {
    CG_FLOAT tmp[C];
    for (int j = 0; j < C; ++j) {
      tmp[j] = 0.0;
    }

    int chunkOffset = chunkPtr[i];
    for (int j = 0; j < chunkLens[i]; ++j) {
      // NOTE: SIMD should be applied here
      for (int k = 0; k < C; ++k) {
        tmp[k] +=
            val[chunkOffset + j * C + k] * x[colInd[chunkOffset + j * C + k]];
      }
    }

    for (int j = 0; j < C; ++j) {
      y[i * C + j] = tmp[j];
    }
  }
}
