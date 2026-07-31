/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/* Checks the invariant the communication/computation overlap in the CG solver
 * relies on:
 *
 *   spMVM_local(A, x, y) followed by spMVM_external(A, x, y) == spMVM(A, x, y)
 *
 * and that chunking the local phase (which is what lets the solver nudge MPI
 * progress mid-SpMV) does not change the result. Under MPI this invariant can
 * only be observed indirectly through the converged residual, so it is pinned
 * here with a hand-built matrix instead.
 */

#include "solverTestsSPMVSplit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef CRS

#include "../../src/matrix.h"
#include "../../src/vtype.h"

#ifdef USE_COMPLEX
#define VDIFF(a, b) VABS((a) - (b))
#else
#define VDIFF(a, b) fabs((double)((a) - (b)))
#endif

#define TOL 1.0e-12

/* 6 local rows, 4 external columns (6..9). Entries are already ordered
 * local-before-external within each row, which is what reorderMatrixForOverlap
 * guarantees at run time. Row 0 and row 4 are purely local, row 3 is purely
 * external, the rest are mixed. */
#define TEST_NR 6
#define TEST_NC 10
#define TEST_NNZ 16

static CG_UINT testRowPtr[TEST_NR + 1]   = { 0, 2, 6, 9, 10, 13, 16 };
static CG_UINT testRowLocalEnd[TEST_NR]  = { 2, 5, 7, 9, 13, 14 };
static CG_UINT testBoundaryRows[]        = { 1, 2, 3, 5 };
static CG_UINT testColInd[TEST_NNZ]      = { 0, 1, 0, 1, 2, 6, 2, 6, 7, 8, 3, 4, 5, 5, 7,
       9 };

static V_ELE testVal[TEST_NNZ];
static V_ELE testX[TEST_NC];

static void initData(void)
{
  for (int j = 0; j < TEST_NNZ; j++) {
    testVal[j] = (V_ELE)(0.5 * (double)(j + 1));
  }

  for (int i = 0; i < TEST_NC; i++) {
    testX[i] = (V_ELE)(1.0 + 0.25 * (double)i);
  }
}

static void initMatrix(Matrix *m, int withExternals)
{
  m->nr       = TEST_NR;
  m->nc       = TEST_NC;
  m->nnz      = TEST_NNZ;
  m->totalNr  = TEST_NR;
  m->totalNnz = TEST_NNZ;
  m->startRow = 0;
  m->stopRow  = TEST_NR - 1;
  m->rowPtr   = testRowPtr;
  m->colInd   = testColInd;
  m->val      = testVal;

  if (withExternals) {
    m->rowLocalEnd   = testRowLocalEnd;
    m->boundaryRows  = testBoundaryRows;
    m->nBoundaryRows = sizeof(testBoundaryRows) / sizeof(testBoundaryRows[0]);
  } else {
    /* No localization happened: every entry counts as local and the external
     * phase must degenerate into a no-op. */
    static CG_UINT allLocalEnd[TEST_NR];
    for (int i = 0; i < TEST_NR; i++) {
      allLocalEnd[i] = testRowPtr[i + 1];
    }
    m->rowLocalEnd   = allLocalEnd;
    m->boundaryRows  = NULL;
    m->nBoundaryRows = 0;
  }
}

static int compare(const char *what, const V_ELE *ref, const V_ELE *got)
{
  for (int i = 0; i < TEST_NR; i++) {
    double diff = VDIFF(ref[i], got[i]);
    if (diff > TOL) {
      printf("  %s: row %d differs by %E\n", what, i, diff);
      return 1;
    }
  }
  return 0;
}

/* local + external must reproduce the full SpMV */
static int testSplitMatchesFull(void)
{
  Matrix m;
  V_ELE ref[TEST_NR];
  V_ELE got[TEST_NR];

  initMatrix(&m, 1);

  spMVM(&m, testX, ref);

  /* Poison the output first: spMVM_local has to overwrite, not accumulate. */
  for (int i = 0; i < TEST_NR; i++) {
    got[i] = (V_ELE)-7.0;
  }

  spMVM_local(&m, testX, got);
  spMVM_external(&m, testX, got);

  return compare("split vs full", ref, got);
}

/* Walking the local phase in chunks must not change anything */
static int testChunkedLocal(void)
{
  Matrix m;
  V_ELE ref[TEST_NR];
  V_ELE got[TEST_NR];

  initMatrix(&m, 1);

  spMVM_local(&m, testX, ref);

  for (int i = 0; i < TEST_NR; i++) {
    got[i] = (V_ELE)-7.0;
  }

  /* Deliberately uneven chunks, including an empty one at the end */
  spMVM_local_range(&m, testX, got, 0, 1);
  spMVM_local_range(&m, testX, got, 1, 4);
  spMVM_local_range(&m, testX, got, 4, TEST_NR);
  spMVM_local_range(&m, testX, got, TEST_NR, TEST_NR);

  return compare("chunked local", ref, got);
}

/* Without localization there are no external entries, so the local phase alone
 * already is the full SpMV and the external phase must do nothing. */
static int testNoExternals(void)
{
  Matrix m;
  V_ELE ref[TEST_NR];
  V_ELE got[TEST_NR];

  initMatrix(&m, 0);

  spMVM(&m, testX, ref);

  for (int i = 0; i < TEST_NR; i++) {
    got[i] = (V_ELE)-7.0;
  }

  spMVM_local(&m, testX, got);
  spMVM_external(&m, testX, got);

  return compare("no externals", ref, got);
}

int solverTestsSPMVSplit(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  initData();

  struct {
    const char *name;
    int (*func)(void);
  } tests[] = {
    { "split SpMV == full SpMV", testSplitMatchesFull },
    { "chunked local phase", testChunkedLocal },
    { "no external entries", testNoExternals },
  };

  int numTests = sizeof(tests) / sizeof(tests[0]);
  int passed   = 0;

  printf("Running %d split SpMV tests:\n", numTests);
  for (int i = 0; i < numTests; i++) {
    printf("[%-2d/%-2d] %-30s ... \n", i + 1, numTests, tests[i].name);
    fflush(stdout);

    if (!tests[i].func()) {
      printf("PASS\n");
      passed++;
    } else {
      printf("FAIL\n");
    }
  }

  printf("\nSummary: %d/%d split SpMV tests passed.\n", passed, numTests);

  return (passed == numTests) ? 0 : 1;
}

#else /* !CRS */

int solverTestsSPMVSplit(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  /* The split kernels only exist for CRS */
  return 0;
}

#endif /* CRS */
