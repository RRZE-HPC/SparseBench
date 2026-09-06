/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file. */
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "matrix.h"
#include "matrixScamac.h"
#include "util.h"
#include "vtype.h"

/* Works without ScaMaC support so that initMatrix can route "scamac:" inputs
 * to a proper error message instead of the generic file-format lookup. */
bool matrixIsScamac(const char *filename)
{
  return filename != NULL &&
         strncmp(filename, SCAMAC_ARG_PREFIX, strlen(SCAMAC_ARG_PREFIX)) == 0;
}

#ifdef _SCAMAC
#include <scamac.h>

static void die(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n--> Abort.\n");
  exit(EXIT_FAILURE);
}

/* Every ScaMaC call is checked: prints the library's own description of the
 * error code plus the call-site context. fmt gives the context. */
#define SCAMAC_CALL(err_, ...)                                                           \
  do {                                                                                   \
    ScamacErrorCode scamac_err_ = (err_);                                                \
    if (scamac_err_ != SCAMAC_EOK) {                                                     \
      fprintf(stderr,                                                                    \
          "ScaMaC error: %s%s\nwhile ",                                                  \
          scamac_error_desc(scamac_err_),                                                \
          scamac_error_dpar(scamac_err_));                                               \
      fprintf(stderr, __VA_ARGS__);                                                      \
      fprintf(stderr, "\n--> Abort.\n");                                                 \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  } while (0)

void matrixGenerateScamac(GMatrix *m, const char *matarg, int rank, int size)
{
  const char *argstr   = matarg + strlen(SCAMAC_ARG_PREFIX);

  ScamacGenerator *gen = NULL;
  char *errstr         = NULL;

  ScamacErrorCode err  = scamac_parse_argstr(argstr, &gen, &errstr);
  if (err != SCAMAC_EOK || gen == NULL) {
    die("Problem with ScaMaC argument string '%s':\n %s",
        argstr,
        errstr ? errstr : "unknown error");
  }

  err = scamac_generator_check(gen, &errstr);
  if (err != SCAMAC_EOK) {
    die("Problem with parameters of '%s':\n %s", argstr, errstr ? errstr : "");
  }
  /* The check call allocates the description string even when it returns
   * SCAMAC_EOK; it is caller-owned either way. */
  free(errstr);
  errstr = NULL;

  err = scamac_generator_finalize(gen);
  if (err == SCAMAC_EOVERFLOW) {
    die("ScaMaC matrix '%s': dimension exceeds the maximal index value", argstr);
  }
  SCAMAC_CALL(err, "finalizing the generator for '%s'", argstr);

  const char *name   = scamac_generator_query_name(gen);
  ScamacIdx nrow     = scamac_generator_query_nrow(gen);
  ScamacIdx maxnzrow = scamac_generator_query_maxnzrow(gen);
  bool gen_cplx      = scamac_generator_query_valtype(gen) == SCAMAC_VAL_COMPLEX;

#ifndef USE_COMPLEX
  if (gen_cplx) {
    die("ScaMaC matrix '%s' has complex values: rebuild with "
        "USE_COMPLEX_ELEMENTS=true",
        argstr);
  }
#endif

  /* Row partition, closed form of sizeOfRank() in comm.c:
   * the first nrow % size ranks get one extra row. stopRow is inclusive. */
  ScamacIdx rowsBase  = nrow / size;
  ScamacIdx rem       = nrow % size;
  ScamacIdx localRows = rowsBase + (rem > rank ? 1 : 0);
  ScamacIdx startRow  = rank * rowsBase + (rank < rem ? rank : rem);
  ScamacIdx stopRow   = startRow + localRows - 1;

  /* Entry indices and counts are CG_UINT; with 32 bit indices the global
   * row count or this rank's nnz can overflow long before generation
   * starts (same constraint as the 27pt stencil's size_t cursor fix). */
#if UINT_TYPE == 1
  if ((uint64_t)nrow > (uint64_t)UINT_MAX ||
      (uint64_t)maxnzrow * (uint64_t)localRows > (uint64_t)UINT_MAX) {
    die("ScaMaC matrix '%s' (nrow %lld, up to %lld entries/row) exceeds the "
        "32 bit index range: rebuild with UINT_TYPE=ULL",
        argstr,
        (long long)nrow,
        (long long)maxnzrow);
  }
#endif

  if (!rank) {
    printf("Generate ScaMaC matrix %s [%s] with %.2e total rows and up to %.2e "
           "nonzeros per row\n",
        name,
        argstr,
        (double)nrow,
        (double)maxnzrow);
  }

  m->entries = (Entry *)allocate(
      ARRAY_ALIGNMENT, (size_t)maxnzrow * (size_t)localRows * sizeof(Entry));
  m->rowPtr =
      (CG_UINT *)allocate(ARRAY_ALIGNMENT, ((size_t)localRows + 1) * sizeof(CG_UINT));

  /* Per-row scratch sized like the upstream examples: maxnzrow doubles,
   * twice that for complex values (reinterpreted as double complex). */
  ScamacIdx *cind =
      (ScamacIdx *)allocate(ARRAY_ALIGNMENT, (size_t)maxnzrow * sizeof(ScamacIdx));
  double *val = (double *)allocate(
      ARRAY_ALIGNMENT, (size_t)maxnzrow * (gen_cplx ? 2 : 1) * sizeof(double));

  ScamacWorkspace *ws;
  SCAMAC_CALL(
      scamac_workspace_alloc(gen, &ws), "allocating the workspace of '%s'", argstr);

  CG_UINT cursor = 0;
  m->rowPtr[0]   = 0;

  for (ScamacIdx irow = startRow; irow <= stopRow; irow++) {
    ScamacIdx nzr = 0;
    SCAMAC_CALL(scamac_generate_row(gen, ws, irow, SCAMAC_DEFAULT, &nzr, cind, val),
        "generating row %lld of '%s'",
        (long long)irow,
        argstr);

    CG_UINT row = (CG_UINT)(irow - startRow);

    for (ScamacIdx j = 0; j < nzr; j++) {
#ifdef USE_COMPLEX
      if (gen_cplx) {
        double complex *cval   = (double complex *)val;
        m->entries[cursor].val = VCONST(creal(cval[j]), cimag(cval[j]));
      } else {
        m->entries[cursor].val = VCONST(val[j], 0.0);
      }
#else
      m->entries[cursor].val = (V_ELE)val[j];
#endif
      /* global column index; commLocalization maps it into the local space */
      m->entries[cursor].col = (CG_UINT)cind[j];
      cursor++;
    }
    m->rowPtr[row + 1] = cursor;
  }

  m->startRow = (CG_UINT)startRow;
  m->stopRow  = (CG_UINT)stopRow;
  m->totalNr  = (CG_UINT)nrow;
  m->nr       = (CG_UINT)localRows;
  m->nc       = (CG_UINT)localRows;
  /* allocMatrix sizes the format conversion by nnz while the copy loop runs
   * to rowPtr[nr], so nnz is the exact fill-level cursor. */
  m->nnz = cursor;

  /* Only needed for the profiler's flop accounting: exact on one rank, an
   * upper bound on several (nrow * maxnzrow), clamped where CG_UINT is 32
   * bit instead of silently wrapping like the stencil generator does. */
  uint64_t totalNnz =
      (size == 1) ? (uint64_t)cursor : (uint64_t)nrow * (uint64_t)maxnzrow;
#if UINT_TYPE == 1
  if (totalNnz > (uint64_t)UINT_MAX) {
    totalNnz = (uint64_t)UINT_MAX;
  }
#endif
  m->totalNnz = (CG_UINT)totalNnz;

#ifdef VERBOSE
  printf("Process %d of %d has %lld rows\n", rank, size, (long long)localRows);
  printf("Global rows %lld through %lld\n", (long long)startRow, (long long)stopRow);
  printf("%llu nonzeros\n", (unsigned long long)cursor);
#endif

  SCAMAC_CALL(scamac_workspace_free(ws), "freeing the workspace of '%s'", argstr);
  SCAMAC_CALL(scamac_generator_destroy(gen), "destroying the generator of '%s'", argstr);
  deallocate(cind);
  deallocate(val);
}

#else /* !_SCAMAC */

void matrixGenerateScamac(GMatrix *m, const char *matarg, int rank, int size)
{
  (void)m;
  (void)rank;
  (void)size;
  fprintf(stderr,
      "Matrix source '%s' requires ScaMaC support: rebuild with "
      "ENABLE_SCAMAC=true (see mk/include_SCAMAC.mk).\n--> Abort.\n",
      matarg);
  exit(EXIT_FAILURE);
}

#endif /* _SCAMAC */
