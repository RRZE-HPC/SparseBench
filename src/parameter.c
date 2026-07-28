/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameter.h"

#include "cli.h" // BenchType / CHEBFD, for the ChebFD section of printParameter
#define MAXLINE 4096

void initParameter(Parameter *param)
{
  param->filename   = "generate";
  param->nx         = 100;
  param->ny         = 100;
  param->nz         = 100;
  param->itermax    = 150;
  param->eps        = 0.0;
  param->blockwidth = NUMVEC;
#ifdef SCS
  param->C     = SELL_CHUNK;
  param->Sigma = SELL_SIGMA;
#endif
  param->verbose = 0;
  param->device  = 0;
  // NTS : if spectrum bounds unpassed Gershgorin is used
  // NTS : lancozs kern mu=2 used by paper
  param->cheb.a           = 0.0;
  param->cheb.b           = 0.0;
  param->cheb.lam_lo      = 0.0;
  param->cheb.lam_hi      = 0.0;
  param->cheb.Np          = 0;
  param->cheb.NS          = 0;
  param->cheb.kernel      = 3;
  param->cheb.mu          = 2; // from paper
  param->cheb.have_bounds = 0;
  param->cheb.have_target = 0;
}

void readParameter(Parameter *param, const char *filename)
{
  FILE *fp = fopen(filename, "re");
  char line[MAXLINE];
  int i;

  int have_a = param->cheb.have_bounds;
  int have_b = param->cheb.have_bounds;
  int have_lam_lo = param->cheb.have_target;
  int have_lam_hi = param->cheb.have_target;

  if (!fp) {
    fprintf(stderr, "Could not open parameter file: %s\n", filename);
    exit(EXIT_FAILURE);
  }

  while (!feof(fp)) {
    line[0] = '\0';
    fgets(line, MAXLINE, fp);
    for (i = 0; line[i] != '\0' && line[i] != '#'; i++)
      ;
    line[i] = '\0';

    /* Delimiter set includes \n/\r so that values parsed from a line read by
     * fgets do not retain a trailing newline (which would break later strcmp
     * on string params such as `filename`). */
    char *tok = strtok(line, " \t\r\n");
    char *val = strtok(NULL, " \t\r\n");

#define PARSE_KEY(key, target, conv, ...)                                                \
  if (strcmp(tok, key) == 0) {                                                           \
    target = conv(val);                                                                  \
    __VA_ARGS__;                                                                         \
  }

    if (tok != NULL && val != NULL) {
      PARSE_KEY("filename", param->filename, strdup);
      PARSE_KEY("nx", param->nx, atoi);
      PARSE_KEY("ny", param->ny, atoi);
      PARSE_KEY("nz", param->nz, atoi);
      PARSE_KEY("itermax", param->itermax, atoi);
      PARSE_KEY("eps", param->eps, atof);
      // NTS : `cheb_` prefix, populating the nested struct
      PARSE_KEY("cheb_a", param->cheb.a, atof, have_a = 1);
      PARSE_KEY("cheb_b", param->cheb.b, atof, have_b = 1);
      PARSE_KEY("cheb_lam_lo", param->cheb.lam_lo, atof, have_lam_lo = 1);
      PARSE_KEY("cheb_lam_hi", param->cheb.lam_hi, atof, have_lam_hi = 1);
      PARSE_KEY("cheb_Np", param->cheb.Np, atoi);
      PARSE_KEY("cheb_NS", param->cheb.NS, atoi);
      PARSE_KEY("cheb_kernel", param->cheb.kernel, atoi);
      PARSE_KEY("cheb_mu", param->cheb.mu, atoi);
    }
  }

  // cheb_a and cheb_b must be supplied together; exactly one silently
  // enables user-bounds mode with the other left at its 0.0 default,
  // disabling the Gershgorin spectrum fallback.
  if (have_a != have_b) {
    fprintf(stderr,
        "Error: 'cheb_a' and 'cheb_b' must be supplied together "
        "(both or neither). Supplying only one silently disables the "
        "Gershgorin spectrum fallback.\n");
    exit(EXIT_FAILURE);
  }
  param->cheb.have_bounds = have_a;

  // Likewise for the target interval: a lone bound would silently pair with
  // the other's 0.0 default and target the wrong interval.
  if (have_lam_lo != have_lam_hi) {
    fprintf(stderr,
        "Error: 'cheb_lam_lo' and 'cheb_lam_hi' must be supplied together "
        "(both or neither).\n");
    exit(EXIT_FAILURE);
  }
  param->cheb.have_target = have_lam_lo;

  fclose(fp);
}

void printParameter(Parameter *param)
{
  printf("Parameters\n");
  printf("Iterative solver parameters:\n");
  printf("\tfile name: %s\n", param->filename);
  printf("\tnx: %d\n", param->nx);
  printf("\tny: %d\n", param->ny);
  printf("\tnz: %d\n", param->nz);
  printf("\tMax iterations: %d\n", param->itermax);
  printf("\tepsilon (stopping tolerance) : %f\n", param->eps);
  printf("\tBlock width: %d\n", param->blockwidth);
#ifdef SCS
  printf("\tSell chunk: %d\n", param->C);
  printf("\tSell sigma: %d\n", param->Sigma);
#endif
  printf("\tVerbose Level: %d\n", param->verbose);
  printf("\tGPU device index: %d\n", param->device);
  
  if (BenchType == CHEBFD) {
    printf("ChebFD parameters:\n");
    printf("\tspectrum [a,b]: %g, %g%s\n",
        param->cheb.a,
        param->cheb.b,
        param->cheb.have_bounds ? "" : " (auto: Gershgorin)");
    printf("\ttarget interval: [%g, %g]\n", param->cheb.lam_lo, param->cheb.lam_hi);
    printf("\tNp / NS: %d / %d\n", param->cheb.Np, param->cheb.NS);
    printf("\tkernel / mu: %d / %d\n", param->cheb.kernel, param->cheb.mu);
  }
}
