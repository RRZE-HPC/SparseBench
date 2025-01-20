/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameter.h"
#define MAXLINE 4096

void initParameter(Parameter* param)
{
  param->filename = "generate";
  param->nx       = 100;
  param->ny       = 100;
  param->nz       = 1000;
  param->itermax  = 1000;
  param->eps      = 0.0001;
}

void readParameter(Parameter* param, const char* filename)
{
  FILE* fp = fopen(filename, "r");
  char line[MAXLINE];
  int i;

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

    char* tok = strtok(line, " ");
    char* val = strtok(NULL, " ");

#define PARSE_PARAM(p, f)                                                      \
  if (strncmp(tok, #p, sizeof(#p) / sizeof(#p[0]) - 1) == 0) {                 \
    param->p = f(val);                                                         \
  }
#define PARSE_STRING(p) PARSE_PARAM(p, strdup)
#define PARSE_INT(p)    PARSE_PARAM(p, atoi)
#define PARSE_REAL(p)   PARSE_PARAM(p, atof)

    if (tok != NULL && val != NULL) {
      PARSE_INT(nx);
      PARSE_INT(ny);
      PARSE_INT(nz);
      PARSE_INT(itermax);
      PARSE_REAL(eps);
    }
  }

  fclose(fp);
}

void printParameter(Parameter* param)
{
  printf("Parameters\n");
  printf("Geometry data:\n");
  printf("\tPoints (x, y, z): %d, %d, %d\n", param->nx, param->ny, param->nz);
  printf("Iterative solver parameters:\n");
  printf("\tMax iterations: %d\n", param->itermax);
  printf("\tepsilon (stopping tolerance) : %f\n", param->eps);
}
