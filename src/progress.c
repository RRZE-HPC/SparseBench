/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "progress.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double _end;
static int _current;

void initProgress(double end)
{
  _end     = end;
  _current = 0;

  printf("[          ]");
  fflush(stdout);
}

void printProgress(double current)
{
  int new = (int)rint((current / _end) * 10.0);

  if (new > _current) {
    char progress[11];
    _current    = new;
    progress[0] = 0;

    for (int i = 0; i < 10; i++) {
      if (i < _current) {
        sprintf(progress + strlen(progress), "#");
      } else {
        sprintf(progress + strlen(progress), " ");
      }
    }
    printf("\r[%s]", progress);
  }
  fflush(stdout);
}

void stopProgress()
{
  printf("\n");
  fflush(stdout);
}
