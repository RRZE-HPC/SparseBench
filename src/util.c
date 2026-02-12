/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

char *changeFileEnding(char *filename, char *newEnding)
{

  char *dot, *newname;
  int len;

  /* Get position of the last dot in the name */
  dot = strrchr(filename, '.');
  /* If a dot was found, calculate the length to this point */
  if (dot) {
    len = dot - filename;
  } else {
    len = strlen(filename);
  }

  newname = malloc(len + 6);
  strncpy(newname, filename, len);
  strcpy(newname + len, newEnding);

  return newname;
}
