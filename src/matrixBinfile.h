/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIXBINFILE_H_
#define __MATRIXBINFILE_H_
#include "comm.h"
#include "matrix.h"

typedef struct {
  unsigned int col;
  float val;
} FEntryType;

// Matrix binary file format:
// All ints are unsigned 32bit ints. All floats are float32.
// <number of rows> <number of non zeroes>
// array of size <number of rows>[<row offset>]
// array of size <number of non zeroes>[<<col id>,<value>>]

extern void matrixBinWrite(GMatrix *m, CommType *c, char *filename);
extern void matrixBinRead(GMatrix *m, CommType *c, char *filename);

#endif // __MATRIXBINFILE_H_
