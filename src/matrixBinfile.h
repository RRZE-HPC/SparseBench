/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __MATRIXBINFILE_H_
#define __MATRIXBINFILE_H_
#include "Comm.h"
#include "Matrix.h"
#include "util.h"

// Matrix binary file format:
// All ints are unsigned 32bit ints. All floats are float16 or float32.
// <number of rows> <number of non zeroes>
// array of size <number of rows>[<row offset>]
// array of size <number of non zeroes>[<<col id>,<value>>]

extern void matrixBinWrite(Matrix* m, Comm* c, char* filename);
extern void matrixBinRead(Matrix* m, Comm* c, char* filename);

#endif // __MATRIXBINFILE_H_
