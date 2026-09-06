/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file. */
#ifndef MATRIXSCAMAC_H
#define MATRIXSCAMAC_H

#include <stdbool.h>

#include "matrix.h"

/* Matrix-source scheme for the -m option: everything after the prefix is a
 * ScaMaC argument string, e.g. "scamac:Anderson,Lx=100,Ly=100,Lz=100". */
#define SCAMAC_ARG_PREFIX "scamac:"

// true if filename selects the ScaMaC generator (prefix match).
extern bool matrixIsScamac(const char *filename);

// Generate the ScaMaC matrix described by matarg (the full -m value,
// SCAMAC_ARG_PREFIX stripped here) into m.
extern void matrixGenerateScamac(GMatrix *m, const char *matarg, int rank, int size);

#endif /* MATRIXSCAMAC_H */
