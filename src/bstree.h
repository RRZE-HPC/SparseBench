/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __BSTREE_H
#define __BSTREE_H

#include "util.h"
typedef struct node {
  CG_UINT key;
  CG_UINT value;
  struct node* left;
  struct node* right;
} node;

extern void bstNew(node**, CG_UINT key, CG_UINT value);
extern void bstFree(node* root);
extern CG_UINT bstSearch(node*, CG_UINT key);
extern void bstInsert(node*, CG_UINT key, CG_UINT value);
#endif
