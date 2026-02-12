/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __BSTREE_H
#define __BSTREE_H

#include "stdbool.h"
#include "util.h"
typedef struct node {
  CG_UINT key;
  CG_UINT value;
  struct node *left;
  struct node *right;
} Node;

typedef struct {
  Node *root;
} Bstree;

extern Bstree *bstNew(void);
extern void bstFree(Bstree *);
extern CG_UINT bstFind(Bstree *, CG_UINT key);
extern bool bstExists(Bstree *, CG_UINT key);
extern void bstInsert(Bstree *, CG_UINT key, CG_UINT value);
extern void bstWalk(Bstree *leaf);
#endif
