/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#include <stdio.h>
#include <stdlib.h>

#include "bstree.h"

void bstNew(node** root, CG_UINT key, CG_UINT value)
{
  (*root)        = malloc(sizeof(node));
  (*root)->key   = key;
  (*root)->value = value;
  (*root)->left  = NULL;
  (*root)->right = NULL;
}

void bstFree(node* root) {}

CG_UINT bstSearch(node* leaf, CG_UINT key)
{
  if (leaf) {
    if (key == leaf->key) {
      return leaf->value;
    }
    if (key < leaf->key) {
      bstSearch(leaf->left, key);
    } else {
      bstSearch(leaf->right, key);
    }
  }

  return -1; // default/error case
}

void bstInsert(node* leaf, CG_UINT key, CG_UINT value)
{
  if (leaf == NULL) {
    bstNew(&leaf, key, value);
  }

  if (key < leaf->key) {
    if (leaf->left != NULL) {
      bstInsert(leaf->left, key, value);
    } else {
      leaf->left        = malloc(sizeof(node));
      leaf->left->key   = key;
      leaf->left->value = value;
      leaf->left->left  = NULL;
      leaf->left->right = NULL;
    }
  } else if (key > leaf->key) {
    if (leaf->right != NULL) {
      bstInsert(leaf->right, key, value);
    } else {
      leaf->right        = malloc(sizeof(node));
      leaf->right->key   = key;
      leaf->right->value = value;
      leaf->right->left  = NULL;
      leaf->right->right = NULL;
    }
  } else {
    fprintf(stderr, "No duplicates permitted! Omitting...\n");
  }
}
