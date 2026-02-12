/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "bstree.h"
#include "util.h"

Bstree *bstNew(void)
{
  Bstree *b;

  b       = (Bstree *)malloc(sizeof(Bstree));
  b->root = NULL;
  return b;
}

static void recursive_free_bstree(Node *bn)
{
  if (bn == NULL)
    return;
  if (bn->left != NULL)
    recursive_free_bstree(bn->left);
  if (bn->right != NULL)
    recursive_free_bstree(bn->right);
  free(bn);
  return;
}

void bstFree(Bstree *b)
{
  recursive_free_bstree(b->root);
  free(b);
  return;
}

bool bstExists(Bstree *b, CG_UINT key)
{
  Node *tmp;

  tmp = b->root;
  while (1) {
    if (tmp == NULL)
      return false;
    if (key < tmp->key) {
      tmp = tmp->left;
    } else if (key > tmp->key) {
      tmp = tmp->right;
    } else {
      return true;
    }
  }
}

CG_UINT bstFind(Bstree *b, CG_UINT key)
{
  Node *tmp;

  tmp = b->root;
  while (1) {
    if (tmp == NULL)
      return 0;
    if (key < tmp->key) {
      tmp = tmp->left;
    } else if (key > tmp->key) {
      tmp = tmp->right;
    } else {
      return tmp->value;
    }
  }
}

static void rWalk(Node *leaf)
{
  if (leaf) {
    rWalk(leaf->left);
    printf("%d\n", leaf->value);
    rWalk(leaf->right);
  }
}

void bstWalk(Bstree *b)
{
  rWalk(b->root);
}

void bstInsert(Bstree *b, CG_UINT key, CG_UINT value)
{
  Node *bs, *tmp;

  /* First, create the new node */
  bs        = (Node *)malloc(sizeof(Node));
  bs->left  = NULL;
  bs->right = NULL;
  bs->key   = key;
  bs->value = value;

  /* If it is the first node, put it there */
  if (b->root == NULL) {
    b->root = bs;
    return;
  }

  tmp = b->root;

  while (1) {
    if (key < tmp->key) {
      if (tmp->left == NULL) {
        tmp->left = bs;
        return;
      } else {
        tmp = tmp->left;
      }
    } else if (key > tmp->key) {
      if (tmp->right == NULL) {
        tmp->right = bs;
        return;
      } else {
        tmp = tmp->right;
      }
    } else {
      fprintf(stderr, "No duplicates permitted! Omitting...\n");
    }
  }
}
