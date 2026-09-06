/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef NVTX_MARKER_H
#define NVTX_MARKER_H

/* ---- range colors (ARGB; nsys ignores the alpha byte) ---------------- */
#define NVTX_C_SETUP 0x76B900u   /* green    init / alloc / teardown      */
#define NVTX_C_CONVERT 0x8B6914u /* brown    matrix read + format convert */
#define NVTX_C_MATVEC 0xD95F02u  /* orange   SpMV / SpMMV / fused kernels */
#define NVTX_C_FILTER 0xA6761Du  /* copper   ChebFD filter phase          */
#define NVTX_C_ORTHO 0x7570B3u   /* purple   orthogonalization / repack   */
#define NVTX_C_RR 0xE7298Au      /* magenta  Rayleigh-Ritz               */
#define NVTX_C_RESID 0xE6AB02u   /* gold     Ritz residuals              */
#define NVTX_C_VECTOR 0x1B9E77u  /* teal     waxpby / ddot               */
#define NVTX_C_CG 0x3182BDu      /* blue     CG solver                   */
#define NVTX_C_STREAM 0x66A61Eu  /* olive    matrix streaming setup/tear */

/* ---- categories (filterable in nsys / ncu --nvtx-include) ------------ */
typedef enum {
  NVTX_CAT_SETUP   = 1,
  NVTX_CAT_CONVERT = 2,
  NVTX_CAT_MATVEC  = 3,
  NVTX_CAT_FILTER  = 4,
  NVTX_CAT_ORTHO   = 5,
  NVTX_CAT_RR      = 6,
  NVTX_CAT_RESID   = 7,
  NVTX_CAT_VECTOR  = 8,
  NVTX_CAT_CG      = 9,
  NVTX_CAT_STREAM  = 10
} NvtxCategory;

#ifdef USE_NVTX
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <nvtx3/nvToolsExt.h>

#if defined(__GNUC__)
#define NVTX_UNUSED __attribute__((unused))
#else
#define NVTX_UNUSED
#endif

static NVTX_UNUSED int nvtxRangePushFmt(unsigned color, const char *fmt, ...)
{
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  (void)vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  nvtxEventAttributes_t attr;
  memset(&attr, 0, sizeof(attr));
  attr.version       = NVTX_VERSION;
  attr.size          = (unsigned)NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.colorType     = NVTX_COLOR_ARGB;
  attr.color         = color;
  attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = buf;
  return nvtxRangePushEx(&attr);
}

#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)

#define NVTX_RANGE_PUSH_C(name_, color_)                                                 \
  do {                                                                                   \
    nvtxEventAttributes_t nvtx_attr_;                                                    \
    memset(&nvtx_attr_, 0, sizeof(nvtx_attr_));                                          \
    nvtx_attr_.version       = NVTX_VERSION;                                             \
    nvtx_attr_.size          = (unsigned)NVTX_EVENT_ATTRIB_STRUCT_SIZE;                  \
    nvtx_attr_.colorType     = NVTX_COLOR_ARGB;                                          \
    nvtx_attr_.color         = (unsigned)(color_);                                       \
    nvtx_attr_.messageType   = NVTX_MESSAGE_TYPE_ASCII;                                  \
    nvtx_attr_.message.ascii = (name_);                                                  \
    nvtxRangePushEx(&nvtx_attr_);                                                        \
  } while (0)

#define NVTX_RANGE_PUSHF(color_, fmt_, ...)                                              \
  nvtxRangePushFmt((color_), (fmt_), __VA_ARGS__)

#define NVTX_RANGE_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)
#define NVTX_NAME_CATEGORY(cat, name) nvtxNameCategoryA((unsigned)(cat), (name))

#define NVTX_INIT()                                                                      \
  do {                                                                                   \
    NVTX_NAME_CATEGORY(NVTX_CAT_SETUP, "setup");                                         \
    NVTX_NAME_CATEGORY(NVTX_CAT_CONVERT, "convert");                                     \
    NVTX_NAME_CATEGORY(NVTX_CAT_MATVEC, "matvec");                                       \
    NVTX_NAME_CATEGORY(NVTX_CAT_FILTER, "filter");                                       \
    NVTX_NAME_CATEGORY(NVTX_CAT_ORTHO, "ortho");                                         \
    NVTX_NAME_CATEGORY(NVTX_CAT_RR, "rayleigh-ritz");                                    \
    NVTX_NAME_CATEGORY(NVTX_CAT_RESID, "residual");                                      \
    NVTX_NAME_CATEGORY(NVTX_CAT_VECTOR, "vector");                                       \
    NVTX_NAME_CATEGORY(NVTX_CAT_CG, "cg");                                               \
    NVTX_NAME_CATEGORY(NVTX_CAT_STREAM, "streaming");                                    \
  } while (0)

#else /* USE_NVTX */

#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_PUSH_C(name, color) ((void)0)
#define NVTX_RANGE_PUSHF(color, fmt, ...) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#define NVTX_MARK(name) ((void)0)
#define NVTX_NAME_CATEGORY(cat, name) ((void)0)
#define NVTX_INIT() ((void)0)

#endif /* USE_NVTX */
#endif /* NVTX_MARKER_H */
