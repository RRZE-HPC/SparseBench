/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * CUDA SpMV kernel for the Sell-C-sigma (SCS) matrix format.
 * One thread block per chunk — maps naturally to the SCS layout.
 *
 * To add kernels for other formats (CRS, CCRS, …), create a new
 * .cu file in src/cuda/  — the build system picks it up automatically.
 */
#include "gpu_backend.h"
#include "cuda_kernels.h"

/* ------------------------------------------------------------------ */
/*  SCS SpMV:  y = A * x                                             */
/*                                                                    */
/*  Each thread block handles one chunk of C rows.                    */
/*  threadIdx.x corresponds to the row within the chunk.              */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmv_scs(CG_UINT nChunks, CG_UINT C,
    const CG_UINT *chunkPtr, const CG_UINT *chunkLens,
    const CG_UINT *colInd, const CG_FLOAT *val,
    const CG_FLOAT *x, CG_FLOAT *y)
{
    CG_UINT chunk = blockIdx.x;
    CG_UINT lane  = threadIdx.x;

    if (chunk >= nChunks || lane >= C) return;

    CG_UINT offset = chunkPtr[chunk];
    CG_UINT len    = chunkLens[chunk];

    CG_FLOAT tmp = 0.0;
    for (CG_UINT j = 0; j < len; j++) {
        CG_UINT idx = offset + j * C + lane;
        tmp += val[idx] * x[colInd[idx]];
    }

    y[chunk * C + lane] = tmp;
}

extern "C"
void gpu_spmv_scs(CG_UINT nChunks, CG_UINT C,
    const CG_UINT *chunkPtr, const CG_UINT *chunkLens,
    const CG_UINT *colInd, const CG_FLOAT *val,
    const CG_FLOAT *x, CG_FLOAT *y)
{
    /* One block per chunk, C threads per block */
    dim3 grid(nChunks);
    dim3 block(C);
    kernel_spmv_scs<<<grid, block>>>(nChunks, C, chunkPtr, chunkLens,
        colInd, val, x, y);
}

/* ------------------------------------------------------------------ */
/*  SCS SpMMV:  Y = A * X   (block-vector, numVecs columns)          */
/*                                                                    */
/*  Each thread handles one (row, vec) pair within a chunk.           */
/* ------------------------------------------------------------------ */
__global__ void kernel_spmmv_scs(CG_UINT nChunks, CG_UINT C,
    CG_UINT numVecs,
    const CG_UINT *chunkPtr, const CG_UINT *chunkLens,
    const CG_UINT *colInd, const CG_FLOAT *val,
    const CG_FLOAT *x, CG_FLOAT *y)
{
    CG_UINT chunk = blockIdx.x;
    CG_UINT lane  = threadIdx.x;   /* row within chunk */
    CG_UINT vec   = threadIdx.y;   /* vector index */

    if (chunk >= nChunks || lane >= C) return;

    CG_UINT offset = chunkPtr[chunk];
    CG_UINT len    = chunkLens[chunk];

    CG_FLOAT tmp = 0.0;
    for (CG_UINT j = 0; j < len; j++) {
        CG_UINT idx = offset + j * C + lane;
        CG_UINT col = colInd[idx];
        tmp += val[idx] * x[col * numVecs + vec];
    }

    y[(chunk * C + lane) * numVecs + vec] = tmp;
}

/* ------------------------------------------------------------------ */
/*  High-level synchronous wrappers matching the CPU solver.h API     */
/*  Expects managed-memory pointers (allocated via gpu_allocate_managed) */
/* ------------------------------------------------------------------ */

extern "C"
void gpu_spMVM(Matrix *m, const CG_FLOAT *x, CG_FLOAT *y)
{
    dim3 grid(m->nChunks);
    dim3 block(m->C);
    kernel_spmv_scs<<<grid, block>>>(m->nChunks, m->C,
        m->chunkPtr, m->chunkLens, m->colInd, m->val, x, y);
    GPU_SAFE_CALL(gpuDeviceSynchronize());
}

extern "C"
void gpu_spMMVM(Matrix *m, const DMatrix *x, DMatrix *y)
{
    CG_UINT numVecs = x->nc;
    dim3 grid(m->nChunks);
    dim3 block(m->C, numVecs);
    kernel_spmmv_scs<<<grid, block>>>(m->nChunks, m->C, numVecs,
        m->chunkPtr, m->chunkLens, m->colInd, m->val,
        x->entries, y->entries);
    GPU_SAFE_CALL(gpuDeviceSynchronize());
}
