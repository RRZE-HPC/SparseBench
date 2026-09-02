# ChebFD: resident matrix, streamed search space — design

Written 2026-09-02. Inverts the previous `gpu_stream_mb` design (matrix
streamed, search-space blocks device-resident), which the size sweep showed
is limited by the dense blocks, not the matrix: at NS 64 the blocks cost
2.5 KB/row on the device while the streamed 7-point matrix costs 84 B/row
on the host. Ceiling then: 336^3 = 37.9 M rows on GH200 (96 GB).

## Status: implemented (same day) — results

Code: `src/cuda_vector_stream.{h,cu}` (pipelines + dense kernels),
`gpu_launch_chebfd` / `gpu_launch_spmmv` in `cuda_spmv_{scs,crs}.cu`,
`chebOrthoCholQR2` and the GPU branches in `chebFDSolver.c`,
`allocateHost` in `allocate.c`, `cuda_matrix_stream.*` and `gpu_stream_mb`
removed, `gpu_alloc` forced to managed for ChebFD in `main.c`. Tests:
`tests/solver/chebFDStreamTests.c` (filter / A*Y / Gram / residual parity
against the resident kernels, block update, Cholesky-QR2 rank revealing,
end-to-end solve) — all pass, plus the existing ChebFD unit tests.

Measured on GH200 (NS 64, Np 150, 2 iterations, `results/sweep/`):

| nx  | rows   | nb | s / iteration | filter | ortho | RR   | resid | peak device |
|-----|--------|----|---------------|--------|-------|------|-------|-------------|
| 256 | 16.8 M | 16 | **3.77** (was 5.9) | 3.30 | 0.17 (was 2.24) | 0.13 | 0.18 | 10.6 GB (was 41) |
| 384 | 56.6 M | 16 | 13.2  | 11.7 | 0.54 | 0.42 | 0.58 | 32.7 GB |
| 512 | 134 M  | 16 | 31.4  | 27.8 | 1.26 | 0.97 | 1.38 | 75.8 GB |
| 544 | 161 M  | 16 | 37.6  | 33.4 | 1.46 | 1.14 | 1.66 | 90.7 GB |
| 576 | 191 M  | 16 | 237.6 — **thrashing** (managed matrix evicted) | | | | | 95.0 GB |
| 576 | 191 M  | 8  | 47.8  | 42.3 | 1.73 | 1.89 | 1.96 | 61.8 GB |
| **640** | **262 M** | 8 | **65.6** | 58.0 | 2.36 | 2.60 | 2.69 | 84.3 GB |
| 672 | 303 M  | 8  | 397 — thrashing | | | | | 95.0 GB |
| 704 | 349 M  | 8  | host-side segfault: 2.4 G nnz exceeds the 32-bit index build (`UINT_TYPE=ULL` needed) | | | | | |

- New ceiling on this node: **640^3 = 262 M rows (1.8 G nnz), 6.9x the
  previous 37.9 M**, at 0.25 us/row/iteration — the same per-row cost as
  at 256^3, so streaming adds no overhead with size.
- Device footprint is `matrix (84 B/row) + 4 * nb * 8 B/row + ~0.8 GB`;
  nb 16 is good to ~150 M rows, nb 8 to ~260 M. Beyond that the managed
  matrix is not refused — it oversubscribes and thrashes (6x slower).
  `gpu_vstream_init` now prints a warning when the estimated footprint
  reaches 92 % of device memory.
- The 7-point generator sized its entry array for 27 points in `CG_UINT`
  and overflowed at 544^3; fixed (`matrixGenerate` sizes for the stencil
  in use, in `size_t`).
- nsys at 256^3 (`results/nsys/chebfd_256_vstream.nsys-rep`): GPU idle
  0.3 % of the iteration, 2 `cudaDeviceSynchronize` + 18 blocking copies
  per run (was 655 + 384 for 3 iterations); filter 88 %, residual 4.6 %,
  ortho 4.2 %, RR 3.2 %. `kernel_chebfd_scs` is 91 % of GPU time — the
  filter kernel is now the whole story (backlog item 3).
- The SpMV launch shape adapts its vector tile (32/16/8 lanes) to the
  sub-block width, so nb 8 / 16 run at full-warp efficiency (nb 16 was 30 %
  slower before that change).

## Target layout

| data | where | how |
|---|---|---|
| matrix (`val`, `colInd`, `chunkPtr`, `chunkLens`) | device, always | `cudaMallocManaged` + `cudaMemPrefetchAsync` once at setup (never first-touch) |
| search block `Y` (nr x NS), `AY` (nr x m) | host, pinned (`cudaMallocHost`) | streamed through the device in column sub-blocks / row chunks |
| `u`, `w` (T_{n-2}, T_{n-1}) | **device only, sub-block sized** | never exist on the host: the recurrence starts from X_sub alone |
| device scratch | 2 x { X_sub, U_sub, W_sub } (nr x nb each) + Gram partials | double-buffered |

The `gpu_alloc` knob stops mattering for ChebFD (matrix forced managed,
blocks forced pinned); `gpu_stream_mb` is replaced by a column width knob
`cheb_nb` (already exists: columns per sub-block) — see "Decision".

## Loop order: block-outer filter

Today the filter is degree-outer (one matrix pass per recurrence step over
the whole block). With the vectors on the host that would move 5 blocks
per step. Chebyshev recurrence is column-independent, so go block-outer:

```
for each column sub-block s (nb columns of Y):          # double-buffered
    H2D  X_sub <- Y[:, s]                                 # cudaMemcpy2DAsync, pitch NS*8
    U_sub = T1(X_sub);  W_sub = T2(U_sub, X_sub);  X_sub = gc0 X + gc1 U + gc2 W
    for nn = 3..Np:  chebfdOp(W_sub -> U_sub, X_sub += gc[nn] T_n); swap(U,W)
    D2H  Y[:, s] <- X_sub
```

Traffic per filter: Y in once, Y out once = `2 * 8 * NS * nr` bytes,
independent of Np. At 110 M rows / NS 64: 112 GB, ~0.3 s at 350 GB/s on
the C2C link, against ~19 s of compute (150 passes x ~126 ms). Fully hidden
by double buffering; the link is >95 % idle during the filter.

Kernel side: `kernel_chebfd_scs` / `kernel_spmmv_scs` already take
`numVecs` and `ld` separately, so `ld = nb` on the sub-block buffers needs
no kernel change. The existing `cheb_nb` loop in `launchChebfdPart` becomes
the outer loop of the pipeline instead of an inner tiling.

## The other steps with a host-resident Y

Column-by-column CGS2 (`gpu_orthoMGS`) is impossible here: it makes ~250
full-block passes per iteration, each of which would now cross the link.
Replace by **Cholesky-QR2 with rank revealing**, which is also backlog
item 1 in `ChebFD-Optimizations.md` (the 38 % ortho cost):

1. `G = Y^T Y` — stream **row chunks** of Y (row-major, so contiguous) and
   accumulate with `kernel_gram_partial`; 1 block H2D.
2. Host: symmetric eigendecomposition of the m x m `G` (reuse
   `jacobiEigen`); drop eigenvalues < tol^2 -> rank m'; `R^-1 = V diag(lambda^-1/2)`.
3. `Y <- Y R^-1` — stream row chunks in, apply the m x m' update in shared
   memory, stream out at stride m' (compaction to stride m' falls out);
   1 block in, 1 out.
4. Repeat 1-3 once (second pass, m' fixed) for CGS2-level orthogonality.

~6 block transfers, 4 kernel passes — vs 250 today.

Rayleigh-Ritz and residual, all row-chunk streamed, one pass each:
- `AY_sub = A * Y_sub` by column sub-block (same pipeline as the filter,
  1 step) -> AY on the host: 1 in, 1 out.
- `H = Y^T AY`: row chunks of Y and AY -> `kernel_gram_partial`: 2 in.
- `jacobiEigen(H)` on the host (unchanged).
- Residuals: row chunks of Y and AY with `evec`, `eval` on the device, one
  kernel computing all in-interval pairs at once (backlog item 2): 2 in.

Per-iteration link traffic: filter 2 + ortho 6 + AY 2 + gram 2 + residual 2
= ~14 block transfers = 14 x 8 x NS x nr bytes. At 110 M rows: ~790 GB,
~2.3 s if not overlapped, against ~19 s filter compute — <= 10 % worst
case, less with the row-chunk pipeline double-buffered.

## Size ceiling after the change

Device: matrix 84 B/row + scratch `2 x 3 x nb x 8` B/row + Gram partials.
Host: `Y` + `AY` = `2 x 8 x NS` B/row (u/w no longer on the host).

| nb | device B/row | max rows on 96 GB | nx^3 | host at NS 64 |
|---|---|---|---|---|
| 16 | 852 | ~110 M | ~480 | 110 GB of 573 |
| 8 | 468 | ~200 M | ~585 | 200 GB |
| 4 | 276 | ~340 M | ~700 | 340 GB |

Hard cap from `CG_UINT` = `unsigned int`: nnz < 2^32 -> nr < ~613 M for
the 7-point matrix (nx ~ 850); dense-block offsets are already `size_t`.
Realistic new ceiling on this node: **nx ~ 600 (~200 M rows), ~5x today's
37.9 M**, with nb 8; nb 16 is the better default when it fits (larger
kernels, 128 B segments for the strided copies).

Kernel time per pass scales linearly (0.36 us/row/iteration measured), so
a 200 M-row iteration is ~70 s; a 2-iteration run is a few minutes.

## Layout choice for the copies

Keep Y row-major with stride NS on the host (unchanged everywhere else,
`m` shrink stays a stride change) and pull column sub-blocks with
`cudaMemcpy2DAsync` (pitch `NS*8`, width `nb*8`, height nr). With nb >= 16
each segment is 128 B — fine for the copy engines. If a measurement shows
the 2D DMA under-performing, fall back to a pack/unpack kernel over a
row-chunk staging buffer; the pipeline structure does not change.

## Code plan

1. `cuda_vector_stream.cu/.h` (new): `GpuVectorStream` — pinned host
   block, nb, 2 x {X,U,W} device buffers, copy/compute streams, events.
   Entry points: `gpu_vstream_filter(A, f, Y, NS)` (block-outer filter),
   `gpu_vstream_spmmv(A, Y, AY)`, `gpu_vstream_rowchunk_map(...)` (generic
   row-chunk pipeline used by gram / update / residual).
2. `allocate.c` / `cuda_vector_ops.cu`: `allocateMatrix()` -> always
   managed; `allocatePinned()` for the blocks; `gpu_matrix_prefetch(A)`
   called from `solveChebFD` setup (already exists).
3. `chebFDSolver.c`: `applyFilter` -> `gpu_vstream_filter` on GPU builds;
   `orthoMGS` -> `gpu_choleskyQR2`; `rayleighRitz` / residual -> row-chunk
   versions. CPU build unchanged. Remove `g_chebStream` / `gpu_stream_mb`
   from ChebFD (see Decision) and the `SCS_MAX_SPMMVM_VLA_BYTES` guard
   stays CPU-only.
4. `parameter.c`: `cheb_nb` gains the meaning "streamed sub-block width"
   (default 16); `gpu_stream_mb` prints "ignored by ChebFD" or is removed.
5. Tests: `chebFDStreamTests` -> cover the vector pipeline on the tiny
   matrices with nb < NS (forces multi-sub-block, multi-row-chunk); compare
   eigenvalues to 1e-10 against the CPU build (Cholesky-QR changes the
   arithmetic, residual digits will differ).
6. Measure: 128^3 vs today's managed-resident 2.40 ms/pass (must match:
   same kernel, matrix in HBM), then the sweep up to nx ~ 600.

## Decision needed

Keep the existing matrix-streaming mode (`gpu_stream_mb`) as an alternative
alongside the new vector-streaming mode, or replace it? Replacing is
simpler (one pipeline, `cuda_matrix_stream.cu` goes away, `gpu_alloc` is
irrelevant to ChebFD); keeping it means two mutually exclusive stream
modes and the `gpu_alloc explicit` requirement stays around for one of
them. Given the sweep result (matrix streaming buys only ~3 % more rows on
this node for a 7-point matrix) the recommendation is to replace it.
