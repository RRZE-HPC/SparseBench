# SparseBench

A hybrid MPI+OpenMP parallel sparse solver benchmark collection for evaluating
the performance of iterative sparse linear solvers and sparse matrix-vector
operations.

## Overview

SparseBench provides a comprehensive benchmarking suite for sparse matrix
computations with support for multiple matrix formats and iterative solvers.
It is designed to measure performance on both single-node and distributed
memory systems.

### Supported Sparse Matrix Formats

SparseBench supports three different sparse matrix storage formats, each with
different performance characteristics:

- **CRS (Compressed Row Storage)**: Standard row-compressed format with three
  arrays (row pointers, column indices, values). Good general-purpose format.
- **SCS (Sell-C-Sigma)**: SELL-C-σ format optimized for vectorization and
  cache efficiency. Particularly effective on modern CPUs with SIMD instructions.
- **CCRS (Compressed CRS)**: Compressed variant of CRS format for improved
  memory efficiency.

### Benchmark Types

The following benchmark types are available:

- **CG**: Conjugate Gradient iterative solver for symmetric positive definite
  systems.
- **SPMV**: Sparse Matrix-Vector Multiplication (SpMV) kernel benchmark.
- **GMRES**: Generalized Minimal Residual method for general sparse systems
  (planned).
- **CHEBFD**: Chebyshev Filter Diagonalization (planned).

### MPI Communication Algorithm

For distributed memory execution with MPI, SparseBench uses an optimized
communication algorithm that reduces communication to a single exchange
operation per iteration. See [MPI-Algorithm.md](file:///Users/jan/prg/SparseBench/MPI-Algorithm.md)
for a detailed explanation of the localization process and communication
strategy.

## Getting Started

To build and run **SparseBench**, you need a C compiler and GNU make. MPI is
optional but required for distributed memory benchmarks.

1. **Install a supported compiler**
   - GCC
   - Clang
   - Intel ICX

2. **Clone the repository**

   ```sh
   git clone <repository-url>
   cd SparseBench
   ```

3. **(Optional) Adjust configuration**

   On first run, `make` will copy `mk/config-default.mk` to `config.mk`. Edit
   `config.mk` to change the toolchain, matrix format, enable MPI/OpenMP, etc.

4. **Build**

   ```sh
   make
   ```

   See the full [Build](#build) section for more details.

5. **Usage**

   ```sh
   ./sparseBench-<TOOLCHAIN> [options]
   ```

   See the full [Usage](#usage) section for more details.

   Get help on command-line arguments:

   ```sh
   ./sparseBench-<TOOLCHAIN> -h
   ```

## Build

### Configuration

Configure the toolchain and additional options in `config.mk`:

```make
# Supported: GCC, CLANG, ICC
TOOLCHAIN ?= CLANG
# Supported: CRS, SCS, CCRS
MTX_FMT ?= CRS
ENABLE_MPI ?= true
ENABLE_OPENMP ?= false
FLOAT_TYPE ?= DP  # SP for float, DP for double
UINT_TYPE ?= U    # U for unsigned int, ULL for unsigned long long int

# Feature options
OPTIONS +=  -DARRAY_ALIGNMENT=64
OPTIONS +=  -DOMP_SCHEDULE=static
#OPTIONS +=  -DVERBOSE
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
```

#### Configuration Options

- **TOOLCHAIN**: Compiler to use (GCC, CLANG, ICC)
- **MTX_FMT**: Sparse matrix storage format (CRS, SCS, CCRS)
- **ENABLE_MPI**: Enable MPI for distributed memory execution
- **ENABLE_OPENMP**: Enable OpenMP for shared memory parallelism
- **FLOAT_TYPE**:
  - `SP`: Single precision (float)
  - `DP`: Double precision (double)
- **UINT_TYPE**:
  - `U`: Unsigned int for matrix indices
  - `ULL`: Unsigned long long int for very large matrices

#### Verbosity Options

The verbosity options enable detailed diagnostic output:

- `-DVERBOSE`: General verbose output
- `-DVERBOSE_AFFINITY`: Print thread affinity settings and processor bindings
- `-DVERBOSE_DATASIZE`: Print detailed memory allocation sizes
- `-DVERBOSE_TIMER`: Print timer resolution information

### Build Commands

Build with:

```sh
make
```

You can build multiple toolchains in the same directory, but the Makefile
only acts on the currently configured one. Intermediate build results are
located in the `./build/<TOOLCHAIN>` directory.

To show all executed commands use:

```sh
make Q=
```

Clean up intermediate build results for the current toolchain with:

```sh
make clean
```

Clean up all build results for all toolchains with:

```sh
make distclean
```

### Optional Targets

Generate assembler files:

```sh
make asm
```

The assembler files will be located in the `./build/<TOOLCHAIN>` directory.

Reformat all source files using `clang-format` (only works if `clang-format`
is in your path):

```sh
make format
```

## Usage

To run the benchmark, call:

```sh
./sparseBench-<TOOLCHAIN> [options]
```

### Command Line Arguments

| Option | Argument           | Description                                                                       |
| ------ | ------------------ | --------------------------------------------------------------------------------- |
| `-h`   | —                  | Show help text.                                                                   |
| `-f`   | `<parameter file>` | Load options from a parameter file.                                               |
| `-m`   | `<MM matrix>`      | Load a Matrix Market (.mtx) file.                                                 |
| `-c`   | `<file name>`      | Convert a Matrix Market file to binary matrix format (.bmx).                      |
| `-t`   | `<bench type>`     | Benchmark type: `cg`, `spmv`, or `gmres`. Default: `cg`.                          |
| `-x`   | `<int>`            | Size in x dimension for generated matrix (ignored if loading file). Default: 100. |
| `-y`   | `<int>`            | Size in y dimension for generated matrix (ignored if loading file). Default: 100. |
| `-z`   | `<int>`            | Size in z dimension for generated matrix (ignored if loading file). Default: 100. |
| `-i`   | `<int>`            | Number of solver iterations. Default: 150.                                        |
| `-e`   | `<float>`          | Convergence criteria epsilon. Default: 0.0.                                       |

### Matrix Input

SparseBench supports multiple ways to provide input matrices:

1. **Generated matrices**: Use default or specify dimensions with `-x`, `-y`, `-z`:
   - Default mode generates a 3D 7-point stencil matrix
   - Use `-m generate7P` for explicit 7-point stencil generation

2. **Matrix Market files** (`.mtx`): Load standard MatrixMarket format:

   ```sh
   ./sparseBench-GCC -m matrix.mtx
   ```

3. **Binary matrix files** (`.bmx`): Load pre-converted binary format (MPI builds only):

   ```sh
   ./sparseBench-GCC -m matrix.bmx
   ```

   Convert Matrix Market to binary format:

   ```sh
   ./sparseBench-GCC -c matrix.mtx
   ```

### Example Usage

Run CG solver with generated 100×100×100 matrix:

```sh
./sparseBench-GCC
```

Run CG solver with custom dimensions:

```sh
./sparseBench-GCC -x 200 -y 200 -z 200
```

Run SpMV benchmark with Matrix Market file:

```sh
./sparseBench-GCC -t spmv -m matrix.mtx -i 1000
```

Run GMRES solver with convergence criteria:

```sh
./sparseBench-GCC -t gmres -m matrix.mtx -e 1e-6
```

Run with MPI (4 processes):

```sh
mpirun -np 4 ./sparseBench-GCC -m matrix.mtx
```

## MPI Algorithm

SparseBench uses an optimized MPI communication algorithm for distributed
sparse matrix computations. The implementation uses a **localization** process
that transforms the distributed matrix to enable efficient communication:

- All column indices are converted to local indices
- External elements are appended to local vectors
- Communication is reduced to a single `MPI_Neighbor_alltoallv` call per iteration
- Elements from the same source rank are stored consecutively

For a detailed explanation of the algorithm, including the four-step
localization process (identify externals, build graph topology, reorder
externals, build global index list), see
[MPI-Algorithm.md](file:///Users/jan/prg/SparseBench/MPI-Algorithm.md).

## Testing

The `tests` directory contains unit tests for various components:

- Matrix format tests
- Solver validation tests
- Communication tests (MPI builds)

To run all tests:

```sh
cd tests
make
./runTests
```

## Thread Pinning

For OpenMP execution, it is recommended to control thread affinity for optimal
performance. We recommend using `likwid-pin` to set the number of threads and
control thread affinity:

```sh
likwid-pin -C 0-7 ./sparseBench-GCC
```

If `likwid-pin` is not available, you can use standard OpenMP environment
variables:

```sh
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./sparseBench-GCC
```

## Support for clangd Language Server

The Makefile will generate a `.clangd` configuration to correctly set all
options for the clang language server. This is only important if you use an
editor with LSP support and want to edit or explore the source code.

It is required to use GNU Make 4.0 or newer. While older make versions will
work, the generation of the `.clangd` configuration for the clang language
server will not work. The default Make version included in MacOS is 3.81!
Newer make versions can be easily installed on MacOS using the
[Homebrew](https://brew.sh/) package manager.

An alternative is to use [Bear](https://github.com/rizsotto/Bear), a tool that
generates a compilation database for clang tooling. This method also enables
jumping to any definition without previously opened buffer. You have to build
SparseBench one time with Bear as a wrapper:

```sh
bear -- make
```

## License

Copyright © NHR@FAU, University Erlangen-Nuremberg.

This project is licensed under the MIT License - see the [LICENSE](file:///Users/jan/prg/SparseBench/LICENSE)
file for details.
