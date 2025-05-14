# SparseBench

A hybrid MPI+OpenMP parallel sparse solver benchmark collection.

## Build

It is recommended to use GNU Make 4.0 or newer. While older make versions will
work, the generation of the `.clangd` configuration for the clang language
server will not work. The default Make version included in MacOS is 3.81! Newer make
versions can be easily installed on MacOS using the
[Homebrew](https://brew.sh/) package manager.

1. Configure the tool-chain and additional options in `config.mk`:

```make
# Supported: GCC, CLANG, ICC
TOOLCHAIN ?= GCC
ENABLE_MPI ?= false
ENABLE_OPENMP ?= false

OPTIONS +=  -DARRAY_ALIGNMENT=64
OPTIONS +=  -DOMP_SCHEDULE=static
#OPTIONS +=  -DDEBUG
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
```

The verbosity options enable detailed output about affinity settings, allocation
sizes and timer resolution.

1. Build with:

```sh
make
```

You can build multiple tool-chains in the same directory, but notice that the
Makefile is only acting on the one currently set. Intermediate build results are
located in the `./build/<TOOLCHAIN>` directory.

To show all executed commands use:

```sh
make Q=
```

1. Clean up intermediate build results for current tool-chain with:

```sh
make clean
```

Clean up all build results for all tool-chains with:

```sh
make distclean
```

1. Optional targets:

Generate assembler:

```sh
make asm
```

The assembler files will also be located in the `./build/<TOOLCHAIN>` directory.
Reformat all source files using `clang-format`. This only works if
`clang-format` is in your path.

```sh
make format
```
