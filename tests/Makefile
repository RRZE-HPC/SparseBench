# # # DL 2025.04.04
# # # Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk
include ../mk/include_SCAMAC.mk

BUILD_DIR=../build
TC_DIR=${BUILD_DIR}/$(MTX_FMT)-$(TOOLCHAIN)
CUDA_DIR=../src/cuda

# List of test modules
MOD1=matrix
MOD2=solver

# Link against all SparseBench object files
LINKS = $(shell ls ${TC_DIR}/*.o 2>/dev/null | grep -v main.o)

# Target executable
TARGET=runTests

# Test sources per module; the object lists follow from them (.c=.o below),
# so adding a suite means adding one .c here and nothing else.
# Module 1 (matrix) tests are the SCS reference kernels: only built when
# the library itself is SCS.
ifeq ($(MTX_FMT),CRS)
MOD1_SOURCES=
else
MOD1_SOURCES=${MOD1}/convertSCS.c ${MOD1}/matrixTests.c
endif
MOD2_SOURCES=${MOD2}/spmvSCS.c ${MOD2}/spmmvSCS.c ${MOD2}/solverTestsSPMV.c ${MOD2}/solverTestsSPMMV.c ${MOD2}/chebFDGershgorinTests.c ${MOD2}/chebFDUnitTests.c ${MOD2}/chebFDStreamTests.c ${MOD2}/sectionTimerTests.c ${MOD2}/vectorOpsTests.c

MOD1_OBJECTS := $(MOD1_SOURCES:.c=.o)
MOD2_OBJECTS := $(MOD2_SOURCES:.c=.o)
OBJECTS := $(sort $(MOD1_OBJECTS) $(MOD2_OBJECTS))

# Always leave debugging flag on
CFLAGS=-g -O0 $(OPENMP) $(DEFINES)

# Expected-data generation (Python; requires scipy + numpy)
PYTHON  ?= python3
GEN_DIR  = test_case_generator
MATRICES = $(wildcard data/testMatrices/*.mtx)

.PHONY: all clean prebuild run prerun list format

all: $(TARGET)

# Build (if needed) and run the whole test suite.
run: all
	@./$(TARGET)

# Build (if needed) and run a single suite, e.g. `make run-chebFDStreamTests`.
# Use `make list` (or `./runTests --list`) to see the available suite names.
run-%: all
	@./$(TARGET) $*

list: all
	@./$(TARGET) --list

# generate the expected test data from the reference matrices
# Output is written to data/expected/ by the script.
prerun:
	@for m in $(MATRICES) ; do \
		echo "Generating expected data for $$m" ; \
		$(PYTHON) $(GEN_DIR)/testfile_generators.py "$$m" || exit 1 ; \
	done

prebuild:
	@make -sC ../

clean:
	@make -sC ../ clean
	$(info ===>  CLEAN TESTS)
	@rm -f $(TARGET) $(OBJECTS) runTests.o

# Link the object files to create the executable
$(TARGET): prebuild runTests.o $(OBJECTS) ${LINKS}
	$(LD) $(LFLAGS) -o $@ runTests.o $(OBJECTS) ${LINKS} $(LIBS)

# Test objects share struct layouts with the library headers (e.g. ChebData
# in chebFDSolver.h): rebuild them whenever any header changes, or a stale
# object silently corrupts the stack.
HEADERS = $(wildcard ../src/*.h) common.h $(wildcard $(MOD1)/*.h) $(wildcard $(MOD2)/*.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

format:
	@make -sC ../ format
	@echo "Formatting test files..."
	@find . -type f \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 -r clang-format -i
	@echo "Done"