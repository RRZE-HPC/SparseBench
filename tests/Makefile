# # # DL 2025.04.04
# # # Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk

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

# Collect objects from all test modules
# Module 1 (matrix) test objects: only include when MTX_FMT is not CRS
ifeq ($(MTX_FMT),CRS)
MOD1_OBJECTS=
else
MOD1_OBJECTS=${MOD1}/convertSCS.o ${MOD1}/matrixTests.o
endif
MOD2_OBJECTS=${MOD2}/spmvSCS.o ${MOD2}/spmmvSCS.o ${MOD2}/solverTestsSPMV.o ${MOD2}/solverTestsSPMMV.o ${MOD2}/chebFDGershgorinTests.o ${MOD2}/chebFDUnitTests.o
OBJECTS := $(shell echo $(MOD1_OBJECTS) $(MOD2_OBJECTS) | tr ' ' '\n' | sort -u | tr '\n' ' ')

# Always leave debugging flag on
CFLAGS=-g -O0 $(OPENMP) $(DEFINES)

# Expected-data generation (Python; requires scipy + numpy)
PYTHON  ?= python3
GEN_DIR  = test_case_generator
MATRICES = $(wildcard data/testMatrices/*.mtx)

.PHONY: all clean prebuild run prerun

all: $(TARGET)

# Build (if needed) and run the test suite.
run: all
	@./$(TARGET) || { rc=$$? ; \
		[ $$rc -le 128 ] || echo "===>  runTests KILLED BY SIGNAL $$((rc - 128)) (rc=$$rc)" ; \
		exit $$rc ; }

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

# Compile runTests.c into an object file
runTests.o: runTests.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Module 1 tests: matrix
$(MOD1)/matrixTests.o: $(MOD1)/matrixTests.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD1)/convertSCS.o: $(MOD1)/convertSCS.c 
	$(CC) $(CFLAGS) -c -o $@ $<

# Module 2 tests: solver
$(MOD2)/solverTestsSPMV.o: $(MOD2)/solverTestsSPMV.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/solverTestsSPMMV.o: $(MOD2)/solverTestsSPMMV.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/spmvSCS.o: $(MOD2)/spmvSCS.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/spmmvSCS.o: $(MOD2)/spmmvSCS.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/chebFDGershgorinTests.o: $(MOD2)/chebFDGershgorinTests.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/chebFDUnitTests.o: $(MOD2)/chebFDUnitTests.c
	$(CC) $(CFLAGS) -c -o $@ $<

format:
	@make -sC ../ format
	@echo "Formatting test files..."
	@find . -type f \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 -r clang-format -i
	@echo "Done"