# # # DL 2025.04.04
# # # Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk

BUILD_DIR=../build
TC_DIR=${BUILD_DIR}/$(TOOLCHAIN)

# List of test modules
MOD1=matrix
MOD2=solver

# Link against all SparseBench object files
LINKS := $(filter-out ${TC_DIR}/main.o, $(wildcard ${TC_DIR}/*.o))

# Target executable
TARGET=runTests

# Collect objects from all test modules
MOD1_OBJECTS=${MOD1}/convertSCS.o ${MOD1}/matrixTests.o
MOD2_OBJECTS=${MOD2}/spmvSCS.o ${MOD2}/solverTests.o
OBJECTS := $(shell echo $(MOD1_OBJECTS) $(MOD2_OBJECTS) | tr ' ' '\n' | sort -u | tr '\n' ' ')

# Always leave debugging flag on
CFLAGS=-g -qopenmp

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) runTests.o

# Link the object files to create the executable
$(TARGET): runTests.o $(OBJECTS) ${LINKS}
	$(CC) $(CFLAGS) -o $@ runTests.o $(OBJECTS) ${LINKS}

# Compile runTests.c into an object file
runTests.o: runTests.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Module 1 tests: matrix
$(MOD1)/matrixTests.o: $(MOD1)/matrixTests.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD1)/convertSCS.o: $(MOD1)/convertSCS.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Module 2 tests: solver
$(MOD2)/solverTests.o: $(MOD2)/solverTests.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/spmvSCS.o: $(MOD2)/spmvSCS.c
	$(CC) $(CFLAGS) -c -o $@ $<