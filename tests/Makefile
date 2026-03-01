# # # DL 2025.04.04
# # # Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk

BUILD_DIR=../build
TC_DIR=${BUILD_DIR}/$(MTX_FMT)-$(TOOLCHAIN)

# List of test modules
MOD1=matrix
MOD2=solver

# Link against all SparseBench object files
LINKS = $(shell ls ${TC_DIR}/*.o 2>/dev/null | grep -v main.o)

# Target executable
TARGET=runTests

# Collect objects from all test modules
MOD1_OBJECTS=${MOD1}/convertSCS.o ${MOD1}/matrixTests.o
MOD2_OBJECTS=${MOD2}/spmvSCS.o ${MOD2}/spmmvSCS.o ${MOD2}/solverTestsSPMV.o ${MOD2}/solverTestsSPMMV.o
OBJECTS := $(shell echo $(MOD1_OBJECTS) $(MOD2_OBJECTS) | tr ' ' '\n' | sort -u | tr '\n' ' ')

# Always leave debugging flag on
CFLAGS=-g -O0 $(OPENMP) $(DEFINES)

.PHONY: all clean prebuild

all: $(TARGET)

prebuild:
	@make -sC ../

clean:
	@make -sC ../ clean
	$(info ===>  CLEAN TESTS)
	@rm -f $(TARGET) $(OBJECTS) runTests.o

# Link the object files to create the executable
$(TARGET): prebuild runTests.o $(OBJECTS) ${LINKS}
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
$(MOD2)/solverTestsSPMV.o: $(MOD2)/solverTestsSPMV.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/solverTestsSPMMV.o: $(MOD2)/solverTestsSPMMV.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/spmvSCS.o: $(MOD2)/spmvSCS.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MOD2)/spmmvSCS.o: $(MOD2)/spmmvSCS.c 
	$(CC) $(CFLAGS) -c -o $@ $<

format:
	@echo "Formatting source files..."
	@find .. -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cc' -o -name '*.hh' -o -name '*.cxx' -o -name '*.hxx' \) -print0 | xargs -0 -r clang-format -i
	@echo "Done"