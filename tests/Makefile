# # # DL 2025.04.04
# # # Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk

BUILD_DIR=../build
TC_DIR=${BUILD_DIR}/$(TOOLCHAIN)

# List of test modules
MODULE1=matrix

# SparseBench object files to link
LINKS=${TC_DIR}/allocate.o ${TC_DIR}/mmio.o ${TC_DIR}/matrix.o

# Target executable
TARGET=runTests

# Collect objects from all test modules
TEST_OBJECTS=${MODULE1}/convertSCS.o ${MODULE1}/matrixTests.o

# Always leave debugging flag on
CFLAGS=-g

.PHONY: all clean

# All target - create the runTests executable
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): runTests.o $(TEST_OBJECTS) ${LINKS}
	$(CC) $(CFLAGS) -o $@ runTests.o $(TEST_OBJECTS) ${LINKS}

# Compile runTests.c into an object file
runTests.o: runTests.c
	$(CC) $(CFLAGS) -c -o $@ $<

# MODULE1 tests
$(MODULE1)/matrixTests.o: $(MODULE1)/matrixTests.c 
	$(CC) $(CFLAGS) -c -o $@ $<

$(MODULE1)/convertSCS.o: $(MODULE1)/convertSCS.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean target to remove generated files
clean:
	rm -f $(TARGET) $(TEST_OBJECTS) runTests.o