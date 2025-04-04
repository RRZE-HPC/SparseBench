# DL 2025.04.04
# Collection of unit tests

include ../config.mk
include ../mk/include_$(TOOLCHAIN).mk
# INCLUDES  += -I$(SRC_DIR)/includes -I$(BUILD_DIR)

BUILD_DIR=../build
TC_DIR=${BUILD_DIR}/$(TOOLCHAIN)
TESTS_1=matrix

.PHONY: all
all: ${TESTS_1}_tests

LINKS=${TC_DIR}/allocate.o ${TC_DIR}/mmio.o

${TESTS_1}_tests:
	$(CC) ${TESTS_1}/convertSCS.c -o $@ ${LINKS} ${TC_DIR}/${TESTS_1}.o

# $(MPICXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)