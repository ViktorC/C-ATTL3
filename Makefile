GCC_CXX := g++
CLANG_CXX := clang++
COV := gcov
CXXFLAGS := -std=c++11 -fmessage-length=0 -ftemplate-backtrace-limit=0
# AVX instructions are problematic with GCC 64 bit on Windows due to its lack of support for 32 byte stack alignment.
GCC_CXXFLAGS := $(CXXFLAGS) -Wno-ignored-attributes -fopenmp
GCC_CUDA_CXXFLAGS := $(GCC_CXXFLAGS) -DCATTL3_USE_CUDA
# Clang does not actually utilize OpenMP on Windows; no libomp or libiomp5.
CLANG_CXXFLAGS := $(CXXFLAGS) -march=native
CLANG_CUDA_CXXFLAGS := $(CLANG_CXXFLAGS) -DCATTL3_USE_CUDA
RELEASE_OPT_FLAGS := -O3 -DNDEBUG
# Without level 1 optimization, the object file is too big.
DEBUG_OPT_FLAGS := -O1 -g
# Support gcov/lcov.
COVERAGE_OPT_FLAGS := $(DEBUG_OPT_FLAGS) -fprofile-arcs -ftest-coverage

HEADER_DIR := C-ATTL3
HEADERS := $(shell find $(HEADER_DIR) -type f -name '*.hpp')

SOURCE_DIR := test
SOURCES := $(wildcard $(SOURCE_DIR)/*.cpp)

BUILD_DIR := build
OBJECTS := $(patsubst $(SOURCE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

TARGET_DIR := bin
TARGET_NAME := cattle_test.exe
TARGET := $(TARGET_DIR)/$(TARGET_NAME)

COV_DIR := $(BUILD_DIR)/cov

GTEST_DIR := test/gtest
GTEST_MAKE_PATH := $(GTEST_DIR)/make

# For Clang on Windows, omp.h must be copied from GCC.
INCLUDES := -I$(HEADER_DIR) -IEigen -I$(GTEST_DIR)/include -I$(SOURCE_DIR)
CUDA_INCLUDES := -I"$(CUDA_INC_PATH)" $(INCLUDES)

LIBS := -lpthread
CUDA_LIBS := $(LIBS) -L"$(CUDA_LIB_PATH)" -lcudart -lcurand -lcublas -lcudnn

$(TARGET): $(OBJECTS)
	@cd $(GTEST_MAKE_PATH) && $(MAKE) gtest.a && cd $(CURDIR)
	@mkdir -p $(TARGET_DIR)
	# Link the gtest static library directly as it is built by default without the 'lib' prefix.
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o $@ $? $(GTEST_MAKE_PATH)/gtest.a $(LIBS)

$(OBJECTS): $(SOURCES) $(HEADERS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp 
	@cd $(GTEST_MAKE_PATH) && $(MAKE) gtest-all.o && cd $(CURDIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(INCLUDES) -c -o $@ $<

.DEFAULT_GOAL := all
.PHONY: all
all:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'

.PHONY: debug
debug:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'

.PHONY: coverage
coverage:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(COVERAGE_OPT_FLAGS)'

.PHONY: cuda_all
cuda_all:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CUDA_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'

.PHONY: cuda_debug
cuda_debug:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CUDA_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'

.PHONY: clang_all
clang_all:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'

.PHONY: clang_debug
clang_debug:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'

.PHONY: clang_cuda_all
clang_cuda_all:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CUDA_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'

.PHONY: clang_cuda_debug
clang_cuda_debug:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CUDA_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'

.PHONY: check
check:
	@bin/cattle_test.exe

.PHONY: report
report:
	$(COV) -o $(BUILD_DIR) -r $(SOURCES)
		@rm -rf $(COV_DIR)
		@mkdir $(COV_DIR) && (mv -f -t $(COV_DIR) $(HEADERS:%=%.gcov) || true)
		@rm -f *.gcov

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR) $(TARGET_DIR)
		@cd $(GTEST_MAKE_PATH) && $(MAKE) clean && cd $(CURDIR)

.PHONY: variables
variables:
	@echo "Sources: " $(SOURCES)
	@echo "Headers: " $(HEADERS)
	@echo "Objects: " $(OBJECTS)