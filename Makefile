MAKE := make -f Makefile
GCC_CXX := g++
CLANG_CXX := clang++
COV := gcov
CXXFLAGS := -std=c++11 -fmessage-length=0 -ftemplate-backtrace-limit=0
# AVX instructions are problematic with GCC 64 bit on Windows due to its lack of support for 32 byte stack alignment.
GCC_CXXFLAGS := $(CXXFLAGS) -Wno-ignored-attributes -fopenmp
# Clang does not actually utilize OpenMP on Windows; no libomp or libiomp5.
CLANG_CXXFLAGS := $(CXXFLAGS) -march=native
RELEASE_OPT_FLAGS := -O3 -DNDEBUG
# Without level 1 optimization, the object file is too big.
DEBUG_OPT_FLAGS := -O1 -g
# Support gcov/lcov.
COVERAGE_OPT_FLAGS := $(DEBUG_OPT_FLAGS) -fprofile-arcs -ftest-coverage
GTEST_DIR := test/gtest
# For Clang on Windows, omp.h must be copied from GCC.
INCLUDES := -IC-ATTL3 -IEigen -I$(GTEST_DIR)/include -Itest/
LIBS := -lpthread -lgomp
HEADERS := $(shell find C-ATTL3 -type f -name '*.hpp' -printf "%f\n")
SOURCE_DIR := test
SOURCES := gradient_test.cpp training_test.cpp main_test.cpp
BUILD_DIR := build
COV_DIR := $(BUILD_DIR)/cov
TARGET_DIR := bin
TARGET_NAME := cattle_test.exe
GTEST_MAKE_PATH := $(GTEST_DIR)/make
TARGET := $(TARGET_DIR)/$(TARGET_NAME)
OBJECTS := $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
$(OBJECTS): $(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@cd $(GTEST_MAKE_PATH) && make gtest-all.o && cd $(CURDIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(INCLUDES) -c -o $@ $<
$(TARGET): $(OBJECTS)
	@cd $(GTEST_MAKE_PATH) && make gtest.a && cd $(CURDIR)
	@mkdir -p $(TARGET_DIR)
	# Link the gtest static library directly as it is built by default without the 'lib' prefix.
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o $@ $? $(GTEST_MAKE_PATH)/gtest.a $(LIBS)
.PHONY: all clean
.DEFAULT_GOAL: all
all:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
debug:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
coverage:
	$(MAKE) $(TARGET) \
		CXX='$(GCC_CXX)' \
		CXXFLAGS='$(GCC_CXXFLAGS)' \
		OPT_FLAGS='$(COVERAGE_OPT_FLAGS)'
clang_all:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CXXFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
clang_debug:
	$(MAKE) $(TARGET) \
		CXX='$(CLANG_CXX)' \
		CXXFLAGS='$(CLANG_CXXFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
check:
	@bin/cattle_test.exe
report:
	$(COV) -o $(BUILD_DIR) -r $(SOURCES)
		@rm -rf $(COV_DIR)
		@mkdir $(COV_DIR) && (mv -f -t $(COV_DIR) $(HEADERS:%=%.gcov) || true)
		@rm -f *.gcov
clean:
	@rm -rf $(BUILD_DIR) $(TARGET_DIR)
		@cd $(GTEST_MAKE_PATH) && make clean && cd $(CURDIR)
.depend:
	$(CC) -MM $(CFLAGS) $(SOURCES) > $@
include .depend