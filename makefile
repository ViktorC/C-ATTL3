MAKE := make -f makefile
GCC_CC := g++
CLANG_CC := clang++
CFLAGS := -std=c++11 -fmessage-length=0 -ftemplate-backtrace-limit=0
# AVX instructions are problematic with GCC 64 bit on Windows due to its lack of support for 32 byte stack alignment.
GCC_CFLAGS := $(CFLAGS) -Wno-ignored-attributes -fopenmp
GCC_CUDA_CFLAGS := $(GCC_CFLAGS) -DCATTLE_USE_CUBLAS
# Clang does not support OpenMP on Windows; no libomp or libiomp5.
CLANG_CFLAGS := $(CFLAGS) -march=native -fopenmp=libgomp
CLANG_CUDA_CFLAGS := $(CLANG_CFLAGS) -DCATTLE_USE_CUBLAS
RELEASE_OPT_FLAGS := -O3 -DNDEBUG
DEBUG_OPT_FLAGS := -O1 -Wa,-mbig-obj -g
# For Clang on Windows, omp.h must be copied from GCC.
INCLUDES := -Iexternal/Eigen/ -Iinclude/ -Itest/
CUDA_INCLUDES := -I"$(CUDA_INC_PATH)" $(INCLUDES)
LIBS := -lpthread -lgomp
CUDA_LIBS := $(LIBS) -L"$(CUDA_LIB_PATH)" -lcudart -lcublas
SOURCE_DIR := test
SOURCES := test.cpp
BUILD_DIR := build
TARGET_DIR := bin
TARGET_NAME := cattle_test.exe
TARGET := $(TARGET_DIR)/$(TARGET_NAME)
OBJECTS := $(BUILD_DIR)/$(SOURCES:%.cpp=%.o)
$(OBJECTS): $(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPT_FLAGS) $(INCLUDES) -c -o $@ $<
$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(OPT_FLAGS) -o $@ $? $(LIBS)
.PHONY: all clean
.DEFAULT_GOAL: all
all:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		CFLAGS='$(GCC_CFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
debug:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		CFLAGS='$(GCC_CFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
clang_all:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		CFLAGS='$(CLANG_CFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
clang_debug:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		CFLAGS='$(CLANG_CFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
cuda_all:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		CFLAGS='$(GCC_CUDA_CFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'
cuda_debug:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		CFLAGS='$(GCC_CUDA_CFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'
clang_cuda_all:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		CFLAGS='$(CLANG_CUDA_CFLAGS)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'
clang_cuda_debug:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		CFLAGS='$(CLANG_CUDA_CFLAGS)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)'
clean:
	$(RM) $(OBJECTS) $(TARGET)
.depend:
	$(CC) -MM $(CFLAGS) $(SOURCES) > $@
include .depend