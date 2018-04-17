MAKE := make -f makefile
GCC_CC := g++
CLANG_CC := clang++
# AVX instructions are problematic with GCC 64 bit on Windows due to its lack of support for 32 byte stack alignment.
CFLAGS := -std=c++11 -fopenmp -fmessage-length=0 -ftemplate-backtrace-limit=0 -march=native
RELEASE_OPT_FLAGS := -O3 -DNDEBUG
DEBUG_OPT_FLAGS := -O1 -Wa,-mbig-obj -g
INCLUDES := -Iext/Eigen/ -Isrc/ -Isrc/utils -Itest/
LIBS := -lpthread -lgomp
SOURCE_DIR := test
SOURCES := test.cpp
BUILD_DIR := build
CUDA_VERSION := 9.1
ifeq ($(OS),Windows_NT)
	TARGET_NAME := cattle_test.exe
	CUDA_TOOLKIT_PATH := C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$(CUDA_VERSION)
	CUDA_INCLUDE_PATH := "$(CUDA_TOOLKIT_PATH)\include"
	CUDA_LIB_PATH = "$(CUDA_TOOLKIT_PATH)\lib\x64"
else
	TARGET_NAME := cattle_test
	CUDA_TOOLKIT_PATH := /usr/local/cuda-$(CUDA_VERSION)
	CUDA_INCLUDE_PATH := $(CUDA_TOOLKIT_PATH)/include
	CUDA_LIB_PATH = $(CUDA_TOOLKIT_PATH)/lib64
endif
CUDA_TOOLKIT_PATH := C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1
CUDA_INCLUDES := $(INCLUDES) -I$(CUDA_INCLUDE_PATH)
CUDA_LIBS := $(LIBS) -L$(CUDA_LIB_PATH) -lcudart -lcublas
CUDA_RELEASE_OPT_FLAGS := $(RELEASE_OPT_FLAGS) -DCATTLE_USE_CUBLAS
CUDA_DEBUG_OPT_FLAGS := $(DEBUG_OPT_FLAGS) -DCATTLE_USE_CUBLAS
TARGET := $(BUILD_DIR)/$(TARGET_NAME)
OBJECTS := $(BUILD_DIR)/$(SOURCES:%.cpp=%.o)
$(OBJECTS): $(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPT_FLAGS) $(INCLUDES) -c -o $@ $<
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OPT_FLAGS) -o $@ $? $(LIBS)
.PHONY: all clean
.DEFAULT_GOAL: all
all:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
debug:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
clang_all:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		OPT_FLAGS='$(RELEASE_OPT_FLAGS)'
clang_debug:
	$(MAKE) $(TARGET) \
		CC='$(CLANG_CC)' \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
cuda_all:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)' \
		OPT_FLAGS='$(CUDA_RELEASE_OPT_FLAGS)'
cuda_debug:
	$(MAKE) $(TARGET) \
		CC='$(GCC_CC)' \
		INCLUDES='$(CUDA_INCLUDES)' \
		LIBS='$(CUDA_LIBS)' \
		OPT_FLAGS='$(CUDA_DEBUG_OPT_FLAGS)'
clean:
	$(RM) $(OBJECTS) $(TARGET)
.depend:
	$(CC) -MM $(CFLAGS) $(SOURCES) > $@
include .depend