MAKE := make -f makefile
CC := g++
ARCH := -m64
CFLAGS := -std=c++11 -fopenmp -fmessage-length=0 -Wno-ignored-attributes
DEF_OPT_FLAGS := -O3
DEBUG_OPT_FLAGS := -g3 -Wa,-mbig-obj -O1
SOURCE_DIR := test
SOURCES := test.cpp
INCLUDES := -Isrc/ -Itest/ -Iext/Eigen/
LIBS := -lpthread -lgomp
BUILD_DIR := build
ifeq ($(OS),Windows_NT)
	TARGET_NAME := cattle_test.exe
else
	TARGET_NAME := cattle_test
endif
TARGET := $(BUILD_DIR)/$(TARGET_NAME)
OBJECTS := $(BUILD_DIR)/$(SOURCES:%.cpp=%.o)
$(OBJECTS): $(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	$(CC) $(ARCH) $(CFLAGS) $(OPT_FLAGS) $(INCLUDES) -c -o $@ $<
$(TARGET): $(OBJECTS)
	$(CC) $(ARCH) $(CFLAGS) $(OPT_FLAGS) $(LIBS) -o $@ $?
.PHONY: all clean
.DEFAULT_GOAL: all
all:
	$(MAKE) $(TARGET) \
		OPT_FLAGS='$(DEF_OPT_FLAGS)'
debug:
	$(MAKE) $(TARGET) \
		OPT_FLAGS='$(DEBUG_OPT_FLAGS)'
clean:
	$(RM) $(OBJECTS) $(TARGET)
.depend:
	$(CC) -MM $(CFLAGS) $(SOURCES) > $@
include .depend