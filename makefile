MAKE := make -f makefile
ifeq ($(OS),Windows_NT)
	TARG_LIB := gtb.dll
	DEFAULT_DEFS := -DMINGW
	DEFAULT_LIBS := -static-libstdc++
else
	TARG_LIB := libgtb.so
	DEFAULT_DEFS :=
	DEFAULT_LIBS := -lm -lpthread
endif
DEFAULT_CC := g++
DEFAULT_ARCH := -m64
DEFAULT_CFLAGS = -fPIC $(DEFAULT_ARCH) $(DEFAULT_DEFS)
OPT_CFLAGS := -msse -O3
DEBUG_CFLAGS := -O0 -g -DDEBUG
DEFAULT_LDFLAGS := -shared
SOURCES := Node.cpp
BUILD_DIR := build
OBJECTS = $(SOURCES:%.cpp=%.o)
$(TARG_LIB): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $@ $?
.PHONY: all clean
.DEFAULT_GOAL := debug
all:
	$(MAKE) $(TARG_LIB) \
		CC='$(DEFAULT_CC)' \
		CFLAGS='$(DEFAULT_CFLAGS)' \
		LDFLAGS='$(DEFAULT_LDFLAGS)' \
		LIBS='$(DEFAULT_LIBS)'
opt:
	$(MAKE) $(TARG_LIB) \
		CC='$(DEFAULT_CC)' \
		CFLAGS='$(DEFAULT_CFLAGS) $(OPT_CFLAGS)' \
		LDFLAGS='$(DEFAULT_LDFLAGS)' \
		LIBS='$(DEFAULT_LIBS)'
debug:
	$(MAKE) $(TARG_LIB) \
		CC='$(DEFAULT_CC)' \
		CFLAGS='$(DEFAULT_CFLAGS) $(DEBUG_CFLAGS)' \
		LDFLAGS='$(DEFAULT_LDFLAGS)' \
		LIBS='$(DEFAULT_LIBS)'
clean:
	$(RM) $(OBJECTS) $(TARG_LIB)
.depend:
	$(CC) -MM $(DEFAULT_CFLAGS) $(SOURCES) > $@
include .depend