# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

SRC:=                      \
       load_data.cpp          \
       main.cpp


# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := dpcpp
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := gpairs

CFLAGS      += -g -O3
#CFLAGS      += -DSYCL_USE_NATIVE_FP_ATOMICS
CFLAGS      += -I./

PREC ?= d
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: $(TARGET)

$(TARGET): $(SRC) gpairs.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ gpairs

.PHONY: all clean
