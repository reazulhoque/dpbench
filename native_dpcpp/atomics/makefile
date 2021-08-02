# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

SRC:=                      \
       perf.cpp


# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := dpcpp
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := perf

CFLAGS      += -g -O3
CFLAGS      += -DSYCL_USE_NATIVE_FP_ATOMICS
CFLAGS      += -I./

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ $(TARGET)

.PHONY: all clean
