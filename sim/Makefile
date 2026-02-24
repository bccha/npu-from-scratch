# Makefile for Cocotb simulation
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/../rtl/mac_pe.v
VERILOG_SOURCES += $(PWD)/../rtl/mac_pe_ctrl.v
VERILOG_SOURCES += $(PWD)/../rtl/npu_ctrl.v
VERILOG_SOURCES += $(PWD)/../rtl/npu_dma.v
VERILOG_SOURCES += $(PWD)/../rtl/systolic_array_8x8.v
VERILOG_SOURCES += $(PWD)/../rtl/systolic_core.v
VERILOG_SOURCES += $(PWD)/../rtl/npu_unit.v

TOPLEVEL = npu_unit
MODULE = test_avalon_bus

include $(shell cocotb-config --makefiles)/Makefile.sim
