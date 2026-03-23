import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, ClockCycles
import logging

async def reset_dut(dut):
    dut.rst_n.value = 0
    
    # Initialize all inputs
    dut.address.value = 0
    dut.write.value = 0
    dut.writedata.value = 0
    dut.read.value = 0
    dut.seq_busy.value = 0
    dut.seq_done.value = 0
    dut.pe_x_out.value = 0
    dut.pe_y_out.value = 0
    dut.pe_valid_out.value = 0
    dut.accum_done.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

async def avalon_write(dut, address, data):
    dut.address.value = address
    dut.writedata.value = data
    dut.write.value = 1
    await RisingEdge(dut.clk)
    dut.write.value = 0
    await ClockCycles(dut.clk, 1)

async def avalon_read(dut, address):
    dut.address.value = address
    dut.read.value = 1
    await RisingEdge(dut.clk)
    dut.read.value = 0
    # Wait for readdatavalid
    while True:
        await ReadOnly()
        if dut.readdatavalid.value == 1:
            data = dut.readdata.value.integer
            await RisingEdge(dut.clk)
            return data
        await RisingEdge(dut.clk)

@cocotb.test()
async def test_npu_ctrl_acc_registers(dut):
    """Test NPU Accumulator CSR read/write via Avalon-MM"""
    
    clock = Clock(dut.clk, 20, units="ns")  # 50 MHz
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test Accumulator Registers (Select ACC logic: address[7:4] == 4'h2)
    # The register mapping based on address[3:0]:
    # 0 = accum_start
    # 1 = accum_done (read-only)
    # 2 = accum_src_addr
    # 3 = accum_dst_addr
    # 5 = accum_num_elements
    
    dut._log.info("Testing ACC SRC Address (Reg 34 = 0x22)")
    # Qsys translates to 0x22
    await avalon_write(dut, 0x22, 0x00080000)
    
    # Check internal RTL register state
    await ReadOnly()
    dut._log.info(f"Internal accum_src_addr: {hex(dut.accum_src_addr.value.integer)}")
    
    # Return to write phase
    await RisingEdge(dut.clk)
    
    # Now try to read it back via Avalon-MM
    read_val = await avalon_read(dut, 0x22)
    dut._log.info(f"Avalon Readback 0x22: {hex(read_val)}")
    assert read_val == 0x00080000, f"Expected 0x80000, got {hex(read_val)}"
    
    dut._log.info("Testing ACC NUM ELEMENTS (Reg 37 = 0x25)")
    await avalon_write(dut, 0x25, 64)
    
    await ReadOnly()
    dut._log.info(f"Internal accum_num_elements: {dut.accum_num_elements.value.integer}")
    
    await RisingEdge(dut.clk)
    
    read_val = await avalon_read(dut, 0x25)
    dut._log.info(f"Avalon Readback 0x25: {read_val}")
    assert read_val == 64, f"Expected 64, got {read_val}"

    # Also test MAC PE Registers
    dut._log.info("Testing MAC PE Registers (Select PE logic: address[7:4] == 4'h0 and address[3]==1)")
    # Reg 8 -> 0x08
    await avalon_write(dut, 0x08, 3) # Control Reg
    read_val = await avalon_read(dut, 0x08)
    dut._log.info(f"Avalon Readback 0x08 (PE_CTRL): {read_val}")
    
    dut._log.info("Test finished")
