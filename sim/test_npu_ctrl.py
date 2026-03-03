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
    dut.pp_done.value = 0
    
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
async def test_npu_ctrl_pp_registers(dut):
    """Test NPU Post Processor CSR read/write via Avalon-MM"""
    
    clock = Clock(dut.clk, 20, units="ns")  # 50 MHz
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test PP Registers (Select PP logic: address[7:4] == 4'h1)
    # The register mapping based on address[3:0]:
    # 0 = pp_start
    # 2 = pp_src_addr
    # 3 = pp_dst_addr
    # 4 = pp_bias_addr
    # 5 = pp_num_elements
    # 6 = pp_shift_val
    
    dut._log.info("Testing PP Source Address (Reg 18 = 0x12)")
    # Wrtiting to offset 18 -> we pass byte index? No, in C `IOWR(base, 18, data)`
    # writes to byte address 18 * 4 = 72 = 0x48.
    # But Qsys translates this back to word address `0x12` for the 8-bit Avalon-MM.
    # So `address` port on the RTL receives `0x12` (18 in decimal).
    # Let's write `0x12` and see what happens inside RTL.
    
    await avalon_write(dut, 0x12, 0x00040000)
    
    # Check internal RTL register state
    await ReadOnly()
    dut._log.info(f"Internal pp_src_addr: {hex(dut.pp_src_addr.value.integer)}")
    
    # Return to write phase
    await RisingEdge(dut.clk)
    
    # Now try to read it back via Avalon-MM
    read_val = await avalon_read(dut, 0x12)
    dut._log.info(f"Avalon Readback 0x12: {hex(read_val)}")
    assert read_val == 0x00040000, f"Expected 0x40000, got {hex(read_val)}"
    
    dut._log.info("Testing PP Shift Val (Reg 22 = 0x16)")
    await avalon_write(dut, 0x16, 8)
    
    await ReadOnly()
    dut._log.info(f"Internal pp_shift_val: {dut.pp_shift_val.value.integer}")
    
    await RisingEdge(dut.clk)
    
    read_val = await avalon_read(dut, 0x16)
    dut._log.info(f"Avalon Readback 0x16: {read_val}")
    assert read_val == 8, f"Expected 8, got {read_val}"

    # Also test MAC PE Registers
    dut._log.info("Testing MAC PE Registers (Select PE logic: address[7:4] == 4'h0 and address[3]==1)")
    # Reg 8 -> 0x08
    await avalon_write(dut, 0x08, 3) # Control Reg
    read_val = await avalon_read(dut, 0x08)
    dut._log.info(f"Avalon Readback 0x08 (PE_CTRL): {read_val}")
    
    dut._log.info("Test finished")
