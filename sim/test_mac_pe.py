import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock
import random

@cocotb.test()
async def mac_pe_basic_test(dut):
    """Test basic functionality of MAC PE: Weight Loading and MAC Operation"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.load_weight.value = 0
    dut.valid_in.value = 0
    dut.x_in.value = 0
    dut.y_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # 1. Weight Loading Test
    weight = 5
    dut.load_weight.value = 1
    dut.x_in.value = weight
    await RisingEdge(dut.clk)
    dut.load_weight.value = 0
    await RisingEdge(dut.clk)
    
    # Internal check might be hard if weight_reg is not a port, 
    # but we can verify it through subsequent MAC operation.

    # 2. MAC Operation Test
    # Result = y_in + (x_in * weight)
    # 0 + (10 * 5) = 50
    dut.valid_in.value = 1
    dut.x_in.value = 10
    dut.y_in.value = 0
    await RisingEdge(dut.clk)
    
    # Check output at next cycle (due to register)
    await FallingEdge(dut.clk)
    assert dut.y_out.value == 50, f"Expected 50, got {dut.y_out.value}"
    assert dut.x_out.value == 10, f"Expected x_out to be 10, got {dut.x_out.value}"
    assert dut.valid_out.value == 1, "Expected valid_out to be 1"

    # 3. Accumulated MAC Test
    # 50 + (2 * 5) = 60
    dut.x_in.value = 2
    dut.y_in.value = 50 # Feed back output (simulated upper PE)
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert dut.y_out.value == 60, f"Expected 60, got {dut.y_out.value}"

    dut._log.info("MAC PE Basic Test Passed!")

@cocotb.test()
async def mac_pe_randomized_test(dut):
    """Test MAC PE with randomized inputs"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    for i in range(20):
        # Load random weight
        weight = random.randint(-128, 127)
        dut.load_weight.value = 1
        dut.x_in.value = weight
        await RisingEdge(dut.clk)
        dut.load_weight.value = 0
        
        # Test MAC
        x = random.randint(-128, 127)
        y = random.randint(-1000, 1000)
        expected = y + (x * weight)
        
        dut.valid_in.value = 1
        dut.x_in.value = x
        dut.y_in.value = y
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        
        assert dut.y_out.value.signed_integer == expected, \
            f"Iteration {i}: Expected {expected}, got {dut.y_out.value.signed_integer} (x={x}, w={weight}, y={y})"
    
    dut._log.info("MAC PE Randomized Test Passed!")
