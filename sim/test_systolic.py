import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import numpy as np
import random
import os

async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.ready_in.value = 0
    dut.load_weight_in.value = 0
    dut.x_in.value = 0
    dut.y_in.value = 0
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_systolic_core_flow_control(dut):
    """Stress Test: Valid/Ready Flow Control with Random Stalls"""
    if os.environ.get("WAVES") == "1":
        # iverilog dump is usually done in verilog or through Makefile
        pass
        
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    
    N = 8
    NUM_MATRICES = 10  # Reduced for random stall test
    TOTAL_ROWS = NUM_MATRICES * N
    
    # Random seeded for reproducibility
    np.random.seed(37)
    random.seed(37)
    
    # 1. Prepare Data
    weights = np.random.randint(-64, 63, size=(N, N)).astype(np.int8)
    batch_inputs = np.random.randint(-64, 63, size=(TOTAL_ROWS, N)).astype(np.int8)
    
    # Expected Result: (TOTAL_ROWS, N)
    expected_batch_y = np.matmul(batch_inputs.astype(np.int32), weights.astype(np.int32))

    dut._log.info("Starting Flow Control Test...")

    # Let's fix the driver for true AXI-Stream random stalls
    async def robust_driver():
        # A) Load Weights
        for t in range(N):
            c = (N - 1) - t 
            cycle_val = 0
            for r in range(N):
                w_val = int(weights[r, c]) & 0xFF
                cycle_val |= (w_val << (r * 8))
            
            dut.x_in.value = cycle_val
            dut.load_weight_in.value = (1 << N) - 1
            
            while True:
                is_valid = random.choice([0, 1, 1])
                dut.valid_in.value = is_valid
                await RisingEdge(dut.clk)
                if is_valid and int(dut.ready_out.value) == 1:
                    break
        # Flush weights through skew buffers and shift registers
        dut.valid_in.value = 0
        dut.load_weight_in.value = 0
        for _ in range(30):
            await RisingEdge(dut.clk)
            
        # Global Weight Latch
        dut.weight_latch_en.value = 1
        await RisingEdge(dut.clk)
        dut.weight_latch_en.value = 0
        
        # B) Stream Inputs
        for t in range(TOTAL_ROWS):
            val = 0
            for r in range(N):
                x_val = int(batch_inputs[t, r]) & 0xFF
                val |= (x_val << (r * 8))
            
            dut.x_in.value = val
            dut.load_weight_in.value = 0
            
            while True:
                is_valid = random.choice([0, 1, 1, 1])
                dut.valid_in.value = is_valid
                await RisingEdge(dut.clk)
                if is_valid and int(dut.ready_out.value) == 1:
                    dut._log.debug(f"Row {t} sent: {batch_inputs[t]}")
                    break
                    
        dut.valid_in.value = 0

    # ---------------------------------------------------------
    # Monitor Task (Receives outputs with random ready drops)
    # ---------------------------------------------------------
    results = []
    async def monitor():
        while len(results) < TOTAL_ROWS:
            is_ready = random.choice([0, 1, 1]) # 66% chance ready
            dut.ready_in.value = is_ready
            await RisingEdge(dut.clk)
            
            if is_ready and int(dut.valid_out.value) == 1:
                # Capture result
                y_val = int(dut.y_out.value)
                row_res = []
                for j in range(N):
                    extract = (y_val >> (j * 32)) & 0xFFFFFFFF
                    if extract & 0x80000000:
                        extract -= 0x100000000
                    row_res.append(extract)
                results.append(row_res)

    # ---------------------------------------------------------
    # Run Simulation
    # ---------------------------------------------------------
    dut.y_in.value = 0
    cocotb.start_soon(robust_driver())
    monitor_task = cocotb.start_soon(monitor())
    
    # Timeout watch
    timeout = 0
    while len(results) < TOTAL_ROWS and timeout < 20000:
        await RisingEdge(dut.clk)
        timeout += 1

    if timeout >= 20000:
        dut._log.error(f"Timeout! Captured {len(results)}/{TOTAL_ROWS} rows")
        assert False

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------
    got = np.array(results)
    
    try:
        np.testing.assert_array_equal(got, expected_batch_y)
        dut._log.info("Flow Control Test Passed! No duplicated or lost data.")
        dut._log.info(f"\n--- Weights Matrix ---\n{weights}")
        dut._log.info(f"\n--- Input Matrix (First 5 Rows) ---\n{batch_inputs[:5]}")
        dut._log.info(f"\n--- Expected Output (First 5 Rows) ---\n{expected_batch_y[:5]}")
        dut._log.info(f"\n--- Hardware Output (First 5 Rows) ---\n{got[:5]}")
    except Exception as e:
        dut._log.error(f"Mismatch:\nExpected:\n{expected_batch_y[:5]}\nGot:\n{got[:5]}")
        raise e
