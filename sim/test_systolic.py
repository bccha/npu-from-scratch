import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import numpy as np

async def reset_dut(dut):
    dut.rst_n.value = 0
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_systolic_core_stress(dut):
    """Stress Test: Load Weight Once, Stream 100 Matrices (400 Rows)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    N = 8
    NUM_MATRICES = 100
    TOTAL_ROWS = NUM_MATRICES * N
    
    # 1. Prepare Data
    weights = np.random.randint(-64, 63, size=(N, N)).astype(np.int8)
    batch_inputs = np.random.randint(-64, 63, size=(TOTAL_ROWS, N)).astype(np.int8)
    
    # Expected Result: (TOTAL_ROWS, N)
    expected_batch_y = np.matmul(batch_inputs.astype(np.int32), weights.astype(np.int32))

    # 2. Phase 1: Load Weights (8 cycles)
    # HW shifts left-to-right, so we feed Col 7 first, then Col 6... Col 0 last.
    # systolic_core handles the diagonal skew internally.
    dut._log.info("Starting Weight Loading (8 cycles)...")
    dut.load_weight.value = 1
    for t in range(N):
        c = (N - 1) - t # Col 7, 6, ..., 0
        cycle_val = 0
        for r in range(N):
            w_val = int(weights[r, c]) & 0xFF
            cycle_val |= (w_val << (r * 8))
        dut.x_in.value = cycle_val
        await RisingEdge(dut.clk)
    
    dut.load_weight.value = 0
    dut.x_in.value = 0
    await RisingEdge(dut.clk)

    # 3. Phase 2: Feed Inputs (Streaming 800 rows without gaps)
    dut._log.info(f"Streaming {TOTAL_ROWS} Input Rows...")
    dut.y_in.value = 0
    
    results = []
    pipeline_errors = 0
    first_valid_cycle = -1
    
    async def feed_inputs():
        for t in range(TOTAL_ROWS):
            val = 0
            for r in range(N):
                x_val = int(batch_inputs[t, r]) & 0xFF
                val |= (x_val << (r * 8))
            
            dut.valid_in.value = 0xF 
            dut.x_in.value = val
            await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        dut.x_in.value = 0

    # 4. Phase 3: Collect Results and Check Pipeline
    async def collect_results():
        nonlocal first_valid_cycle, pipeline_errors
        # Initial Latency: Skew(0) + Array(8) + Deskew(7) = 15 cycles.
        # Plus 1 cycle PE output reg? Let's check.
        
        for cycle in range(TOTAL_ROWS + 50): 
            await RisingEdge(dut.clk)
            val_out = dut.valid_out.value
            
            if val_out.is_resolvable and (int(val_out) == 0xF):
                if first_valid_cycle == -1:
                    first_valid_cycle = cycle
                    dut._log.info(f"First valid result captured at cycle {cycle}")
                
                y_out_raw = str(dut.y_out.value)
                row_res = []
                for col in range(N):
                    start = (N - 1 - col) * 32
                    end = start + 32
                    chunk_str = y_out_raw[start:end]
                    
                    if 'x' in chunk_str.lower() or 'z' in chunk_str.lower():
                        row_res.append(0xDEADBEEF)
                    else:
                        chunk = int(chunk_str, 2)
                        if chunk & 0x80000000:
                            chunk -= 0x100000000
                        row_res.append(chunk)
                results.append(row_res)
                
                if len(results) == TOTAL_ROWS:
                    break
            else:
                # If we are in the middle of a batch and valid drops, that's a pipeline gap!
                if first_valid_cycle != -1 and len(results) < TOTAL_ROWS:
                    dut._log.error(f"Cycle {cycle}: Pipeline gap detected! valid_out={val_out}")
                    pipeline_errors += 1

    # Start streaming
    cocotb.start_soon(feed_inputs())
    await collect_results()

    # 5. Final Evaluation
    got = np.array(results)
    dut._log.info(f"Stream Complete. Rows Captured: {len(results)}, Pipeline Errors: {pipeline_errors}")
    
    # 6. Detailed Visibility for USER
    dut._log.info("--- Detailed Matrix Comparison (First 4x4 Matrix: A_0) ---")
    dut._log.info(f"Weight Matrix (B):\n{weights}")
    dut._log.info(f"First Input Matrix (A_0):\n{batch_inputs[0:4]}")
    dut._log.info(f"Expected Result (A_0 * B):\n{expected_batch_y[0:4]}")
    dut._log.info(f"Actual Result Captured:\n{got[0:4]}")
    
    dut._log.info("--- Detailed Matrix Comparison (Last 4x4 Matrix: A_99) ---")
    dut._log.info(f"Last Input Matrix (A_99):\n{batch_inputs[-4:]}")
    dut._log.info(f"Expected Result (A_99 * B):\n{expected_batch_y[-4:]}")
    dut._log.info(f"Actual Result Captured:\n{got[-4:]}")
    
    if len(results) < TOTAL_ROWS:
        assert False, f"Verification Failed: Captured only {len(results)}/{TOTAL_ROWS} rows."

    assert pipeline_errors == 0, "Verification Failed: Pipeline bubbles detected during streaming!"

    # Math Verification for all 100 matrices
    np.testing.assert_array_equal(got, expected_batch_y, "Streaming result mismatch!")
    dut._log.info("Systolic Core Stress Test (100 Matrices) Passed!")
