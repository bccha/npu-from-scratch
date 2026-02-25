import cocotb
from cocotb.triggers import Timer, RisingEdge, ReadOnly
from cocotb.clock import Clock

async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.avs_write.value = 0
    dut.avs_read.value = 0
    dut.st_sink_valid.value = 0
    dut.st_source_ready.value = 1
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def avs_write(dut, addr, data):
    dut.avs_address.value = addr
    dut.avs_writedata.value = data
    dut.avs_write.value = 1
    await RisingEdge(dut.clk)
    dut.avs_write.value = 0
    await RisingEdge(dut.clk)

async def send_avalon_st(dut, data_list):
    """Send a list of 64-bit words to the Avalon-ST Sink."""
    for i, data in enumerate(data_list):
        dut.st_sink_data.value = data
        dut.st_sink_valid.value = 1
        dut.st_sink_startofpacket.value = 1 if i == 0 else 0
        dut.st_sink_endofpacket.value = 1 if i == len(data_list) - 1 else 0
        
        # Wait until accepted
import numpy as np

# Wait until accepted
async def send_avalon_st(dut, data_list):
    """Send a list of 64-bit words to the Avalon-ST Sink."""
    for i, data in enumerate(data_list):
        dut.st_sink_data.value = int(data)
        dut.st_sink_valid.value = 1
        dut.st_sink_startofpacket.value = 1 if i == 0 else 0
        dut.st_sink_endofpacket.value = 1 if i == len(data_list) - 1 else 0
        
        while True:
            await RisingEdge(dut.clk)
            if int(dut.st_sink_ready.value) == 1:
                break
    
    # Deassert valid after sending all
    dut.st_sink_valid.value = 0
    await RisingEdge(dut.clk)

async def monitor_avalon_st(dut, expected_count):
    """Monitor Avalon-ST Source and capture 64-bit words, checking SOP/EOP."""
    captured = []
    timeout = 0
    sop_count = 0
    eop_count = 0
    while len(captured) < expected_count and timeout < 1000:
        await RisingEdge(dut.clk)
        timeout += 1
        if int(dut.st_source_valid.value) == 1 and int(dut.st_source_ready.value) == 1:
            if int(dut.st_source_startofpacket.value) == 1:
                assert len(captured) == 0, f"SOP asserted at flit index {len(captured)}"
                sop_count += 1
            if int(dut.st_source_endofpacket.value) == 1:
                assert len(captured) == expected_count - 1, f"EOP asserted at flit index {len(captured)}"
                eop_count += 1
            captured.append(int(dut.st_source_data.value))
            timeout = 0 # reset timeout on valid data
            
    if timeout >= 1000:
        dut._log.error(f"Timeout waiting for Avalon-ST Source. Captured {len(captured)}/{expected_count}")
        
    assert sop_count == 1, f"Expected 1 SOP, got {sop_count}"
    assert eop_count == 1, f"Expected 1 EOP, got {eop_count}"
    
    return captured

@cocotb.test()
async def test_npu_stream(dut):
    """Test NPU with Avalon-ST Sink and Source (8x8)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    N = 8
    
    # 1. Prepare Data
    # Numpy arrays for easy verification
    np.random.seed(37)
    weights = np.random.randint(-64, 63, size=(N, N)).astype(np.int8)
    # Using small positive numbers to avoid complicated two's complement decoding in this visual check
    batch_inputs = np.random.randint(0, 10, size=(N, N)).astype(np.int8)
    
    expected_batch_y = np.matmul(batch_inputs.astype(np.int32), weights.astype(np.int32))

    # 2. Set mode to Load Weight (bit 1 of REG_CTRL (seq_mode))
    dut._log.info("Setting mode to Load Weight...")
    # seq_mode == 1 (Load Weight) is bits [2:1], so we write 2 (0b010)
    await avs_write(dut, 0, 2)
    for _ in range(5):
        await RisingEdge(dut.clk)

    # 3. Send Weights (Column by Column, from N-1 down to 0)
    weight_stream = []
    for t in range(N):
        c = (N - 1) - t # Col 7, 6, ..., 0
        cycle_val = 0
        for r in range(N):
            w_val = int(weights[r, c]) & 0xFF
            cycle_val |= (w_val << (r * 8))
        weight_stream.append(cycle_val)

    # DO NOT append dummy zeros to weight_stream!
    # The systolic_core input_skew delays BOTH `load_weight` AND `valid_in` along with the data.
    # Therefore, if we assert `valid_in` for exactly N cycles, EVERY row gets exactly N load pulses!
    # Row 0 gets them cycles 0..7. Row 7 gets them cycles 7..14.

    dut._log.info("Sending Weights over Avalon-ST...")
    await send_avalon_st(dut, weight_stream)

    # 3.5. Wait for the pipeline to flush weights
    for _ in range(30):
        await RisingEdge(dut.clk)
    
    # 3.6. Trigger Global Latch Enable
    dut._log.info("Triggering Global Weight Latch Enable...")
    await avs_write(dut, 7, 1)
    await RisingEdge(dut.clk)
    await avs_write(dut, 7, 0)
    for _ in range(5):
        await RisingEdge(dut.clk)

    # Check internal weight registers
    dut._log.info("--- Internal Weight Registers Loaded ---")
    for row in range(8):
        row_weights = []
        for col in range(8):
            # Access internal path
            w = dut.u_systolic_core.u_array.row[row].col[col].u_pe.active_weight_reg.value
            if w.is_resolvable:
                row_weights.append(int(w))
                if int(w) & 0x80:
                    row_weights[-1] -= 0x100
            else:
                row_weights.append(None)
        dut._log.info(f"Row {row}: {row_weights}")

    # 4. Set mode to Execute (Clear seq_mode[0])
    dut._log.info("Setting mode to Execute...")
    await avs_write(dut, 0, 0)
    for _ in range(5):
        await RisingEdge(dut.clk)

    # 4.5 Configure total rows for EOP generation
    dut._log.info(f"Setting seq_total_rows to {N}...")
    await avs_write(dut, 6, N)
    for _ in range(5):
        await RisingEdge(dut.clk)

    # 5. Connect monitor for N rows * 4 flits = 32
    monitor_task = cocotb.start_soon(monitor_avalon_st(dut, N * 4))

    # 6. Send Inputs over Avalon-ST (Row by Row)
    input_stream = []
    for t in range(N):
        val = 0
        for r in range(N):
            x_val = int(batch_inputs[t, r]) & 0xFF
            val |= (x_val << (r * 8))
        input_stream.append(val)
        
    dut._log.info(f"Sending Data over Avalon-ST...")
    await send_avalon_st(dut, input_stream)

    dut._log.info("Waiting for Output (8 rows * 4 flits = 32 64-bit flits)...")
    results = await monitor_task

    if len(results) < 32:
        dut._log.error(f"Failed to capture 32 flits, got {len(results)}")
        assert False

    # 7. Reconstruct the 256-bit Row Vectors
    reconstructed_rows = []
    for i in range(N): # 8 Output rows
        val256 = 0
        # The hardware transmits tx_shift_reg[63:0] and shifts right: {64_d0, tx_shift_reg[255:64]}
        # Thus:
        # flit 0 is original [63:0]
        # flit 1 is original [127:64]
        # flit 2 is original [191:128]
        # flit 3 is original [255:192]
        for j in range(4):
            flit = int(results[i * 4 + j])
            val256 |= (flit << (j * 64))
        reconstructed_rows.append(val256)

    # 8. Unpack the 256-bit vectors and verify against Numpy expected_batch_y
    # Note: Output row timing from systolic_core is vertically skewed, meaning row 0 comes out first.
    # We must wait 15 cycles (N-1 + 8) for the first valid result of row 0.
    got_matrix = []
    for t in range(N):
        row_res = []
        row_val = reconstructed_rows[t]
        for col in range(N):
            start = col * 32
            # Extract 32-bit chunk
            chunk = (row_val >> start) & 0xFFFFFFFF
            # Convert unsigned to signed 32-bit
            if chunk & 0x80000000:
                chunk -= 0x100000000
            row_res.append(chunk)
        got_matrix.append(row_res)

    got = np.array(got_matrix)
    
    dut._log.info("--- Detailed Matrix Comparison ---")
    dut._log.info(f"Weight Matrix:\n{weights}")
    dut._log.info(f"Input Matrix:\n{batch_inputs}")
    dut._log.info(f"Expected Result:\n{expected_batch_y}")
    dut._log.info(f"Actual Result Captured:\n{got}")

    np.testing.assert_array_equal(got, expected_batch_y, "Avalon-ST Streaming Result Mismatch!")
    dut._log.info("Avalon-ST Streaming Test (8x8) Passed Successfully without Duplication!")
