import cocotb
from cocotb.triggers import Timer, RisingEdge, ReadOnly
from cocotb.clock import Clock
import numpy as np

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

async def avs_read(dut, addr):
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    val = int(dut.avs_readdata.value)
    await RisingEdge(dut.clk)
    return val

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
    """Monitor Avalon-ST Source and capture 64-bit words."""
    captured = []
    timeout = 0
    while len(captured) < expected_count and timeout < 2000:
        await RisingEdge(dut.clk)
        timeout += 1
        if int(dut.st_source_valid.value) == 1 and int(dut.st_source_ready.value) == 1:
            captured.append(int(dut.st_source_data.value))
            timeout = 0
            
    if timeout >= 2000:
        dut._log.error(f"Timeout waiting for Avalon-ST Source. Captured {len(captured)}/{expected_count}")
        
    return captured

async def avalon_mm_slave(dut):
    """Simulates an OCM (On-Chip Memory) responding to Avalon-MM Masters from Accumulator."""
    memory = {}
    dut.pp_read_waitrequest.value = 0
    dut.pp_write_waitrequest.value = 0
    dut.pp_read_readdatavalid.value = 0
    
    pending_read = False
    pending_addr = 0
    
    while True:
        await RisingEdge(dut.clk)
        
        # Read latency execution (1 cycle delay)
        if pending_read:
            dut.pp_read_readdatavalid.value = 1
            dut.pp_read_readdata.value = memory.get(pending_addr, 0)
            dut._log.info(f"[OCM] Serving READ Addr={hex(pending_addr)} Data={hex(memory.get(pending_addr, 0))}")
            pending_read = False
        else:
            dut.pp_read_readdatavalid.value = 0
            
        def is_one(sig):
            try: return int(sig.value) == 1
            except: return False

        # Handle Writes
        if is_one(dut.pp_write_write):
            try:
                addr = int(dut.pp_write_address.value)
                data = int(dut.pp_write_writedata.value)
                memory[addr] = data
                dut._log.info(f"[OCM] WRITE Addr={hex(addr)} Data={hex(data)}")
            except: pass
            
        # Handle Reads
        if is_one(dut.pp_read_read):
            try:
                pending_read = True
                pending_addr = int(dut.pp_read_address.value)
                dut._log.info(f"[OCM] REQ READ Addr={hex(pending_addr)}")
            except: pass

@cocotb.test()
async def test_hardware_fusion_pipeline(dut):
    """Test full NPU Fusion Pipeline: Array -> OCM Accumulator -> Post-Processor -> Drain"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    cocotb.start_soon(avalon_mm_slave(dut)) # Start OCM Memory Mock
    
    await reset_dut(dut)

    N = 8
    
    # 1. Prepare Data
    np.random.seed(42)
    # Increase range to avoid ReLU zeroing everything out after 8-bit >> shift
    weights = np.random.randint(-64, 64, size=(N, N)).astype(np.int8)
    batch_inputs = np.random.randint(0, 100, size=(N, N)).astype(np.int8)
    fused_biases = np.random.randint(-2000, 2000, size=(N)).astype(np.int32)
    
    # Emulate Math Fusion Pipeline
    expected_mac = np.matmul(batch_inputs.astype(np.int32), weights.astype(np.int32))
    expected_fused = expected_mac + fused_biases
    expected_shifted = np.right_shift(expected_fused, 8)
    # ReLU and 8-bit clipping
    expected_output = np.clip(expected_shifted, 0, 127).astype(np.int8)

    # 2. Write Biases to NPU_CTRL SRAM (Mapped at 0x200 to 0x3FF)
    dut._log.info("Loading Fused Biases into NPU_CTRL SRAM...")
    for i in range(N):
        await avs_write(dut, 0x200 + i, int(fused_biases[i]))

    # 3. Load Weights (seq_mode[0] = 1)
    dut._log.info("Setting mode to Load Weight...")
    await avs_write(dut, 0x0, 2) # Mode: Load Weight
    
    weight_stream = []
    for t in range(N):
        c = (N - 1) - t
        cycle_val = 0
        for r in range(N):
            w_val = int(weights[r, c]) & 0xFF
            cycle_val |= (w_val << (r * 8))
        weight_stream.append(cycle_val)

    await send_avalon_st(dut, weight_stream)
    for _ in range(30): await RisingEdge(dut.clk)
    
    # Latch weights
    await avs_write(dut, 7, 1) # weight_latch_en
    await RisingEdge(dut.clk)
    await avs_write(dut, 7, 0)
    for _ in range(5): await RisingEdge(dut.clk)

    # Remove the broken debug_monitor. The Avalon-MM log is enough.

    # 4. Setup Sequence properties (Total rows)
    await avs_write(dut, 6, N) # seq_total_rows = 8
    
    # 5. Execute ACCUMULATE Mode (seq_mode[1] = 0, seq_mode[0] = 0)
    dut._log.info("Setting mode to Execute (ACCUMULATE)...")
    await avs_write(dut, 0x0, 0) # Mode: Execute
    
    # Configure OCM Accumulator for ACCUMULATE (starts automatically on array stream)
    await avs_write(dut, 0x22, 0x00000000) # src_addr (not used for logic flow, but good practice)
    await avs_write(dut, 0x23, 0x80000000) # dst_addr (OCM Base Address)
    await avs_write(dut, 0x25, N * N)      # total 32-bit words (8 rows * 8 elements)
    await avs_write(dut, 0x20, 1)          # accum_start = 1
    
    input_stream = []
    for t in range(N):
        val = 0
        for r in range(N):
            x_val = int(batch_inputs[t, r]) & 0xFF
            val |= (x_val << (r * 8))
        input_stream.append(val)
        
    dut._log.info("Streaming Input Feature Map...")
    await send_avalon_st(dut, input_stream)

    # Poll accumulator done via CSR
    dut._log.info("Waiting for ACCUMULATE to finish...")
    timeout_acc = 0
    while True:
        done = await avs_read(dut, 0x221) # Base + 0x20 + 0x1? Wait, npu_ctrl.v address map!
        if done == 1:
            break
        timeout_acc += 1
        if timeout_acc > 1000:
            dut._log.error("Timeout polling accum_done!")
            break

    # 6. Execute DRAIN Mode (seq_mode[1] = 1, seq_mode[0] = 0 -> 2'b10 = 4)
    # But wait, seq_mode is mapped exactly to `seq_mode <= writedata[2:1]`.
    # So if writedata=0b100 (4), seq_mode = 0b10 (mode_drain = 1, mode_exec = 0).
    dut._log.info("Switching to DRAIN mode to extract Fused Results...")
    await avs_write(dut, 0x0, 4) # seq_mode = 2 means seq_mode[1]=1
    
    # Re-trigger OCM Accumulator for DRAIN
    await avs_write(dut, 0x23, 0x80000000) # Base Address to drain from
    await avs_write(dut, 0x25, N * N)      # total 32-bit words
    await avs_write(dut, 0x20, 1)          # accum_start = 1
    
    # 7. Monitor Output from Post-Processor -> Stream Source
    # 8x8 matrix output means 64 bytes total.
    # Since st_source_data is 64-bit (8 bytes), we expect exactly 8 flits!
    monitor_task = cocotb.start_soon(monitor_avalon_st(dut, N))
    
    dut._log.info("Waiting for Output (8 rows, 1 64-bit flit per row)...")
    results = await monitor_task
    
    assert len(results) == N, f"Failed to capture exactly {N} flits."

    # 8. Unpack Output Re-construct the matrix
    got_matrix = []
    for t in range(N):
        row_res = []
        # Remember that st_source sends 64-bit flits, but captured might be standard integers
        # N=8, so exactly one 64-bit flit per row output.
        # But wait, earlier I wrote `flit = int(results[t])`. If `results` is None, this fails.
        flit = int(results[t])
        for col in range(N):
            val = (flit >> (col * 8)) & 0xFF
            # Results are unsigned Int8 because of ReLU
            row_res.append(val)
        got_matrix.append(row_res)

    got = np.array(got_matrix)
    
    dut._log.info("--- Detailed Matrix Comparison ---")
    dut._log.info(f"Expected Result (After ReLU):\n{expected_output}")
    dut._log.info(f"Actual Result Captured:\n{got}")

    np.testing.assert_array_equal(got, expected_output, "Fusion Streaming Result Mismatch!")
    dut._log.info("Full Hardware Fusion Pipeline Test Passed Successfully!")
