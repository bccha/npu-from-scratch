import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge, Combine, ClockCycles
from cocotb.clock import Clock
from cocotb.types import LogicArray
import numpy as np

async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.mode.value = 0
    dut.total_rows.value = 0
    dut.dma_data_in.value = 0
    dut.dma_data_in_valid.value = 0
    dut.dma_data_out_ready.value = 0
    dut.core_y_out.value = 0
    dut.core_valid_out.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

@cocotb.test()
async def test_sequencer_throughput_full(dut):
    """Back-to-Basics: 16-Row Throughput with Hyper-Verbose ID Tracking"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    TOTAL_ROWS = 16
    BEATS_PER_ROW_IN = 1
    BEATS_PER_ROW_OUT = 4
    
    dut.total_rows.value = TOTAL_ROWS
    dut.mode.value = 1 
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    stats = {"fed": 0, "captured": 0, "errors": 0}

    # 1. Feeder: modeled after test_systolic.py synchronous feed
    async def feeder():
        for r in range(TOTAL_ROWS):
            for b in range(BEATS_PER_ROW_IN):
                while True:
                    await FallingEdge(dut.clk)
                    # Check ready (handle LogicArray)
                    rdy = dut.dma_data_in_ready.value
                    if rdy.is_resolvable and int(rdy) == 1:
                        break
                
                # ID is the high 16 bits of the word
                val = (r << 16) | (b << 8) | 0xAA
                dut.dma_data_in.value = val
                dut.dma_data_in_valid.value = 1
                await RisingEdge(dut.clk)
            
            await FallingEdge(dut.clk)
            dut.dma_data_in_valid.value = 0
            stats["fed"] += 1
            await RisingEdge(dut.clk)
        dut._log.info("TB_FEED: Done streaming inputs.")

    # 2. Collector: Cycle-accurate tracking with ID validation
    async def collector():
        dut.dma_data_out_ready.value = 1
        expected_total_beats = TOTAL_ROWS * BEATS_PER_ROW_OUT
        
        cycle = 0
        while stats["captured"] < expected_total_beats:
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            cycle += 1
            
            # Check valid (handle LogicArray)
            vld = dut.dma_data_out_valid.value
            if vld.is_resolvable and int(vld) == 1:
                dat = dut.dma_data_out.value
                val = int(dat) if dat.is_resolvable else 0xDEADBEEF
                
                stats["captured"] += 1
                row_idx = (stats["captured"] - 1) // BEATS_PER_ROW_OUT
                sub_beat = (stats["captured"] - 1) % BEATS_PER_ROW_OUT
                
                # Validation Logic
                actual_id = (val >> 16) & 0xFFFF
                expected_id = row_idx
                
                status = "PASS" if actual_id == expected_id else "FAIL"
                if actual_id != expected_id: stats["errors"] += 1
                
                dut._log.info(f"[Cyc {cycle:4}] Beat {stats['captured']:3}/{expected_total_beats} | Row {row_idx:2}, Sub {sub_beat} | ID: Exp={expected_id:2}, Act={actual_id:2} | {status} | Val=0x{val:08x}")
                
        dut._log.info("TB_COLL: All beats collected.")

    # 3. Flat Core Mock: Sync with core_valid_in
    async def core_mock():
        queue = [] # [latency, data]
        total_batches = TOTAL_ROWS
        batches_processed = 0
        
        while stats["captured"] < (TOTAL_ROWS * BEATS_PER_ROW_OUT):
            await FallingEdge(dut.clk)
            
            # Sample core_valid_in
            c_vld = dut.core_valid_in.value
            if c_vld.is_resolvable and (int(c_vld) & 1):
                c_data = dut.core_x_in.value
                raw_x = int(c_data) if c_data.is_resolvable else 0
                queue.append([15, raw_x]) # Latency = 15
            
            # Update queue
            for item in queue:
                if item[0] > 0: item[0] -= 1
            
            # Output if ready
            if queue and queue[0][0] == 0:
                _, data = queue.pop(0)
                # Input (data) is 32-bit. Row output (core_y_out) is 128-bit.
                # Replicate 32-bit data 4 times to get 128 bits.
                clean_data = int(data) & 0xFFFFFFFF
                res_hex = f"{clean_data:08x}" * 4
                dut.core_y_out.value = int(res_hex, 16)
                dut.core_valid_out.value = 0xF
                batches_processed += 1
            else:
                dut.core_valid_out.value = 0
                dut.core_y_out.value = 0
            
            await RisingEdge(dut.clk)

    await Combine(
        cocotb.start_soon(feeder()),
        cocotb.start_soon(collector()),
        cocotb.start_soon(core_mock())
    )

    # Verification
    dut._log.info(f"Verification Results: Captured={stats['captured']}, Errors={stats['errors']}")
    assert stats["errors"] == 0, f"Row ID Mismatch detected in stream! Total Errors: {stats['errors']}"
    
    # Wait for DONE
    success = False
    for i in range(1000):
        await RisingEdge(dut.clk)
        if dut.done.value.is_resolvable and int(dut.done.value) == 1:
            success = True
            break
    assert success, "Done signal not reached!"
    dut._log.info("NPU Sequencer Back-to-Basics Test PASSED!")