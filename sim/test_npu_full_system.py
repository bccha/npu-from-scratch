import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge

# Unified Register Map (from npu_ctrl.v)
REG_CTRL = 0x0
REG_G_STAT = 0x1
REG_DMA_RD_ADDR = 0x2
REG_DMA_RD_LEN = 0x3
REG_DMA_WR_ADDR = 0x4
REG_DMA_WR_CTRL = 0x5
REG_SEQ_ROWS = 0x6
REG_DMA_STAT = 0x7

async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.avs_address.value = 0
    dut.avs_write.value = 0
    dut.avs_read.value = 0
    dut.avs_writedata.value = 0
    dut.dma_rd_m_waitrequest.value = 0
    dut.dma_rd_m_readdata.value = 0
    dut.dma_rd_m_readdatavalid.value = 0
    dut.dma_wr_m_waitrequest.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

async def write_reg(dut, addr, data):
    dut.avs_address.value = addr
    dut.avs_writedata.value = data
    dut.avs_write.value = 1
    await RisingEdge(dut.clk)
    dut.avs_write.value = 0
    await Timer(1, "ns")

async def read_reg(dut, addr):
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    
    # Wait for readdatavalid
    timeout = 0
    while True:
        await Timer(1, "ns")
        valid = dut.avs_readdatavalid.value
        if valid.is_resolvable and int(valid) == 1:
            data = int(dut.avs_readdata.value)
            await RisingEdge(dut.clk)
            return data
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > 100:
            assert False, "Timeout waiting for readdatavalid"

@cocotb.test()
async def test_csr_access(dut):
    """Test read/write access to the unified CSRs"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    test_addr = 0x12345678
    test_len  = 0x400

    dut._log.info("Writing DMA Read Config...")
    await write_reg(dut, REG_DMA_RD_ADDR, test_addr)
    await write_reg(dut, REG_DMA_RD_LEN, test_len)

    dut._log.info("Reading back DMA Read Config...")
    rd_addr = await read_reg(dut, REG_DMA_RD_ADDR)
    rd_len  = await read_reg(dut, REG_DMA_RD_LEN)

    assert rd_addr == test_addr, f"RD_ADDR mismatch: {hex(rd_addr)} != {hex(test_addr)}"
    assert rd_len == test_len, f"RD_LEN mismatch: {hex(rd_len)} != {hex(test_len)}"
    
    dut._log.info("CSR Access Test Passed!")

import numpy as np

@cocotb.test()
async def test_dma_burst_read_write(dut):
    """Test end-to-end NPU MatMul (Load Weights -> Exec -> Verify with Numpy)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    TOTAL_ROWS = 8
    BEATS_IN = 1
    BEATS_OUT = 4
    RD_LEN = TOTAL_ROWS * BEATS_IN # 8 words (64-bit)
    WR_LEN = TOTAL_ROWS * BEATS_OUT # 32 words (64-bit)

    dut._log.info("Generating Random 8x8 Matrices via Numpy...")
    # Generate random 8-bit signed integers (-128 to 127)
    # Let's define the weights as they physically exist in the MAC array (Row i, Col j)
    weights = np.random.randint(-128, 127, size=(8, 8), dtype=np.int8)
    inputs  = np.random.randint(-128, 127, size=(8, 8), dtype=np.int8)
    
    # Calculate Ground Truth
    expected_output = np.dot(inputs.astype(np.int32), weights.astype(np.int32))

    # Memory Mocks
    rd_memory = []
    wr_memory = []

    # Pack 8x8 Inputs into DMA 64-bit beats (1 beat per row sequence)
    # HW expects Column vectors per cycle: core_x_in = [Row7, Row6, ..., Row0]
    def pack_matrix_cols(mat, reverse=False):
        mem = []
        cols = range(8)
        if reverse:
            cols = reversed(cols)
            
        for col in cols:
            beat = 0
            for i in range(8):
                beat |= (int(mat[i, col]) & 0xFF) << (i * 8)
            mem.append(beat)
        return mem

    def pack_matrix_rows(mat):
        mem = []
        for row in range(8):
            beat = 0
            for i in range(8):
                beat |= (int(mat[row, i]) & 0xFF) << (i * 8)
            mem.append(beat)
        return mem


    # Avalon Read Slave Task
    async def avalon_rd_slave():
        while True:
            await FallingEdge(dut.clk)
            if dut.dma_rd_m_read.value == 1:
                burst = int(dut.dma_rd_m_burstcount.value)
                dut.dma_rd_m_waitrequest.value = 0
                await RisingEdge(dut.clk)
                for b in range(burst):
                    await FallingEdge(dut.clk)
                    dut.dma_rd_m_readdatavalid.value = 1
                    if len(rd_memory) > 0:
                        dut.dma_rd_m_readdata.value = rd_memory.pop(0)
                    else:
                        dut.dma_rd_m_readdata.value = 0
                    await RisingEdge(dut.clk)
                
                await FallingEdge(dut.clk)
                dut.dma_rd_m_readdatavalid.value = 0
            else:
                dut.dma_rd_m_waitrequest.value = 0
                await RisingEdge(dut.clk)

    # Avalon Write Slave Task
    async def avalon_wr_slave():
        while True:
            await FallingEdge(dut.clk)
            if dut.dma_wr_m_write.value == 1:
                dut.dma_wr_m_waitrequest.value = 0
                await Timer(1, "ns")
                val = int(dut.dma_wr_m_writedata.value)
                # Received a 64-bit value, which comprises 2 32-bit output elements
                # Let's split it into 2 32-bit elements to keep `wr_memory` elements 32-bits (which match the C code's integer expectation)
                val0 = val & 0xFFFFFFFF
                val1 = (val >> 32) & 0xFFFFFFFF
                
                # Convert 32-bit unsigned to signed
                if val0 & 0x80000000:
                    val0 -= 0x100000000
                if val1 & 0x80000000:
                    val1 -= 0x100000000
                wr_memory.extend([val0, val1])
                await RisingEdge(dut.clk)
            else:
                dut.dma_wr_m_waitrequest.value = 0
                await RisingEdge(dut.clk)

    cocotb.start_soon(avalon_rd_slave())
    cocotb.start_soon(avalon_wr_slave())

    # --- PHASE 1: LOAD WEIGHTS ---
    dut._log.info("Phase 1: Loading Weights...")
    rd_memory = pack_matrix_cols(weights, reverse=True)
    
    await write_reg(dut, REG_SEQ_ROWS, TOTAL_ROWS)
    await write_reg(dut, REG_DMA_RD_ADDR, 0x1000)
    await write_reg(dut, REG_DMA_RD_LEN, RD_LEN)
    
    await write_reg(dut, REG_CTRL, (0 << 1) | 1) # Mode=0 (Load Weight), Start=1
    await write_reg(dut, REG_DMA_WR_CTRL, (1 << 16)) # Start RD only

    # Wait for Sequencer to finish Loading Weights
    for _ in range(100):
        stat = await read_reg(dut, REG_G_STAT)
        if stat & 0x2: break
        await Timer(100, "ns")

    dut._log.info("Dumping Hardware MAC PE Weight Registers after Phase 1:")
    for r in range(8):
        row_weights = []
        for c in range(8):
            # Access the internal weight_reg of each PE
            pe = getattr(dut.u_systolic_core.u_array, f"row[{r}]").col[c].u_pe
            w_val = int(pe.weight_reg.value)
            if w_val & 0x80: w_val -= 0x100 # signed 8-bit
            row_weights.append(w_val)
        dut._log.info(f"HW Row {r} Weights: {row_weights}")
    
    dut._log.info(f"NP Reference Weights (weights matrix):")
    for r in range(8):
        dut._log.info(f"NP Row {r} Weights: {weights[r].tolist()}")

    # --- PHASE 2: EXECUTION ---
    # --- PHASE 2: EXECUTION ---
    dut._log.info("Phase 2: Execution...")
    rd_memory = pack_matrix_rows(inputs)
    
    await write_reg(dut, REG_DMA_RD_ADDR, 0x2000)
    await write_reg(dut, REG_DMA_WR_ADDR, 0x3000)
    
    await write_reg(dut, REG_CTRL, (1 << 1) | 1) # Mode=1 (Exec), Start=1
    await write_reg(dut, REG_DMA_WR_CTRL, (1 << 17) | (1 << 16) | WR_LEN) # Start WR & RD

    # Wait for DMA Write to complete
    timeout = 0
    while len(wr_memory) < 64:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > 200000:
            assert False, f"Timeout! Captured {len(wr_memory)}/64 outputs. Last timeout was {timeout}"

    dut._log.info(f"Captured {len(wr_memory)} output elements. Verifying with Numpy...")
    
    # Verify outputs
    matched = True
    for row in range(8):
        for col in range(8):
            hw_val = wr_memory[row * 8 + col]
            # The NPU computes straightforward matrix multiplication X * W
            np_val = expected_output[row, col]
            # Actually, let's just log mismatched values to see the pattern
            if hw_val != np_val:
                dut._log.error(f"Mismatch at [{row},{col}]: HW={hw_val}, NP={np_val}")
                matched = False

    if matched:
        dut._log.info("System DMA & Sequencer Numpy Verification Passed!")
    else:
        assert False, "Verification Mismatch!"
