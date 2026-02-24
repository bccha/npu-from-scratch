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

    TOTAL_ROWS = 4
    BEATS_IN = 1
    BEATS_OUT = 4
    RD_LEN = TOTAL_ROWS * BEATS_IN # 4 words
    WR_LEN = TOTAL_ROWS * BEATS_OUT # 16 words

    dut._log.info("Generating Random 4x4 Matrices via Numpy...")
    # Generate random 8-bit signed integers (-128 to 127)
    # Let's define the weights as they physically exist in the MAC array (Row i, Col j)
    weights = np.random.randint(-128, 127, size=(4, 4), dtype=np.int8)
    inputs  = np.random.randint(-128, 127, size=(4, 4), dtype=np.int8)
    
    # Calculate Ground Truth
    expected_output = np.dot(inputs.astype(np.int32), weights.astype(np.int32))

    # Memory Mocks
    rd_memory = []
    wr_memory = []

    # Pack 8x8 Inputs into DMA 32-bit beats (2 beats per row sequence)
    # HW expects Column vectors per cycle: core_x_in = [Row7, Row6, ..., Row0]
    # So beat0 = [Row3, Row2, Row1, Row0] of the CURRENT column.
    def pack_matrix_cols(mat, reverse=False):
        mem = []
        # Normal (Inputs): Col 0, Col 1, Col 2...
        # Weights: HW shifts left-to-right. Element fed first ends up in Col 7.
        # So Weights must feed Col 7, Col 6, ..., Col 0.
        cols = range(4)
        if reverse:
            cols = reversed(cols)
            
        for col in cols:
            beat0 = ((int(mat[3, col]) & 0xFF) << 24) | ((int(mat[2, col]) & 0xFF) << 16) | ((int(mat[1, col]) & 0xFF) << 8) | (int(mat[0, col]) & 0xFF)
            mem.extend([beat0])
        return mem

    def pack_matrix_rows(mat):
        mem = []
        for row in range(4):
            beat = ((int(mat[row, 3]) & 0xFF) << 24) | ((int(mat[row, 2]) & 0xFF) << 16) | ((int(mat[row, 1]) & 0xFF) << 8) | (int(mat[row, 0]) & 0xFF)
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
                # Convert 32-bit unsigned to signed
                if val & 0x80000000:
                    val -= 0x100000000
                wr_memory.append(val)
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
    for r in range(4):
        row_weights = []
        for c in range(4):
            # Access the internal weight_reg of each PE
            pe = getattr(dut.u_systolic_core.u_array, f"row[{r}]").col[c].u_pe
            w_val = int(pe.weight_reg.value)
            if w_val & 0x80: w_val -= 0x100 # signed 8-bit
            row_weights.append(w_val)
        dut._log.info(f"HW Row {r} Weights: {row_weights}")
    
    dut._log.info(f"NP Reference Weights (weights matrix):")
    for r in range(4):
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
    while len(wr_memory) < WR_LEN:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > 200000:
            assert False, f"Timeout! Captured {len(wr_memory)}/{WR_LEN} beats. Last timeout was {timeout}"

    dut._log.info(f"Captured {len(wr_memory)} output beats. Verifying with Numpy...")
    
    # Verify outputs
    matched = True
    for row in range(4):
        for col in range(4):
            hw_val = wr_memory[row * 4 + col]
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
