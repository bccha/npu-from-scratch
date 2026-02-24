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

@cocotb.test()
async def test_dma_burst_read_write(dut):
    """Test end-to-end DMA operation (Memory -> DMA -> Sequencer -> Core -> Sequencer -> DMA -> Memory)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    TOTAL_ROWS = 8
    BEATS_IN = 2
    BEATS_OUT = 8
    RD_LEN = TOTAL_ROWS * BEATS_IN # 16 words
    WR_LEN = TOTAL_ROWS * BEATS_OUT # 64 words

    # 1. Configure System via CSR
    await write_reg(dut, REG_SEQ_ROWS, TOTAL_ROWS)
    await write_reg(dut, REG_DMA_RD_ADDR, 0x1000)
    await write_reg(dut, REG_DMA_RD_LEN, RD_LEN)
    await write_reg(dut, REG_DMA_WR_ADDR, 0x2000)
    
    # 2. Start Operations
    await write_reg(dut, REG_CTRL, (1 << 1) | 1) # Mode=1 (Exec), Start=1
    await write_reg(dut, REG_DMA_WR_CTRL, (1 << 17) | (1 << 16) | WR_LEN) # Start WR & RD

    # Memory Mocks
    rd_memory = [(i << 16) | 0xAA for i in range(RD_LEN)]
    wr_memory = []

    # Mock Avalon-MM Memory Slave for Read Master
    async def avalon_rd_slave():
        idx = 0
        while idx < RD_LEN:
            await FallingEdge(dut.clk)
            if dut.dma_rd_m_read.value == 1:
                burst = int(dut.dma_rd_m_burstcount.value)
                dut.dma_rd_m_waitrequest.value = 0
                await RisingEdge(dut.clk)
                
                # Provide burst data
                for b in range(burst):
                    await FallingEdge(dut.clk)
                    dut.dma_rd_m_readdatavalid.value = 1
                    dut.dma_rd_m_readdata.value = rd_memory[idx]
                    idx += 1
                    await RisingEdge(dut.clk)
                
                await FallingEdge(dut.clk)
                dut.dma_rd_m_readdatavalid.value = 0
            else:
                dut.dma_rd_m_waitrequest.value = 0
                await RisingEdge(dut.clk)

    # Mock Avalon-MM Memory Slave for Write Master
    async def avalon_wr_slave():
        while len(wr_memory) < WR_LEN:
            await FallingEdge(dut.clk)
            if dut.dma_wr_m_write.value == 1:
                dut.dma_wr_m_waitrequest.value = 0
                await Timer(1, "ns")
                val = int(dut.dma_wr_m_writedata.value)
                wr_memory.append(val)
                dut._log.info(f"WR_SLAVE: Captured Beat {len(wr_memory)}/{WR_LEN} -> 0x{val:08x}")
                await RisingEdge(dut.clk)
            else:
                dut.dma_wr_m_waitrequest.value = 0
                await RisingEdge(dut.clk)

    rd_task = cocotb.start_soon(avalon_rd_slave())
    wr_task = cocotb.start_soon(avalon_wr_slave())

    # Wait for DMA Write to complete
    timeout = 0
    while len(wr_memory) < WR_LEN:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > 200000:
            assert False, f"Timeout! Captured {len(wr_memory)}/{WR_LEN} beats. Last timeout was {timeout}"

    dut._log.info("All output beats captured via Avalon Write Master.")

    # Wait for System Done Status
    for _ in range(50):
        stat = await read_reg(dut, REG_G_STAT)
        if stat & 0x2: # seq_done
            break
        await Timer(100, "ns")

    dut._log.info("System DMA & Sequencer Burst Test Passed!")
