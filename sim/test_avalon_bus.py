import cocotb
from cocotb.triggers import Timer, RisingEdge, ReadOnly
from cocotb.clock import Clock
from cocotb.queue import Queue
import random

async def reset_dut(dut):
    dut.rst_n.value = 0
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def avs_write(dut, addr, data):
    dut.avs_address.value = addr
    dut.avs_writedata.value = data
    dut.avs_write.value = 1
    await RisingEdge(dut.clk)
    dut.avs_write.value = 0

async def avs_read(dut, addr):
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    # Wait for readdatavalid (1 cycle delay)
    while True:
        await RisingEdge(dut.clk)
        if dut.avs_readdatavalid.value == 1:
            return dut.avs_readdata.value

@cocotb.test()
async def test_slave_registers(dut):
    """Test Avalon-MM Slave Read/Write to all internal registers"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 1. Test Controller Registers (Address 0-3)
    # 0: CTRL (load_weight, valid_in)
    # 1: X_IN
    # 2: Y_IN
    # 3: Y_OUT (Read-only)
    test_val_x = 0x7F
    test_val_y = 0x12345678
    
    await avs_write(dut, 1, test_val_x)
    await avs_write(dut, 2, test_val_y)
    
    read_x = await avs_read(dut, 1)
    read_y = await avs_read(dut, 2)
    
    assert int(read_x) == test_val_x, f"X_IN mismatch: exp {test_val_x:x}, got {int(read_x):x}"
    assert int(read_y) == test_val_y, f"Y_IN mismatch: exp {test_val_y:x}, got {int(read_y):x}"
    
    # 2. Test DMA Registers (Address 4-7)
    # 4: RD_ADDR, 5: RD_LEN, 6: WR_ADDR, 7: WR_LEN/START/STATUS
    src_addr = 0x10000000
    rd_len = 512
    dst_addr = 0x20000000
    
    await avs_write(dut, 4, src_addr)
    await avs_write(dut, 5, rd_len)
    await avs_write(dut, 6, dst_addr)
    
    read_src = await avs_read(dut, 4)
    read_rd_len = await avs_read(dut, 5)
    read_dst = await avs_read(dut, 6)
    
    assert int(read_src) == src_addr, f"RD_ADDR mismatch: exp {src_addr:x}, got {int(read_src):x}"
    assert int(read_rd_len) == rd_len, f"RD_LEN mismatch: exp {rd_len}, got {int(read_rd_len)}"
    assert int(read_dst) == dst_addr, f"WR_ADDR mismatch: exp {dst_addr:x}, got {int(read_dst):x}"
    
    dut._log.info("Slave Register Read/Write Test Passed!")

@cocotb.test()
async def test_dma_loopback(dut):
    """Test full DMA Loopback: Read Master -> FIFO -> Write Master"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Simulated External Memory
    mem_src = {}
    mem_dst = {}
    SRC_BASE = 0x1000
    DST_BASE = 0x2000
    NUM_WORDS = 64
    
    # Initialize all master inputs to 0 to avoid X-propagation
    dut.dma_rd_m_waitrequest.value = 0
    dut.dma_rd_m_readdatavalid.value = 0
    dut.dma_rd_m_readdata.value = 0
    dut.dma_wr_m_waitrequest.value = 0
    
    # Initialize Source Memory
    for i in range(NUM_WORDS):
        mem_src[SRC_BASE + i*4] = random.getrandbits(32)

    # rd_master_handler refactored to handle overlapping requests
    rd_cmd_queue = Queue()

    async def rd_master_cmd_monitor():
        while True:
            await RisingEdge(dut.clk)
            if dut.dma_rd_m_read.value == 1:
                addr = int(dut.dma_rd_m_address.value)
                burst = int(dut.dma_rd_m_burstcount.value)
                rd_cmd_queue.put_nowait((addr, burst))

    async def rd_master_data_sender():
        dut.dma_rd_m_waitrequest.value = 0
        dut.dma_rd_m_readdatavalid.value = 0
        dut.dma_rd_m_readdata.value = 0
        while True:
            addr, burst = await rd_cmd_queue.get()
            for i in range(burst):
                await RisingEdge(dut.clk)
                dut.dma_rd_m_readdata.value = mem_src.get(addr + i*4, 0xDEADBEEF)
                dut.dma_rd_m_readdatavalid.value = 1
            await RisingEdge(dut.clk)
            dut.dma_rd_m_readdatavalid.value = 0

    async def wr_master_handler():
        dut.dma_wr_m_waitrequest.value = 0
        while True:
            await RisingEdge(dut.clk)
            if dut.dma_wr_m_write.value == 1:
                addr = int(dut.dma_wr_m_address.value)
                burst = int(dut.dma_wr_m_burstcount.value)
                for i in range(burst):
                    while dut.dma_wr_m_waitrequest.value == 1:
                        await RisingEdge(dut.clk)
                    mem_dst[addr + i*4] = int(dut.dma_wr_m_writedata.value)
                    if i < burst - 1:
                        await RisingEdge(dut.clk)

    cocotb.start_soon(rd_master_cmd_monitor())
    cocotb.start_soon(rd_master_data_sender())
    cocotb.start_soon(wr_master_handler())

    # 3. Trigger DMA via Slave Interface
    # Register 7: {14'd0, wr_start, rd_start, wr_len[15:0]}
    # RD_START = bit 16, WR_START = bit 17
    dma_ctrl_val = (1 << 17) | (1 << 16) | (NUM_WORDS & 0xFFFF)
    
    await avs_write(dut, 4, SRC_BASE)   # RD_ADDR
    await avs_write(dut, 5, NUM_WORDS)  # RD_LEN
    await avs_write(dut, 6, DST_BASE)   # WR_ADDR
    await avs_write(dut, 7, dma_ctrl_val) # WR_LEN & START

    # 4. Wait for Done (Bit 16 = RD_DONE, Bit 17 = WR_DONE)
    for i in range(2000):
        status = await avs_read(dut, 7)
        if (int(status) & (1 << 17)) and (int(status) & (1 << 16)):
            break
        
        if i % 100 == 0:
            # Probe internal signals via hierarchy
            def safe_int(val):
                return val.value.integer if val.value.is_resolvable else val.value.binstr

            rd_pend  = safe_int(dut.u_npu_dma.rd_pending_beats)
            wr_rem   = safe_int(dut.u_npu_dma.wr_rem_len)
            fifo_cnt = safe_int(dut.u_npu_dma.fifo_count)
            rd_state = safe_int(dut.u_npu_dma.rd_state)
            wr_state = safe_int(dut.u_npu_dma.wr_state)
            
            dut._log.info(f"DMA Polling: status={int(status):08x}, fifo={fifo_cnt}, rd_st={rd_state}, wr_st={wr_state}, rd_pend={rd_pend}, wr_rem={wr_rem}")
            
        await RisingEdge(dut.clk)
    else:
        # Final dumping of state
        fifo_cnt = int(dut.u_npu_dma.fifo_count.value)
        rd_st = int(dut.u_npu_dma.rd_state.value)
        wr_st = int(dut.u_npu_dma.wr_state.value)
        dut._log.error(f"FATAL: DMA Timeout. Final state: fifo_cnt={fifo_cnt}, rd_st={rd_st}, wr_st={wr_st}")
        assert False, "DMA Timeout"

    # 5. Verify Results
    for i in range(NUM_WORDS):
        addr = SRC_BASE + i*4
        out_addr = DST_BASE + i*4
        assert mem_dst.get(out_addr) == mem_src[addr], \
            f"Mismatch at index {i}: addr {out_addr:x}, got {mem_dst.get(out_addr):x}, exp {mem_src[addr]:x}"

    dut._log.info("DMA Loopback Test Passed!")
