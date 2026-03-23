import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

# Simple mocked Avalon-MM SRAM
class AvalonMM_SRAM_Mock:
    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk
        self.memory = {} # 32-bit dict
        
        # Tie-offs
        self.dut.avm_read_readdata.value = 0
        self.dut.avm_read_waitrequest.value = 0
        self.dut.avm_read_readdatavalid.value = 0
        self.dut.avm_write_waitrequest.value = 0
        
        # Start monitoring coroutine
        cocotb.start_soon(self.monitor())

    async def monitor(self):
        while True:
            await RisingEdge(self.clk)
            
            # --- Read Port ---
            self.dut.avm_read_readdatavalid.value = 0
            if self.dut.avm_read_read.value == 1:
                # 1 cycle latency read
                addr = int(self.dut.avm_read_address.value)
                await RisingEdge(self.clk)
                
                # Fetch data, default to 0 if uninitialized
                val = self.memory.get(addr, 0)
                
                # Python int to 32-bit unsigned to avoid cocotb signed issues
                self.dut.avm_read_readdata.value = val & 0xFFFFFFFF
                self.dut.avm_read_readdatavalid.value = 1
            
            # --- Write Port ---
            if self.dut.avm_write_write.value == 1:
                addr = int(self.dut.avm_write_address.value)
                # Read signed 32-bit value being written
                val_raw = int(self.dut.avm_write_writedata.value)
                # Convert to signed int to store in python
                if val_raw & 0x80000000:
                    val = val_raw - 0x100000000
                else:
                    val = val_raw
                    
                self.memory[addr] = val

@cocotb.test()
async def test_ocm_accumulator_basic(dut):
    """Test standard accumulation between two memory regions"""
    
    # Start 50MHz Clock
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    
    # Init SRAM
    sram = AvalonMM_SRAM_Mock(dut, dut.clk)
    
    # Initialize inputs
    dut.start.value = 0
    dut.src_addr.value = 0
    dut.dst_addr.value = 0
    dut.num_elements.value = 0
    
    # Reset
    dut.rst_n.value = 0
    await Timer(50, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Reset complete.")
    
    # -------------------------------------------------------------
    # Test Setup: Generate random 8x8 matrices (64 elements)
    # -------------------------------------------------------------
    NUM_ELEMENTS = 64
    SRC_BASE = 0x1000
    DST_BASE = 0x2000
    
    expected_result = []
    
    dut._log.info(f"Populating mock SRAM with {NUM_ELEMENTS} random elements...")
    for i in range(NUM_ELEMENTS):
        # Generate random 32-bit signed values
        val_a = random.randint(-10000, 10000)
        val_b = random.randint(-10000, 10000)
        
        # Hardware expects MSGDMA byte-swapped format
        def swap32(x):
            return (((x << 24) & 0xFF000000) |
                    ((x << 8)  & 0x00FF0000) |
                    ((x >> 8)  & 0x0000FF00) |
                    ((x >> 24) & 0x000000FF))
                    
        a_u32 = swap32(val_a & 0xFFFFFFFF)
        b_u32 = swap32(val_b & 0xFFFFFFFF)
        
        # Store in Mock Memory
        sram.memory[SRC_BASE + (i*4)] = a_u32
        sram.memory[DST_BASE + (i*4)] = b_u32
        
        # Calculate Expected
        expected_result.append(val_a + val_b)
    
    await RisingEdge(dut.clk)
    
    # -------------------------------------------------------------
    # Trigger hardware accumulator
    # -------------------------------------------------------------
    dut._log.info("Triggering OCM Accumulator...")
    dut.src_addr.value = SRC_BASE
    dut.dst_addr.value = DST_BASE
    dut.num_elements.value = NUM_ELEMENTS
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # -------------------------------------------------------------
    # Wait for completion
    # -------------------------------------------------------------
    timeout_cycles = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        timeout_cycles += 1
        assert timeout_cycles < 2000, "Timeout: Hardware never asserted 'done'"
        
    dut._log.info(f"Accumulation complete in {timeout_cycles} cycles!")
    
    # -------------------------------------------------------------
    # Verify Results
    # -------------------------------------------------------------
    def swap32(x):
        return (((x << 24) & 0xFF000000) |
                ((x << 8)  & 0x00FF0000) |
                ((x >> 8)  & 0x0000FF00) |
                ((x >> 24) & 0x000000FF))
                
    errors = 0
    for i in range(NUM_ELEMENTS):
        hw_val_swapped = sram.memory.get(DST_BASE + (i*4), None)
        exp_val = expected_result[i]
        
        if hw_val_swapped is None:
            dut._log.error(f"Element {i} was not written to SRAM!")
            errors += 1
        else:
            hw_val_u32 = swap32(hw_val_swapped & 0xFFFFFFFF)
            if hw_val_u32 & 0x80000000:
                hw_val = hw_val_u32 - 0x100000000
            else:
                hw_val = hw_val_u32
                
            if hw_val != exp_val:
                dut._log.error(f"Mismatch at element {i}: HW={hw_val}, Expected={exp_val}")
                errors += 1
            
    assert errors == 0, f"Test Failed with {errors} errors!"
    dut._log.info("Test Passed! All 64 elements accumulated correctly.")
