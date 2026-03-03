import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

def to_32b(val):
    return val & 0xFFFFFFFF

class AvalonMemoryModel:
    def __init__(self, dut, clock):
        self.dut = dut
        self.clock = clock
        self.mem = {} # Address to 32-bit int
        
        # Init inputs
        self.dut.avm_read_waitrequest.value = 0
        self.dut.avm_read_readdatavalid.value = 0
        self.dut.avm_read_readdata.value = 0
        
        self.dut.avm_write_waitrequest.value = 0
        
        self.read_thread = cocotb.start_soon(self.read_monitor())
        self.write_thread = cocotb.start_soon(self.write_monitor())

    def write_mem(self, addr, val):
        self.mem[addr] = val & 0xFFFFFFFF

    def read_mem(self, addr):
        return self.mem.get(addr, 0)

    async def read_monitor(self):
        while True:
            await RisingEdge(self.clock)
            self.dut.avm_read_readdatavalid.value = 0
            
            # Random waitrequest
            self.dut.avm_read_waitrequest.value = random.choice([0, 1])
            
            if self.dut.avm_read_read.value == 1 and self.dut.avm_read_waitrequest.value == 0:
                addr = int(self.dut.avm_read_address.value)
                cocotb.start_soon(self.respond_read(addr))
                
    async def respond_read(self, addr):
        wait_cycles = random.randint(1, 5)
        for _ in range(wait_cycles):
            await RisingEdge(self.clock)
        self.dut.avm_read_readdata.value = self.read_mem(addr)
        self.dut.avm_read_readdatavalid.value = 1

    async def write_monitor(self):
        while True:
            await RisingEdge(self.clock)
            self.dut.avm_write_waitrequest.value = random.choice([0, 0, 1])
            
            if self.dut.avm_write_write.value == 1 and self.dut.avm_write_waitrequest.value == 0:
                addr = int(self.dut.avm_write_address.value)
                data = int(self.dut.avm_write_writedata.value)
                self.write_mem(addr, data)


@cocotb.test()
async def test_npu_post_processor(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Init
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.src_addr.value = 0x1000
    dut.dst_addr.value = 0x2000
    dut.bias_addr.value = 0x3000
    dut.num_elements.value = 8
    dut.shift_val.value = 2
    
    mem_model = AvalonMemoryModel(dut, dut.clk)
    
    # Reset
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    num_elements = 16
    bias_addr = 0x3000
    src_addr = 0x1000
    dst_addr = 0x2000
    shift_val = 2
    
    # Generate test data (only 8 biases for 8 columns, applied cyclically)
    biases = [random.randint(-100, 100) for _ in range(8)]
    zs = [random.randint(-500, 500) for _ in range(num_elements)]
    
    for i in range(8):
        mem_model.write_mem(bias_addr + i*4, to_32b(biases[i]))
        
    for i in range(num_elements):
        mem_model.write_mem(src_addr + i*4, to_32b(zs[i]))
            
    # Start PP
    dut.src_addr.value = src_addr
    dut.dst_addr.value = dst_addr
    dut.bias_addr.value = bias_addr
    dut.num_elements.value = num_elements
    dut.shift_val.value = shift_val
    dut.start.value = 1
    
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for done
    timeout_cnt = 0
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
        timeout_cnt += 1
        if timeout_cnt > 10000:
            assert False, "Timeout waiting for done signal"
        
    await Timer(50, units="ns")
    
    # Check outputs
    # The new hardware natively applies an endianness swap (bswap32) on the incoming Z 
    # and a column swap (XOR id with 1) when packing the 8-bit output 
    # to perfectly match what Layer 2 (main.c) expects from OCM!
    
    expected_bytes = []
    # 1. Compute raw expected sequentially
    for i in range(num_elements):
        # Emulate __builtin_bswap32 of the Z input:
        # zs[i] are simply written as little endian natively by the testbench, 
        # so reading them as bswap32 means swapping the bytes of the 32-bit int.
        z_val = zs[i]
        z_bswap = ((z_val & 0xFF) << 24) | (((z_val >> 8) & 0xFF) << 16) | (((z_val >> 16) & 0xFF) << 8) | ((z_val >> 24) & 0xFF)
        # However, due to Python negative numbers vs infinite bits, 
        # let's map it safely to a 32-bit signed int
        if z_bswap & 0x80000000:
            z_bswap -= 0x100000000
            
        raw = z_bswap + biases[i % 8]
        shifted = raw >> shift_val
        if shifted < 0:
            final = 0
        elif shifted > 127:
            final = 127
        else:
            final = shifted
        expected_bytes.append(final)
        
    # Check memory considering the XOR column swap
    # Hardware packs into 32-bit words. Byte offsets 0,1,2,3 correspond to elements 0,1,2,3
    # BUT hardware XORs the index with 1.
    # So expected index logically mapped to byte pos: pos = index ^ 1
    for i in range(0, num_elements, 4):
        word_addr = dst_addr + i
        word_val = mem_model.read_mem(word_addr)
        
        b0 = word_val & 0xFF
        b1 = (word_val >> 8) & 0xFF
        b2 = (word_val >> 16) & 0xFF
        b3 = (word_val >> 24) & 0xFF
        
        hw_bytes = [b0, b1, b2, b3]
        
        for local_idx in range(4):
            global_idx = i + local_idx
            if global_idx < num_elements:
                expected_val = expected_bytes[global_idx]
                
                # The hardware placed expected_bytes[global_idx] into byte position (local_idx ^ 1)
                hw_pos = local_idx ^ 1
                hw_val = hw_bytes[hw_pos]
                
                assert hw_val == expected_val, f"Mismatch at global_idx {global_idx}: hardware_at_byte_{hw_pos}={hw_val} != expected={expected_val}"
                
    dut._log.info(f"SUCCESS: Post processor test passed with Layer-2 Endian/XOR Swapping!")
