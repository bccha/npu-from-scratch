# Lessons Learned: CPU-to-FPGA Hardware Synchronization

This document summarizes critical lessons learned during the porting of the NPU verification framework from a simple bare-metal Nios II environment to a highly optimized Linux ARM Cortex-A9 environment.

## 1. The Danger of Arbitrary Delays (`usleep` / `HW_DELAY`) in High-Speed Systems

In the initial Nios II bare-metal implementation, the verification code often relied on arbitrary delays (e.g., `usleep()`, simple `for` loops, or macro-based `HW_DELAY()`) after commanding the FPGA to perform tasks like DMA transfers or MAC PE execution.

- **Why it worked in Nios II**: The Nios II soft-core processor operates at a low clock frequency (e.g., 50MHz ~ 100MHz), comparable to the FPGA fabric itself. It executes instructions strictly in-order, natively introducing sufficient physical time between I/O write operations for the FPGA to keep up.
- **Why it fails spectacularly in ARM Linux**: The ARM Cortex-A9 runs at 800MHz+. It features sophisticated L1/L2 caches, deep execution pipelines, and **Out-of-Order Execution**. If the C code issues an array of writes and then sleeps for 1ms assuming the FPGA has finished, the ARM CPU might execute the subsequent register triggers before the DMA has even fetched half the required data. This leads to profound data corruption, race conditions, and completely erratic `verify_streaming_batch()` read-outs.

**Lesson**: **Never use blind physical delays (`usleep`) for system synchronization.**

## 2. The Power of Status Register Polling

The solution to the synchronization disaster was shifting from time-based delays to **Status Register Polling**.

Instead of guessing when the hardware might be done, the software must explicitly read the hardware's internal state machine:

```c
// Wait for MSGDMA Read Dispatcher to finish its transaction (Busy Bit == 0)
while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0) { }

// Wait for NPU Core Sequencer to finish computation (Busy Bit == 0)
while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) { }
```

By chaining the MSGDMA busy flag polling with the NPU Sequencer busy flag polling, the C code forms a **perfectly interlocked software-hardware handshake**. The ARM CPU, no matter how fast it is, is forced to halt within the `while` loop until the FPGA explicitly signals that its pipeline is idle. This guaranteed a 100% pass rate in the 10-batch streaming tests on both ARM Linux and Nios II.

*Note: The only exception is isolated, basic IP blocks like the legacy `mac_pe` that lack a dedicated status CSR mapped to the Avalon bus. In those rare manual bit-banging scenarios, `usleep(1)` acts as a necessary localized physical clock-cycle guarantee.*

## 3. Demystifying Memory-Mapped I/O on ARM Linux (`O_SYNC` & `volatile`)

When moving from a bare-metal environment to a Linux Virtual Memory (`mmap`) environment, memory coherency becomes the biggest obstacle. If data is written to a C array, it lives in the CPU's cache, not the DDR3 RAM where the FPGA's DMA reads from.

- **Nios II**: Required manual software intervention via `alt_dcache_flush_all()` before triggering the DMA, pushing the cache to physical RAM.
- **ARM Linux**: Solved systemically at the OS level using the `O_SYNC` flag when opening `/dev/mem` combined with `MAP_SHARED` in `mmap()`:
  ```c
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  lw_bridge_map = mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, LWHPS2FPGA_BASE);
  ```

### How `volatile` and `O_SYNC` replace `__sync_synchronize()` (Memory Barriers)
We observed that the ARM CPU successfully wrote 64 matrix elements sequentially without needing explicit ARM Assembly barrier instructions (`dsb`, `dmb`). This works because of a three-tier defense:
1. **Compiler Barrier**: The `volatile` keyword in our `IOWR_32DIRECT` macro forces GCC to emit distinct assembly store instructions for every write, preventing the compiler from optimizing out "redundant" sequential writes to the same or nearby memory addresses.
2. **OS Mapping**: The `O_SYNC` flag instructs the Linux kernel to map the virtual memory page as **Strictly Ordered Device Memory** (uncached) rather than Normal RAM.
3. **Hardware MMU**: Because the ARM MMU flags the memory region as Device Memory, the CPU architecture mathematically guarantees that Memory Accesses to this region will NOT be aggressively re-ordered by its Out-of-Order execution engine. It naturally stalls the pipeline to ensure sequential delivery over the AXI/Avalon bridge, naturally syncing with the 50MHz FPGA without breaking the matrix layout.

## 4. Pipelined RTL Interfacing (`valid` / `ready` Handshakes)

During testing, the MAC PE failed with an output of `10` instead of the expected `31`. This was root-caused to an RTL design update in `mac_pe.v` that the legacy C code had not tracked.
The RTL moved from a simple "enable" to a full **Elastic Pipeline (Valid/Ready Handshake)**.

- **Old Verification Code**: Asserted `load_weight=1` and expected the PE to instantly latch it.
- **New Pipelined RTL requirement**: The PE now ignores any inputs unless **`valid_in_x == 1`** is asserted alongside the load command. Furthermore, since the PE uses Double-Buffering for weights, a separate `weight_latch_en` (mapped to NPU_CTRL offset 7) must be pulsed to commit the weight from the shadow register to the active multiplication stage.

**Lesson**: **Software porting is not just about adapting to a new OS; it requires intimately tracking RTL micro-architecture changes.** The Python `cocotb` testbench, Nios II firmware, and Linux ARM application all had to be simultaneously retrofitted to drive the `valid_in` and `weight_latch` signals defined by the new decoupled elastic pipeline.

## 5. Qsys MSGDMA Limits: The "Maximum Transfer Length" Truncation

While scaling the CPU vs NPU performance benchmark on the DE10-Nano, a silent DMA truncation bug emerged where the hardware correctly computed exactly `32` sequential batches of matrix data (8KB), but completely halted when pushed to `35` batches, causing outputs from batch 33 to 35 to match previous memory garbage (returning arrays filled with `ffffffa8`).

The root cause was isolated to the FPGA Hardware architecture. The Qsys (Platform Designer) configuration for the `DDR_READ_ST` and `DDR_WRITE_ST` MSGDMA cores contains a parameter called **"Maximum Transfer Length"**, which typically defaults to tight constants like `1024 Bytes` (1KB) or `8192 Bytes` (8KB). 

- 1 NPU output matrix (8x8 elements * 4 Bytes) = **256 Bytes**
- 32 iterations requested natively by software = 32 * 256 = **8,192 Bytes**

When the C code ordered a dynamic processing request for `100` batches (25,600 Bytes), the MSGDMA correctly transferred 8,192 Bytes (32 matrices) and then unconditionally severed the remainder of the transfer limit, leaving the NPU FSM hanging for data and the C loop validating empty arrays. 

**Lesson Checkpoint**:  
When massively scaling hardware-accelerated streams via DMA, cross-verify the native Transfer Length definitions within the Qsys instantiation settings window. Bumping the Maximum Transfer Length value symmetrically to **`1,048,576` (1MB)** inside Qsys definitively solved the bottleneck, unlocking thousands of hardware cycles without software truncation or stack overflows.
