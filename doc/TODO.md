# NPU Hardware Bug Fixes & Optimization Roadmap

## 1. OCM Accumulator Bias Addition Bug (Critical)
**Issue:**  
Currently, the hardware `npu_ocm_accumulator.v` injects a 32-bit hardware `$signed(bias_X)` scalar directly into the 256-bit `$signed(accum_val)` bus using a standard `+` operator. 
```verilog
ocm_ram[wr_addr] <= accum_val + 
    (wr_addr == 0 ? bias_0 : ...);
```
This is fatally incorrect for two reasons:
1. **Addressing Axis Mismatch for CNNs**: `wr_addr` during CNN operations sweeps across the *Sequence Index* (Spatial Patches) rather than the Output Channels. This means `bias_0` is applied to Patch 0 (Output Channel 0), and `bias_1` is applied to Patch 1 (Output Channel 0). Output Channels 1~7 entirely miss receiving their biases.
2. **Carry Propagation Cross-Contamination**: A 32-bit integer added to a 256-bit bus intrinsically allows the addition `carry-out` of the lower 32-bits (`y_out_0`) to propagate into the higher bits (`y_out_1` to `y_out_7`), completely corrupting adjacent neural outputs.

**Planned Fix:**
1. Rearchitect the 256-bit accumulator to apply the 8 `bias_N` elements uniquely to their respective 32-bit channels in parallel, strictly forbidding cross-boundary carry propagation.
2. Example Solution:
```verilog
wire [31:0] sum_0 = accum_val[31:0]     + bias_0;
wire [31:0] sum_1 = accum_val[63:32]    + bias_1;
wire [31:0] sum_2 = accum_val[95:64]    + bias_2;
wire [31:0] sum_3 = accum_val[127:96]   + bias_3;
wire [31:0] sum_4 = accum_val[159:128]  + bias_4;
wire [31:0] sum_5 = accum_val[191:160]  + bias_5;
wire [31:0] sum_6 = accum_val[223:192]  + bias_6;
wire [31:0] sum_7 = accum_val[255:224]  + bias_7;

wire [255:0] biased_accum = {sum_7, sum_6, sum_5, sum_4, sum_3, sum_2, sum_1, sum_0};
```
3. Remove the software `npu_extract_32bit_ocm` fallback in the Python Compiler and reactivate the fully offloaded MSGDMA `npu_drain_to_ddr` which uses hardware ReLU natively.

## 2. True Post-Training Quantization (PTQ) via CSR Registers
**Issue:**  
Currently, the Python SDK compiler forces a static `>> 8` division on all layers, which survives primarily due to Batch Normalization. True PTQ requires an exact decimal scale multiplier and dynamic shifting per layer to prevent accuracy drops on non-linear data distributions.

**Planned Fix:**  
1. **Verilog Updates (`npu_ctrl.v` & `npu_post_processor.v`):** 
   Replace the hardcoded `>>> SHIFT_VAL` with two new Memory-Mapped Control Registers: `REG_QUANT_MUL` and `REG_QUANT_SHIFT`.
   ```verilog
   wire signed [31:0] scaled_sum = stg3_sum * reg_quant_mul;
   wire signed [31:0] final_out_8bit = (scaled_sum >>> reg_quant_shift) + reg_quant_zero_point;
   ```
2. **Compiler SDK Updates (`cyclone_npu_sdk.py`):**
   - Inject **Calibration Forward Hooks** during `export_to_fpga()` to run ~500 images through the PyTorch model and record the true empirical `Min/Max` of every activation tensor.
   - Calculate output precision constants: $M = \frac{Scale_{in} \times Scale_{weight}}{Scale_{out}}$
   - Decompose $M$ mathematically into an Integer Multiplier ($M_0$) and bit-shift ($n$) utilizing `math.log2()` approximations: $M \approx M_0 \times 2^{-n}$
   - Emit standard C code to dynamically `IOWR` these 2 values into the NPU `npu_ctrl` before every channel/layer execution.

## 3. Resurrecting 100% Hardware Pipeline (Fixing MSGDMA Deadlock)
**Issue:**  
Currently, CNN processing relies on a Software CPU Fallback (`npu_extract_32bit_ocm`) because routing `npu_post_processor.v` to the `MSGDMA Stream-to-Memory TX` module caused a System Deadlock (`out_ready` permanently tied low). This limits CNN execution time to ~65ms due to the CPU extracting and calculating 169 matrix patches sequentially.

**Planned Fix:**  
1. **Verilog MSGDMA Alignment (`npu_stream_ctrl.v`):**
   - Ensure the Avalon-ST `Packer` module exactly matches the Byte Length and `SOP/EOP` boundaries expected by the MSGDMA C-descriptors.
   - If `seq_total_rows` (Hardware) and the Descriptor length (Software) disagree by even 1 byte, MSGDMA will hang waiting for `EOP`, blocking the `out_ready` bus permanently.
2. **Re-activate Hardware Bias API (`npu_api.c`):**
   - Remove the `IOWR(NPU_CTRL_BASE, 0x200 + f, 0); // Disable HW Bias permanently` hack.
   - Restore the explicit memory-mapped Write of the native 32-bit Bias arrays into the NPU `0x200` CSR space.
3. **Compiler Code-Gen Update (`cyclone_npu_sdk.py`):**
   - Delete the Software Extraction C-loops generated during `emit_c_code()`.
   - Replace them exclusively with the `npu_drain_to_ddr(output_address)` MSGDMA instruction, returning the CNN pipeline to an undisputed Zero-Software-Overhead state. (Expected CNN performance jump: 65ms -> ~3ms).
