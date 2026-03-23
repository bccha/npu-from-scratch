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

## 2. Dynamic Hardware Post-Processor ReLU and Shift
*   Once the bias issue is fixed, dynamically pipe `shift_val` from the AVS Memory Mapped CSR space into the Post-Processor instead of relying on the hardcoded `>>> 8` arithmetic shift.

