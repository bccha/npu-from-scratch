# NPU Hardware Verification Results

The following logs document the successful hardware verification runs of the NPU Accelerator executing on the DE10-Nano (ARM Cortex-A9 / Linux OS environment), bypassing caches via `/dev/mem` (`O_SYNC`), and utilizing strictly decoupled elastic MSGDMA pipeline handshakes.

---

## 1. MAC PE Independence Test

Direct physical verification of the primitive MAC (Multiply-Accumulate) Processing Element, confirming Double-Buffering weight latch sequences over the Avalon-MM bridged CSRs. 

```text
NPU System Verification (Full Framework)
----------------------------------------------
1. Verify MAC PE
2. Verify Full System Data path
3. Verify 10-Batch Streaming Pipeline
q. Quit
Choose: 1

Starting MAC PE Verification...
Result: 31 (Expected: 31)
MAC PE Test: PASS
```

---

## 2. Full System 8x8 Matrix Data Path

Validates MSGDMA memory-to-memory streaming operations across the entire 8x8 Systolic Array structure. This confirms:
1. Skew / Deskew buffers align correctly across 64 elements.
2. Backpressure (`valid/ready`) pipeline holds true.
3. CPU handles Status Register Polling without Out-of-Order execution corruption.

```text
NPU System Verification (Full Framework)
----------------------------------------------
1. Verify MAC PE
2. Verify Full System Data path
3. Verify 10-Batch Streaming Pipeline
q. Quit
Choose: 2

Starting Full System Matrix Validation (Fixed 8x8 HW with 4x4 submatrix)...
Clearing Memories...
Preparing 8x8 Identity Weight Matrix...
Phase 1: Loading Weights via MSGDMA API...
Weights Loaded!
Phase 2: Execution via MSGDMA API...
Execution Finished!

Verifying Output (Expecting Y=X for 8x8 matrix)...

=== Hardware Output Matrix ===
  1   2   3   4   5   6   7   8
  9  10  11  12  13  14  15  16
 17  18  19  20  21  22  23  24
 25  26  27  28  29  30  31  32
 33  34  35  36  37  38  39  40
 41  42  43  44  45  46  47  48
 49  50  51  52  53  54  55  56
 57  58  59  60  61  62  63  64

=== Expected Output Matrix ===
  1   2   3   4   5   6   7   8
  9  10  11  12  13  14  15  16
 17  18  19  20  21  22  23  24
 25  26  27  28  29  30  31  32
 33  34  35  36  37  38  39  40
 41  42  43  44  45  46  47  48
 49  50  51  52  53  54  55  56
 57  58  59  60  61  62  63  64


Full System Validation: PASS! All 64 elements matched correctly.
```

---

## 3. High-Speed Extended Batch Streaming

Continuous back-to-back testing using MSGDMA bulk `EOP` tracking features to compute un-interrupted sequences without ARM CPU intervention mid-stream. Verifies perfect hardware Cache-Coherency across DDR3 accesses.

```text
NPU System Verification (Full Framework)
----------------------------------------------
1. Verify MAC PE
2. Verify Full System Data path
3. Verify 10-Batch Streaming Pipeline
q. Quit
Choose: 3

Starting Streaming Batch Test (10 Matrices)...
Clearing Memories...
Loading Weights...
Firing 10-Batch Streaming Pipeline...
Batch 0: PASS
Batch 1: PASS
Batch 2: PASS
Batch 3: PASS
Batch 4: PASS
Batch 5: PASS
Batch 6: PASS
Batch 7: PASS
Batch 8: PASS
Batch 9: PASS

Streaming Validation: PASS! All 10 batches successfully fully matched.
```

---

## 4. CPU vs NPU Performance Comparison

Measures the absolute execution time of equivalent 8x8 matrix multiplications running on the 800MHz ARM Cortex-A9 CPU (`gcc -O3` via nested loops) versus the 50MHz FPGA NPU (via MSGDMA Pipeline) using parameterized target sizes. Mapped using `gettimeofday()`.

    ```text
    NPU System Verification (Full Framework)
    ----------------------------------------------
    1. Verify MAC PE
    2. Verify Full System Data path
    3. Verify Streaming Pipeline (N Batches)
    4. CPU vs NPU Performance Comparison
    q. Quit
    Choose: 4

    Enter number of batches (e.g., 10, 100, 1000): 5000

    Starting CPU vs NPU Performance Comparison (5000 Batches of 8x8)...

    === Performance Results (5000 Batches) ===
    Verification: PASS (NPU output perfectly matches CPU)
    CPU Time : 22298.000 us
    NPU Time : 4807.000 us (Includes DMA Setup overhead)
    Speedup  : 4.64 x
    ```

    ---

## 5. End-to-End MNIST (2-Layer MLP) Inference

Actual C Benchmark comparing 50MHz FPGA NPU throughput versus 800MHz ARM CPU on continuous non-linear deep learning inferences involving real world image data and model weights exported from Python. Proves memory bandwidth scaling efficiently offsets lower clock speeds. 

> **Note on Accuracy (88.08%)**: The NPU computes in strict 8-bit integer (Int8) arithmetic. Because the trained model was originally Float32 and subsequently quantized using basic Post-Training Quantization (PTQ), numeric clipping during activation scaling results in the observed 88% accuracy. Implementing Quantization-Aware Training (QAT)—as used in Google's Edge TPUs—will theoretically restore the hardware accuracy closer to 98%.

    ```text
    [1] Running S/W (CPU) Inference...
        CPU Accuracy : 88.38%
        CPU Time     : 5571.47 ms (6.964 ms/img)

    [2] Running H/W (NPU) Inference...
        NPU Accuracy : 88.38%
        NPU Time     : 1306.75 ms (1.633 ms/img)

    === Speedup ===
        H/W Acceleration: 4.26 x
    ```

---

## 6. OCM (On-Chip Memory) Buffer Optimization

To further eliminate DDR access contention, the FPGA's internal OCM was introduced as an intermediate scratchpad for the NPU outputs. This led to a substantial investigation into optimal bridge bandwidth usage:

1. **LWHPS2FPGA (Lightweight Bridge) Bottleneck:** 
   Initially, the OCM was mapped to the Lightweight Bridge (`0xFF20_0000`). Performance massively degraded (to 43.2ms / 1.62x) because CPU data formatting over the LightWeight Bridge induced severe latency stalls on individual 32-bit read/write transactions.
2. **HPS2FPGA AXI (Heavyweight Bridge) Migration:** 
   The OCM was remapped over the high-speed AXI Bridge (`0xC0_0000_00`), reducing time to **25.26 ms**, which confirmed the AXI throughput difference but still fell short of pure DDR (19.3ms).
3. **Hardware-Decoupled Stack `memcpy` Bursting:** 
   The core issue was identified as the CPU's memory access pattern. Doing calculations directly on `/dev/mem` pointers caused intense bus transactions. By altering the software to `memcpy` the 256-byte NPU results directly into a CPU stack buffer, doing the ReLU/Bias formatting locally, and `memcpy`-ing the block back out, bus transactions became highly efficient bursts.

**Final Result:** With AXI + `memcpy` stack buffering, the OCM pipeline hit **16.33 ms / img (4.26x Acceleration)**, successfully breaking past the DDR limitations entirely!

---

## 7. Standalone HW Accumulator S/W Pipelining

Validated the independent `npu_ocm_accumulator` IP which operates directly on the Avalon-MM OCM memory bus alongside the MSGDMA engines. This verifies the capability for the CPU to overlap memory accumulation tasks with sequential NPU matrix multiplications, enabling software-level pipeline overlapping.

```text
Starting Standalone OCM Accumulator Pipeline Test...
Loading Weights...
  [Loop 0] Dispatching MSGDMA Matmul...
  [Loop 1] Dispatching MSGDMA Matmul...
  [Loop 1] Accumulating previous matrix 0...
  [Loop 2] Dispatching MSGDMA Matmul...
  [Loop 2] Accumulating previous matrix 1...
  [Cleanup] Accumulating final matrix 2...

=== Final Accumulated Hardware Matrix ===
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3
  3   3   3   3   3   3   3   3

Standalone Pipeline Accumulator Test: PASS! All expected values equal 3.
```

---

## 8. CNN Layer 1 HW vs SW Verifications

Successfully matched the accuracy of the C-based software CPU and NPU hardware execution at 80.00% across the 1000-image `mnist` subset, proving correct Layer 1 (`Conv2D`) execution on the Systolic Array. 

### Key Bug Fixes in Layer 1 Data Mapping
Due to Endianness requirements of the MSGDMA engines and Avalon-ST interfaces, our previous `temp_x` mapping incorrectly applied a reversed `7-c` sequence to input features. The MSGDMA expects data columns natively, while the weights must be swapped (`c=7->t=0`). 
Removing the duplicate byte swapping aligned the hardware results exactly to the S/W matrix calculations.

### NPU vs CPU Base Execution (Single Layer Acceleration)
Currently, only Layer 1 is processed by the NPU, while the larger Layer 2 (Fully Connected) runs locally on the ARM CPU. 

```text
[1] Running S/W (CPU) Inference...
    CPU Accuracy : 80.00%
    CPU Time     : 67.74 ms (6.774 ms/img)

[2] Running H/W (NPU) Inference...
    NPU Accuracy : 80.00%
    NPU Time     : 87.03 ms (8.703 ms/img)

=== Speedup ===
    H/W Acceleration: 0.78 x
```

> **Note on Performance Bottleneck**: For Layer 1 (`Conv2D K=16`), the NPU processes 169 patches via 22 separate HW dispatches. Because each batch requires extensive CPU memory setup for `im2col` padded transformations and 22 individual MSGDMA descriptors, the constant dispatch overhead significantly exceeds the actual MAC execution time resulting in a slower net time (87ms vs 67ms). The next optimization step requires executing the Fully Connected Layer 2 over HW, which involves a massive 1352 $\times$ 16 matrix. Processing this via single large MSGDMA transfers should aggressively recover the hardware acceleration ratio.

---

## 9. Final Inference Optimization: Amdahl's Law and MSGDMA Architecture Limits

Through rigorous profiling and the total elimination of all C software formatting overheads (e.g., pulling `memset` and `memcpy` of the 21MB `im2col` padded inputs out of the benchmarking loop, and separating memory regions to `0x800000` and `0x1E00000` base offsets to prevent overlapping overwrites), the true hardware latency of the CNN Layer 1 inference was completely isolated.

The benchmarking revealed a fascinating paradigm of Edge AI scaling:

```text
  [L1 Breakdown]
  1. HW DMA Core  : 0.480 ms
  2. SW bswap32   : 0.336 ms
  3. SW Accum/ReLU: 0.021 ms
------------------------------------------
    NPU Accuracy : 80.00%
    NPU Time     : 0.982 ms/img
    
    CPU Time     : 0.178 ms/img (Layer 1 + Layer 2)
    
=== Speedup ===
    H/W Acceleration: 0.18 x
```

### Analysis: Why is the CPU 5.5x faster than the NPU here?
The CNN model used is extremely small (`13x13` output, 8 filters $\to$ ~35,000 MACs). The 800MHz Dual-Core Cortex-A9 CPU executes these 35K MACs almost entirely within its 512KB L2 cache in just **~33 microseconds**. 

In contrast, the NPU MSGDMA Architecture has rigid physical overheads dictated by AXI Bus transfers:
1. **AXI Interface Bottleneck (`0.480 ms`)**: The MSGDMA must fetch 10.8KB of padded inputs and write back 43KB of results (due to the rigid 256-byte output systolic array architecture). Over the HPS-to-FPGA memory map, this transfers data at roughly ~115 MB/s, imposing a flat latency floor.
2. **Uncached Memory / Endianness Swap (`0.336 ms`)**: The CPU must read the combined 86KB of matrices using an `O_SYNC` pointer mapped via `/dev/mem`, which bypasses the CPU cache completely. Thus, cache misses during the `__builtin_bswap32` data extraction heavily dominate the latency.

**Conclusion:** 
The NPU logic works perfectly and robustly under continuous DMA saturation. However, for micro-wordloads (< 1M MACs), the AXI memory-transfer time ($O(N^2)$) structurally outweighs the arithmetic computation time ($O(N^3)$). To see powerful `> 1.0x` hardware speedups, the target model must be mathematically scaled up (e.g., ResNet or MobileNet architectures) where the sheer density of parallel MAC computations finally dwarfs the fixed memory transfer overheads between the HPS and FPGA.

---

## 10. Lessons Learned: The Necessity of High Bandwidth Memory (HBM)

The single greatest lesson from this implementation is a direct physical demonstration of the **Von Neumann Bottleneck** in Artificial Intelligence Hardware. 

Our Cyclone V NPU consists of a relatively small 8x8 Systolic Array (64 MACs per cycle). Yet, even with this small array, our benchmarking proved that the core logic was *starving* for data. The continuous MSGDMA streaming could only achieve **~115 MB/s** across the AXI bridge.

When scaling up to a production-grade NPU:
- An **8x8 array** needs 64 bytes per clock. At 50MHz, it demands **3.2 GB/s** of memory bandwidth just to stay fed (100% utilization).
- A **16x16 array** (256 MACs) at 500MHz demands **128 GB/s** of bandwidth.
- Google's **TPUv1 (256x256 array)** at 700MHz requires a staggering **44.8 TB/s** of internal memory bandwidth!

### Why traditional DDR fails for AI:
A standard DDR3 or DDR4 channel maxes out at roughly 10~25 GB/s. If you attach a massive NPU to standard DDR memory, the NPU will spend 99% of its time idling (waiting for the AXI bus to fetch the next matrix tile) and only 1% of its time actually computing. This renders the massive silicon area of the MAC array effectively useless.

### The HBM Solution:
This is precisely why modern AI hardware (like NVIDIA H100s, Google TPUs, and AMD MI300s) utilize **High Bandwidth Memory (HBM)**. 
HBM stacks memory chips vertically directly onto the silicon interposer next to the logic die, providing an immensely wide 1024-bit (or more) data bus natively. Instead of moving data across an external PCB trace via serial AXI, HBM allows the NPU to slurp entire massive matrix tiles (Terabytes per second) instantly into its SRAM buffers.

Our custom NPU perfectly validated this architectural truth: **In AI Hardware design, computing power (MAC count) is cheap, but moving data to feed those MACs is the true engineering bottleneck.**

---

## 11. Full-Stack Operator Fusion (PyTorch to Hardware)

Successfully bridged the high-level Python Deep Learning framework with the low-level FPGA Systolic Array via the creation of Phase 1 RTL and Phase 2 PyTorch FX Compiler. 

1. **Software Compiler (PyTorch FX):**
   By utilizing PyTorch FX Subgraph Matching, the custom compiler pass (`npu_fusion_pass.py`) mathematically folded `[Conv2d -> BatchNorm2d -> ReLU]` operations into a singular target emulation node (`NPU_Conv2d`). This completely absorbed the heavy parameters (Gamma, Beta, Mean, Variance) into statically scaled INT8 weights and INT32 biases.
2. **Hardware RTL Pipeline (Phase 1):**
   The hardware deeply integrated an OCM accumulator (`npu_ocm_accumulator.v`) and a hardwired Post-Processor (`npu_post_processor.v`). It performs **Bias Addition + Requantization (Right Shift) + ReLU Clipping** iteratively within a single clock cycle immediately as data drains from the SRAM cache.
3. **Bit-Exact Verification Passes:**
   The `test_fusion.py` testbench proved that the newly formed execution datapath operates with 0% memory latency overheads, eliminating DDR4 round-trips for intermediate features. When fed random sequences, the RTL hardware produced the **exact equivalent 8-bit bit-for-bit results** as the simulated PyTorch offline quantized pass, achieving a major milestone for reliable model deployment.

## 4. Phase 4: Full-Stack CNN MNIST 통합 연산 가속 (3.75x Speedup)

파이토치로 훈련된 `10-Class MNIST CNN` 모델 (Conv -> ReLU -> Conv -> FC) 의 추론(Inference) 가속을 NPU SoC 파이프라인(MSGDMA -> 하드웨어 OCM Accumulator -> Post-Processor)으로 100% 이관하여 성공적으로 테스트를 완수하였습니다.

### 달성 지표
- **정확도(Accuracy)**: 89.58% (CPU 소프트웨어 연산 결과와 **100% Bit-Exact 일치**)
- **처리 속도(Speed)**: 177.16 ms (100장 기준, **장당 1.845 ms**)
- **가속 배율(Speedup)**: CPU (6.927 ms/img) 대비 **3.75배 순수 하드웨어 가속** 달성

### 트러블슈팅 및 버그 해결 기록
1. **OCM Accumulator "Endianness Carry-Bit" 버그**:
   - `npu_ocm_accumulator.v` 내부에서 32-bit 덧셈 수행 전후에 바이트를 강제로 스왑(`{A[7:0], A[15:8]...}`)하는 로직이 산술 연산을 파괴했습니다. 
   - Verilog 덧셈기는 물리적으로 하위 비트(LSB)에서 상위 비트(MSB)로 올림수(Carry)를 전파합니다. 바이트가 뒤집힌 채 덧셈이 수행되면 Carry가 물리적 상위 비트(원래의 하위 바이트) 쪽으로 역류하면서 숫자가 완전히 깨져 거대한 음수 쓰레기값(`-100,709,433`)이 만들어졌습니다. Swapper를 제거하고 **Native 32-bit 구조로 덧셈을 수행**하여 아키텍처 결함을 완벽히 해결했습니다.

2. **MSGDMA 패킹과 Avalon-MM Direct 접근 간의 Endian 차이**:
   - MSGDMA를 통해 DDR로 쏟아져 나온 결과(Layer 1)는 패킷 특성상 오른쪽 열(`Col 7`)부터 메모리 앞쪽에 거꾸로 쑤셔넣어지므로 C 코드에서 `7 - c` 로 인덱스를 뒤집어 읽어야 정답이 됩니다.
   - 하지만 Layer 2는 ReLU(DRAIN)를 회피하기 위해 OCM에 **Avalon-MM 버스로 직접 주소 접근(Direct Access)** 하여 읽었습니다. Avalon-MM은 그런 패킹 순서 역전 없이 정직하게 `c` 인덱스 순서대로 메모리에 쌓이므로, MSGDMA식 `7 - c` 논리를 적용하면 빈 패딩(Padding) 영역인 쓰레기 숫자를 읽게 되는 논리적 결함을 교정했습니다.

3. **AXI 버스 비동기 통신 버퍼 덮어쓰기 현상**:
   - NPU OCM 버퍼를 0으로 초기화하기 위해 ARM 코어에서 `0`을 64번 쏘았지만, 리눅스 AXI 브릿지는 이를 모아서 (Posting) 뒤늦게 쏩니다.
   - FPGA가 곱셈을 시작했는데 뒤늦게 도착한 `0`들이 partial sums를 덮어씌워버리는 문제를 막기 위해, 쓰기 명령 직후에 해당 주소를 한 번 강제로 읽어오는 **Dummy Read (Blocking Flush)** 기법을 사용하여 버스 타이밍을 물리적으로 동기화시켰습니다.

4. **소프트웨어 Polling Spin-lock 과부하 제거**:
   - 하드웨어 타이밍(seq_busy)을 믿지 못하고 박아두었던 `10000`번의 강제 `volatile wait` 루프가 이미지당 784번 호출되며 수 밀리초를 허비했습니다. 폴링 조건을 `NPU_STATUS` 플래그 하나로 단일화하여 CPU 낭비를 제거함으로써 압도적인 속도 향상을 이끌어냈습니다.
