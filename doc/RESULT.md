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

Measures the absolute execution time of equivalent 8x8 matrix multiplications running on the 800MHz ARM Cortex-A9 CPU (`gcc -O2` via nested loops) versus the 50MHz FPGA NPU (via MSGDMA Pipeline) using parameterized target sizes. Mapped using `gettimeofday()`.

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
