# NPU Output Duplication Issue & Architecture Change

## Issue Description
- **8x8 NPU Implementation**: The simulation (cocotb) passed successfully, but hardware execution failed due to an output duplication issue.
- **4x4 NPU Scale-down**: To isolate the problem (suspected to be a bus issue), the NPU was scaled down to a 4x4 architecture. However, the output duplication issue (`[1, 2, 3, 3]`) still persisted in hardware.

## Architecture Change Decision
Since the output duplication issue persists regardless of the NPU scale (8x8 or 4x4) and is isolated to the Avalon-MM bus/DMA handshake during hardware execution, the architecture will be heavily revised. 

Instead of relying on the current Avalon-MM DMA-based design, the architecture will transition to an **Avalon Streaming (Avalon-ST)** interface. This change aims to simplify the data path, providing a more robust and predictable data flow between the NPU and memory, thereby eliminating the handshake/timing mismatches causing data duplication.

## Resolution (Completed)
The migration to the decoupled Avalon-ST architecture (via `npu_stream_ctrl` and elastic valid/ready pipelining) was a complete success. The data duplication issues completely vanished. Both the 4x4 and the final 8x8 NPU configurations now correctly execute continuous hardware sequences and pass 100% of the Linux C-driver memory validation checks under MSGDMA load without repeating any output elements.
