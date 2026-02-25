# NPU Architecture Design Document

## Avalon-MM to Avalon-ST Transition

### 1. System Overview

NPU는 ARM HPS (또는 NIOS II 소프트 코어)와 FPGA 간의 **Avalon-MM 브릿지**를 통해 연결됩니다. 제어는 Avalon-MM Slave 포트를 통해 `npu_ctrl` 레지스터 맵에 접근하며 이루어지고, 대용량 데이터 이동은 NPU 내부의 독립적인 **Avalon ST Read/Write Master** 엔진을 통해 버스 마스터링 방식으로 수행됩니다.

### 2. Avalon-ST Data Path 및 Bufferless 설계

Avalon-ST (Streaming) 인터페이스는 `valid`와 `ready` 신호를 통한 엄격한 핸드셰이크(Handshake)를 지원합니다. 
따라서 NPU 내부(RTL)에는 과거 Avalon-MM 구조처럼 별도의 거대한 SRAM 버퍼를 둘 필요가 없습니다.

- **입력 (Sink)**: MSGDMA 내부에 이미 Data Path FIFO(예: Depth 32~128)가 존재하므로, 이 FIFO가 메모리에서 오는 데이터의 불규칙한 지연(Latency)을 흡수합니다. NPU는 `valid`가 1일 때만 데이터를 소모하며, 내부 연산기가 꽉 차면 `ready`를 0으로 내려 MSGDMA의 전송을 멈출 수 있습니다 (배압, Backpressure).
- **출력 (Source)**: NPU 연산 결과가 도출될 때마다 `valid`를 1로 올려 다음 단계(출력용 MSGDMA)로 밀어냅니다. 

결과적으로 NPU 코어는 거대한 버퍼 관리 로직 없이, 스트림으로 들어오는 데이터를 즉각적으로 MAC 어레이(Shift Register 등)에 밀어넣는 **순수 연산 중심의(Bufferless/Streaming-friendly) 구조**로 단순화됩니다.

### 3. MSGDMA (Modular Scatter-Gather DMA) 통합

데이터의 흐름은 Qsys(Platform Designer)에 내장된 MSGDMA 인스턴스를 통해 양손(Read/Write)으로 제어됩니다.
- **Read MSGDMA (Memory-Mapped to Streaming)**: DDR 메모리에서 데이터(예: 8x8 행렬 64바이트)를 읽어 `npu_stream_ctrl`의 `st_sink` 포트로 쏟아냅니다.
- **Write MSGDMA (Streaming to Memory-Mapped)**: NPU에서 연산이 끝난 결과를 `st_source` 포트에서 받아 DDR 메모리에 기록합니다.
- **제어(Control)**: 두 MSGDMA의 `csr` 및 `descriptor_slave` 포트는 Nios II (또는 HPS)의 데이터 마스터에 연결되어 있어, 메인 CPU가 디스크립터를 작성해 넣는 것만으로 대규모 데이터 전송을 지시할 수 있습니다.

### 4. 256-bit to 64-bit Serialization (직렬화 로직) 및 Batch EOP 처리

8x8 시스톨릭 어레이(`systolic_core`)는 연산 완료 시 한 번에 256-bit (32비트 x 8개)의 결괏값을 병렬로 출력합니다. 
반면 시스템(MSGDMA)으로 나가는 Avalon-ST Data 버스 폭은 64-bit이므로, 이를 안전하게 쪼개서 보내는 직렬화(Serialization) FSM이 `npu_stream_ctrl` 내부에 구현되어 있습니다.

- NPU 결과가 나올 때 256비트를 레지스터에 캡처합니다.
- `st_source_ready` 신호와 동기화하여 정확히 4클럭 동안 64비트씩 잘라서 전송합니다.
- 첫 전송 시 `startofpacket`(SOP) 플래그를 발생시킵니다.
- **연속 트랜잭션 (Batch Streaming) 보장**: MSGDMA가 중간에 조기 종료되는 것을 막기 위해 `npu_ctrl` 레지스터에서 받은 `REG_SEQ_ROWS` 값을 기준으로 전체 배치 크기를 파악하고, 단일 행렬이 아닌 **가장 마지막 배치의 마지막 행 결괏값을 전송할 때만 `endofpacket`(EOP) 플래그를 발생시킵니다**.
- 이 전송이 이루어지는 4클럭 동안은 시스톨릭 어레이의 내부 레이턴시(Pipelines) 구간이므로 데이터 병목(Stall)이나 데이터 유실, 출력 중복 버그가 하드웨어적으로 원천 차단됩니다.

### 5. Automated Execution Flow (자동화된 파이프라인 흐름)

과거처럼 FSM이 상태를 하나하나 전이시키며 대기할 필요 없이, 아래의 소프트웨어 제어만으로 NPU 파이프라인이 자동 실행됩니다.

1. **Batch 사이즈 등록**: `REG_SEQ_ROWS`에 CPU가 처리할 전체 행(Row) 개수(예: 행렬 100개 * 8)를 입력해 EOP 기준점을 세팅합니다.
2. **Write DMA 세팅**: CPU가 Write용 MSGDMA의 디스크립터에 "NPU 결과를 DDR B 주소에 저장해라"라고 지시합니다. (DMA 대기 상태 돌입)
3. **Read DMA 세팅**: CPU가 Read용 MSGDMA의 디스크립터에 "DDR A 주소에서 NPU로 데이터를 쏴라"라고 지시합니다. (전송 시작!)
4. **하드웨어 체인 리액션**: Read DMA가 데이터를 밀어 넣으면, `npu_stream_ctrl`을 거쳐 시스톨릭 코어가 연산하고, 결과가 다시 `npu_stream_ctrl`의 Serializer를 통해 Write DMA로 전달되어 메모리에 꽂힙니다. 이 모든 과정이 `valid` / `ready` 배압(Backpressure)을 통해 한 치의 오차 없이 자동 수행됩니다.

### 6. 시스템 아키텍처 다이어그램 (Mermaid)

```mermaid
graph TD
    %% Define HPS Subsystem
    subgraph HPS ["ARM HPS / NIOS II"]
        CPU[메인 프로세서]
        DDR[(DDR3 메모리)]
    end

    %% Define Qsys (Platform Designer) Bridges
    subgraph Qsys ["Platform Designer Interconnect"]
        AXI_LW[Avalon-MM Lightweight Bridge]
        AXI_DAT[Avalon-MM Data Bridge]
        
        ReadDMA[MSGDMA (Read)]
        WriteDMA[MSGDMA (Write)]
    end

    %% Define NPU Subsystem
    subgraph NPU ["npu_unit (FPGA Fabric)"]
        NPU_CTRL[npu_ctrl <br> CSR Registers]
        
        subgraph STREAM_CTRL ["npu_stream_ctrl"]
            SINK_FSM[Sink Controller <br> 64-bit I/F]
            SER_FSM[Serializer FSM <br> 256 to 64-bit]
        end
        
        MAC_ARRAY[8x8 Systolic Core <br> 64 MAC PEs]
    end

    %% Control Flow (Memory Mapped)
    CPU -. "제어/상태 확인" .-> AXI_LW
    AXI_LW -. "CSR 포트 접근" .-> NPU_CTRL
    AXI_LW -. "Descriptor 포트" .-> ReadDMA
    AXI_LW -. "Descriptor 포트" .-> WriteDMA

    %% Data Flow (Streaming / Memory Mapped)
    DDR <== "Avalon-MM (Read)" ==> AXI_DAT
    AXI_DAT ==> ReadDMA
    WriteDMA ==> AXI_DAT
    AXI_DAT <== "Avalon-MM (Write)" ==> DDR

    %% High-Speed Avalon-ST Streams
    ReadDMA == "Avalon-ST <br> 64-bit (Input Data)" ==> SINK_FSM
    SINK_FSM == "64-bit 스트림" ==> MAC_ARRAY
    MAC_ARRAY == "256-bit 병렬 결과" ==> SER_FSM
    SER_FSM == "Avalon-ST <br> 64-bit 직렬화 스트림" ==> WriteDMA
```
