# NPU Architecture Design Document

이 문서는 DE10-Nano FPGA 상에 구현된 **Systolic NPU**의 하드웨어 아키텍처와 소프트웨어 제어 설계를 상술합니다.

---

## 1. System Overview

NPU는 ARM HPS (또는 NIOS II 소프트 코어)와 FPGA 간의 **Avalon-MM 브릿지**를 통해 연결됩니다. 제어는 Avalon-MM Slave 포트를 통해 `npu_ctrl` 레지스터 맵에 접근하며 이루어지고, 대용량 데이터 이동은 NPU 내부의 독립적인 **DMA Read/Write Master** 엔진을 통해 버스 마스터링 방식으로 수행됩니다.

### 1.1 계층 구조 (Hierarchy)
- **`npu_unit.v`**: NPU 최상단 통합 파이프라인.
  - **`npu_ctrl.v`**: 중앙 컨트롤러. Avalon-MM Slave 인터페이스 파싱, CSR 통제.
  - **`npu_dma.v`**: 메모리-NPU 간의 데이터 이동 전담.
  - **`npu_sequencer.v`**: 제어 상태 머신(FSM). DMA 데이터를 바탕으로 Systolic Array에 먹일 타이밍 및 텐서 형상을 관리.
  - **`systolic_core.v`**: 핵심 연산기 래퍼(Wrapper).
    - `systolic_array_8x8.v`: 8x8 MAC 배열 (총 64개의 곱셈-누산기).

---

## 2. Hardware Design

### 2.1 Decoupled DMA & Dual FIFO Architecture
초기 구조의 병목과 교착 상태(Deadlock)를 해결하기 위해 **데이터 스트림 경로를 완전히 분리**했습니다.
- **Read FIFO / Write FIFO**: 메모리에서 NPU로 들어오는 `in_fifo` (Memory -> NPU)와 계산 결과가 나가는 `out_fifo` (NPU -> Memory)가 물리적으로 512-word 깊이로 분리되어 있습니다.
- **Full-Duplex Data Streaming**: NPU 연산 시 입력 데이터 수신과 동시에 이전 사이클의 계산 결과를 메모리로 배출할 수 있어 100%의 연산기 가동률을 달성합니다.

### 2.2 Systolic Array Pipeline (Skew/Deskew)
연산기(`systolic_array_8x8`) 내부 데이터 흐름은 **Systolic** 방식(맥박처럼 옆으로 한 칸씩 이동)을 따릅니다.
- **Input Skew Buffer**: 배열 왼쪽으로 들어오는 각 행(Row) 데이터를 1클럭씩 계단식 지연시킵니다 (0번 행은 0딜레이, 7번 행은 7딜레이). 이 처리를 통해 대각선으로 데이터 웨이브가 맞춰져 올바른 행렬곱 연산이 일어납니다.
- **Output Deskew Buffer**: 배열 아래쪽으로 출력되는 계산 결과(Partial Sum) 역시 대각선 형태로 나오므로, 역방향 계단식 지연을 통해 메모리로 내보낼 때는 완전한 하나의 열(Column) 묶음으로 정렬(Alight)시켜 줍니다.

---

## 3. Register Map (Word-Addressing)

NPU는 4-bit 워드 단위 주소를 사용하며, Qsys의 인터페이스 속성(`addressUnits=WORDS`)과 맞춰져 있습니다.

| Word Offset | Name | Description |
|:--- |:--- |:--- |
| `0x0` | `REG_CTRL` | [1] Start, [2:1] Mode (0: Load Weight, 1: Execution) |
| `0x1` | `REG_STATUS`| [17:16] DMA Done Status Flags |
| `0x2` | `REG_DMA_RD_ADDR`| DMA Source (메모리 입력) 물리 주소 |
| `0x3` | `REG_DMA_RD_LEN`| 읽어올 총 Word 개수 |
| `0x4` | `REG_DMA_WR_ADDR`| DMA Destination (결과 저장) 물리 주소 |
| `0x5` | `REG_DMA_WR_CTRL`| [15:0] Write Length, [16] Start DMA RD, [17] Start DMA WR |
| `0x6` | `REG_SEQ_ROWS`| 수행할 총 행렬 행 개수 |

> [!NOTE]
> `0x8 ~ 0xB` 오프셋에는 하드웨어 검증을 위해 격리된 구형 싱글 `mac_pe` 제어 레지스터들이 레거시 형태로 남아있습니다.

---

## 4. Initialization & Execution Sequence

C(NIOS II) / Python(Cocotb) 펌웨어에서의 표준 제어 흐름은 다음과 같습니다.

1. **설정 기록**: 
   - `REG_SEQ_ROWS`, `REG_DMA_RD_ADDR`, `REG_DMA_RD_LEN`, `REG_DMA_WR_ADDR` 에 값 등록
2. **NPU Sequencer 구동**: 
   - `REG_CTRL` 에 `0x3` (모드 1, 시작 플래그 세트) 쓰기.
3. **DMA 동시 구동**: 
   - `REG_DMA_WR_CTRL` 에 `length | (1<<17) | (1<<16)` 값 쓰기.
   - 이때부터 하드웨어 버스 마스터가 개입하여 자동으로 메모리 R/W 를 수행합니다.
4. **대기 및 검증**:
   - `REG_STATUS` 를 폴링(Polling)하며 마스크 `0x00030000` 비트가 모두 1이 될 때까지 대기합니다.
   - 완료 후 DMA WR 물리 주소 영역에서 연산된 결과를 읽습니다.
