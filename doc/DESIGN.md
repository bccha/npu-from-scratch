# NPU Architecture Design Document

이 문서는 DE10-Nano FPGA 상에 구현된 **Multi-core Systolic NPU**의 하드웨어 아키텍처와 소프트웨어 제어 설계를 상술합니다.

---

## 1. System Overview

NPU는 ARM HPS와 FPGA 간의 **Lightweight HPS-to-FPGA Bridge (Avalon-MM)**를 통해 연결됩니다. ARM Linux 커널의 `mmap`을 통해 사용자 공간(User-space)에서 직접 가속기 레지스터에 접근하여 연산을 오프로드(Offload)합니다.

### 1.1 계층 구조 (Hierarchy)
- **`npu_ctrl.v`**: 메인 컨트롤러. Avalon-MM 슬레이브 인터페이스 및 주소 디코딩, 멀티코어 관리 담당.
- **`npu_core.v`**: 개별 NPU 연산 코어 (4개 탑재).
  - `mac.v`: 8-bit Signed MAC 유닛.
  - `systolic_dot4.v`: 1x4 내적 연산 유닛.
  - `systolic_4x4.v`: 4x4 Systolic Array 행렬곱 엔진.

---

## 2. Hardware Design

### 2.1 Multi-core Architecture (4x4 x 4)
단일 대형 어레이 대신 **4개의 독립된 4x4 코어**를 사용하는 분산형 구조를 채택했습니다.
- **병렬성**: 4개의 코어에 서로 다른 타일(Tile)을 동시에 할당하여 병렬 처리 가능 ($64$ MACs/cycle).
- **안정성**: 코어 단위 모듈화를 통해 하드웨어 배선 복잡도를 낮추고 타이밍 마진을 확보.

### 2.2 Fully Synchronous Trigger
모든 제어 신호는 클럭에 동기화되어 동작합니다.
- **Registered Trigger**: 버스 쓰기 발생 1클럭 뒤에 내부 트리거(`start`, `valid`)가 발생하여 데이터 레지스터 안정성을 보장합니다.
- **Sticky Status**: 연산 완료 신호는 소프트웨어가 새로운 연산을 시작할 때까지 유지(Latch)되어, 고속 동작에서도 소프트웨어가 상태를 놓치지 않도록 설계되었습니다.

---

## 3. Register Map

전체 128워드(512바이트) 주소 공간을 사용하며, 각 코어는 32워드씩 점유합니다.

| Offset (Hex) | Name | Description |
|:--- |:--- |:--- |
| `0x00 / 0x20 / 0x40 / 0x60` | `REG_CTRL` | [3:2] Mode (MAC/Dot4/Mat44), [1] Valid, [0] Start |
| `0x01 / 0x21 / 0x41 / 0x61` | `REG_STATUS`| [0] Done (Sticky bit) |
| `0x02 / 0x22 / 0x42 / 0x62` | `REG_A_DATA`| {a3, a2, a1, a0} 8-bit inputs |
| `0x03 / 0x23 / 0x43 / 0x63` | `REG_B_DATA`| {b3, b2, b1, b0} 8-bit inputs |
| `0x04 / 0x24 / 0x44 / 0x64` | `REG_ACC_IN`| 32-bit Accumulator Input |
| `0x05 / 0x25 / 0x45 / 0x65` | `REG_RES`   | MAC/Dot4 Result |
| `0x10 ~ 0x1F / ...` | `REG_C_BASE`| 4x4 MatMul Result Registers (16 words) |

---

## 4. Software Strategies

### 4.1 Matrix Tiling
9x9나 100x100과 같이 NPU 어레이 크기(4x4)보다 큰 행렬은 **Tiling(타일링)** 기법으로 처리합니다.
- **Padding**: 행렬을 4의 배수 크기로 Zero-padding 처리.
- **Sub-block multiplication**: $C_{ij} = \sum (A_{ik} \times B_{kj})$ 공식을 기반으로 4x4 조각들을 순차적으로 NPU에 할당.

### 4.2 Multi-threaded Acceleration
ARM Linux의 `pthreads` 라이브러리를 사용하여 4개의 NPU 코어에 작업을 분산합니다.
- 각 스레드는 전담 NPU 코어의 레지스터를 제어하며, 하드웨어 연산 대기 시간 동안 컨텍스트를 유지하여 병렬 효율을 극대화합니다.

---

## 5. Implementation Notes
- **Precision**: 8-bit signed integer inputs, 32-bit accumulator.
- **Clock**: Avalon-MM System Clock (50MHz).
- **Interface**: Registered Read Data Path (1-cycle read latency).
