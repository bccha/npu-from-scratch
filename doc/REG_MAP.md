# NPU System Register Map

이 문서는 NPU 제어 레지스터(`npu_ctrl`)와 Altera MSGDMA (Avalon-ST DMA)의 CSR 및 Descriptor 레지스터 맵을 설명합니다.

## 개요 (Overview)

NPU 및 DMA의 메모리 맵(Memory-mapped) 제어 레지스터는 시스템의 CPU에 따라 접근 방식이 다릅니다:

1.  **NIOS II 프로세서 환경 (Baremetal/RTOS):**
    *   Qsys(Platform Designer)에서 시스템을 생성할 때, BSP 생성 도구가 `system.h` 헤더 파일을 자동 생성합니다. 
    *   이 파일에는 각 하드웨어 IP의 매핑된 Base Address가 매크로 상수로 정의됩니다 (예: `NPU_CTRL_0_BASE`, `MSGDMA_0_CSR_BASE`).
    *   개발자는 제공되는 `IORD()`, `IOWR()` 매크로 함수에 대상의 Base Address 매크로를 넘겨주어 별도의 포인터 맵핑 연산 없이 바로 레지스터 단위로 데이터를 읽고 쓸 수 있습니다.

2.  **ARM (HPS) 프로세서 환경 (Linux):**
    *   ARM HPS 코어는 LWH2F (Light-Weight HPS-to-FPGA) Bridge를 거쳐 FPGA 내부의 Avalon-MM 슬레이브 장치들(NPU, DMA 등)에 접근합니다.
    *   Linux와 같은 운영체제 환경에서는 보안 및 메모리 가상화로 인해 물리 주소(Physical Address) 범위인 브릿지 주소에 직접 접근할 수 없습니다.
    *   따라서 사용자 공간(User Space) 앱에서는 `/dev/mem` 디바이스 드라이버를 열고, `mmap()` 시스템 콜을 사용하여 LWH2F Bridge의 주소 영역(`0xFF200000`)을 애플리케이션의 가상 메모리 주소(Virtual Address) 영역으로 맵핑(Virtual memory mapping)합니다.
    *   맵핑되어 반환된 가상 Base 변수 포인터에 IP별 Offset(예: `NPU_CTRL_OFFSET`)을 더하는 방식으로 각각의 가상 하드웨어 포인터 주소를 구한 뒤 접근합니다.

---

## 1. NPU Control Register (`npu_ctrl.v`)

Base Address: `NPU_CTRL_BASE` 
(주소 단위는 Word(4 Byte)이며, 32-bit 데이터 폭을 가집니다. `Address[3] == 0`일 때 System Register 영역에 접근합니다.)

| Byte Offset | Word Addr | Name | Access | Bits | Description |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **0x00** | 0x0 | `SEQ_CTRL` | R/W | [0] | `seq_start` (W, Command) |
| | | | | [2:1] | `seq_mode` (RW) - 0: Weight Load, 1: Execution |
| **0x04** | 0x1 | `SEQ_STATUS` | R | [0] | `seq_busy` |
| | | | | [1] | `seq_done` |
| **0x08** ~ **0x14** | 0x2 ~ 0x5 | *(Reserved)* | - | - | *(구 내부 DMA 제어/상태 레지스터, Avalon-ST MSGDMA 전환으로 삭제됨)* |
| **0x18** | 0x6 | `SEQ_TOTAL_ROWS`| R/W | [31:0] | `seq_total_rows` |
| **0x1C** | 0x7 | `WEIGHT_LATCH_EN`| R/W | [0] | `weight_latch_en` |

*(참고: `Address[3] == 1` 즉 Byte Offset `0x20 ~ 0x3C` 영역은 Legacy MAC PE 제어 인스턴스 `mac_pe_ctrl` 에 할당되어 있습니다.)*

---

## 2. MSGDMA (Avalon-ST) CSR Register Map

Base Address: `MSGDMA_CSR_BASE`

| Byte Offset | Name | Access | Bits | Description |
| :---: | :--- | :---: | :--- | :--- |
| **0x00** | `STATUS` | R /<br>Clr | [0] | Busy |
| | | | [1] | Descriptor Buffer Empty |
| | | | [2] | Descriptor Buffer Full |
| | | | [3] | Response Buffer Empty |
| | | | [4] | Response Buffer Full |
| | | | [5] | Stop State |
| | | | [6] | Reset State |
| | | | [7] | Stopped on Error |
| | | | [8] | Stopped on Early Termination |
| | | | [9] | IRQ (1 쓰기 시 Clear) |
| **0x04** | `CONTROL` | R/W | [0] | Stop |
| | | | [1] | Software Reset (전체 DMA 리셋) |
| | | | [2] | Stop on Error |
| | | | [3] | Stop on Early Termination |
| | | | [4] | Global Interrupt Enable Mask |
| | | | [5] | Stop dispatcher |
| **0x08** | `DESCRIPTOR_FILL_LEVEL` | R | [15:0] | Read Fill Level |
| | | | [31:16]| Write Fill Level |
| **0x0C** | `RESPONSE_FILL_LEVEL` | R | [15:0] | Response Fill Level |
| **0x10** | `SEQUENCE_NUMBER` | R/W | [15:0] | Sequence Number (Enhanced Feature Only) |

---

## 3. MSGDMA (Avalon-ST) Descriptor Map

Base Address: `MSGDMA_DESCRIPTOR_BASE`

### Standard Format

| Byte Offset | Name | Description |
| :---: | :--- | :--- |
| **0x00** | `READ_ADDRESS` | Read Address [31:0] |
| **0x04** | `WRITE_ADDRESS` | Write Address [31:0] |
| **0x08** | `LENGTH` | Length [31:0] |
| **0x0C** | `CONTROL_STANDARD` | Control Register [31:0] |

**Control Register 주요 비트 (Standard / Enhanced 공통):**
* `[7:0]` : Transmit Channel
* `[8]` : Generate SOP (Start of Packet 포함 전송)
* `[9]` : Generate EOP (End of Packet 포함 전송)
* `[10]` : Park Reads
* `[11]` : Park Writes
* `[12]` : End on EOP
* `[14]` : Transfer Complete IRQ 발생 활성화
* `[15]` : Early Termination IRQ 발생 활성화
* `[23:16]` : Error IRQ Mask
* `[24]` : Early Done Enable
* `[31]` : Go (이 비트에 1을 쓰면 Descriptor가 Queue(Dispatcher)로 Commit 됩니다.)

### Enhanced Format (Extended)

| Byte Offset | Name | Description |
| :---: | :--- | :--- |
| **0x00** | `READ_ADDRESS` | Read Address [31:0] |
| **0x04** | `WRITE_ADDRESS` | Write Address [31:0] |
| **0x08** | `LENGTH` | Length [31:0] |
| **0x0C** | `BURST_SEQ` | `[15:0]` Sequence Number <br> `[23:16]` Read Burst Count <br> `[31:24]` Write Burst Count |
| **0x10** | `STRIDE` | `[15:0]` Read Stride <br> `[31:16]` Write Stride |
| **0x14** | `READ_ADDRESS_HIGH` | Read Address [63:32] |
| **0x18** | `WRITE_ADDRESS_HIGH`| Write Address [63:32] |
| **0x1C** | `CONTROL_ENHANCED` | Control Register [31:0] |

---

## 4. NIOS II 사용 예제 (C 언어)

다음은 NIOS II 환경의 `system.h`에 정의된 매크로 상수를 활용하여, NPU와 MSGDMA를 통해 Weight(가중치)를 로드하고 Stream 데이터 전송을 시작하는 예제 코드입니다.

```c
#include "system.h"
#include "io.h"  // IOWR_32DIRECT, IORD_32DIRECT 등 

// system.h 에 정의되었다고 가정하는 컴포넌트 Base Address
// #define NPU_CTRL_0_BASE        0x1000
// #define MSGDMA_RX_CSR_BASE     0x2000  (Memory to NPU)
// #define MSGDMA_RX_DESC_BASE    0x2010
// #define MSGDMA_TX_CSR_BASE     0x3000  (NPU to Memory)
// #define MSGDMA_TX_DESC_BASE    0x3010

// NPU Control Register Offsets
#define NPU_SEQ_CTRL_REG       0x0
#define NPU_SEQ_STATUS_REG     0x4
#define NPU_SEQ_TOTAL_ROWS_REG 0x18
#define NPU_WEIGHT_LATCH_REG   0x1C

// MSGDMA Descriptor Control Flag
#define MSGDMA_DESC_GO_BIT     (1 << 31)
#define MSGDMA_DESC_GEN_SOP    (1 << 8)
#define MSGDMA_DESC_GEN_EOP    (1 << 9)

void run_npu_inference(uint32_t *weight_addr, uint32_t *input_stream_addr, uint32_t *output_stream_addr, uint32_t in_bytes, uint32_t out_bytes, uint32_t rows) {
    
    // ====================================================================
    // 1. NPU 초기화 및 Weight Load 모드 진입
    // ====================================================================
    
    // seq_mode = 0 (Weight Load 모드)로 설정, start 비트는 0
    IOWR_32DIRECT(NPU_CTRL_0_BASE, NPU_SEQ_CTRL_REG, 0x00000000);

    // ====================================================================
    // 2. MSGDMA (Avalon-ST)를 통해 Weight 데이터 전송 (Descriptor 세팅)
    // ====================================================================
    // 여기서 MSGDMA(RX 채널이라 가정)는 메모리에서 Avalon-ST로 데이터를 쏘아줍니다.
    
    // Read Address 설정 (메모리 상의 가중치 버퍼 주소)
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x00, (uint32_t)weight_addr); 
    // Write Address 설정 (Avalon-ST 스트림 전송에선 대체로 쓰지 않음)
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x04, 0x00000000); 
    // 전송할 총 Byte 길이
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x08, in_bytes); 
    
    // Control Register 설정 후 Go 비트 켜서 전송(Dispatcher Commit) 시작
    // SOP(Start of Packet)나 EOP(End of Packet)가 필요하다면 OR 연산으로 추가
    uint32_t desc_ctrl = MSGDMA_DESC_GEN_SOP | MSGDMA_DESC_GEN_EOP | MSGDMA_DESC_GO_BIT;
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x0C, desc_ctrl);

    // ====================================================================
    // 3. Weight 데이터 전송 완료 대기 (DMA CSR 이용)
    // ====================================================================
    
    // MSGDMA 상태 레지스터 [0]번째 비트(Busy)가 0이 될 때까지 폴링(Polling) 대기
    while (IORD_32DIRECT(MSGDMA_RX_CSR_BASE, 0x00) & 0x1) {
        // Wait until DMA is not busy...
    }

    // NPU 내부 시프트 레지스터에 Weight가 모두 들어왔으므로, 
    // PE의 실제 가중치 레지스터로 찰칵(Latch)해서 업데이트
    IOWR_32DIRECT(NPU_CTRL_0_BASE, NPU_WEIGHT_LATCH_REG, 0x00000001);
    // (선택 사항: 하드웨어 구현에 따라 펄스로 동작한다면 직후에 0으로 내림)
    IOWR_32DIRECT(NPU_CTRL_0_BASE, NPU_WEIGHT_LATCH_REG, 0x00000000);

    // ====================================================================
    // 4. Output Stream (Write DMA) 수신 대기 설정
    // ====================================================================
    // NPU가 결과를 뱉어내기 시작하면 받을 곳이 필요하므로, 
    // Input을 쏘아주기 전에 Output을 받을 TX 채널을 반드시 먼저 세팅하고 Go 해야 합니다.
    
    // Read Address 0으로 둠 (Avalon-ST Source로부터 데이터를 받아오므로)
    IOWR_32DIRECT(MSGDMA_TX_DESC_BASE, 0x00, 0x00000000); 
    // Write Address 설정 (결과를 저장할 메모리 버퍼 주소)
    IOWR_32DIRECT(MSGDMA_TX_DESC_BASE, 0x04, (uint32_t)output_stream_addr); 
    // 수신할 총 Byte 길이
    IOWR_32DIRECT(MSGDMA_TX_DESC_BASE, 0x08, out_bytes); 
    
    // 수신 측(Write DMA)은 NPU 하드웨어(Avalon-ST Source)가 보내주는 SOP/EOP 신호를 
    // 수동적으로 받아서 패킷을 끊으므로, S/W Descriptor에 SOP/EOP 생성 비트를 켤 필요가 없습니다.
    // 대신 전송 완료 인터럽트나 폴링을 위해 Transfer Complete IRQ 비트를 켤 수 있습니다.
    uint32_t tx_desc_ctrl = (1 << 14) | MSGDMA_DESC_GO_BIT; // 14번 비트: Transfer Complete IRQ Enable
    IOWR_32DIRECT(MSGDMA_TX_DESC_BASE, 0x0C, tx_desc_ctrl); // 수신 대기 시작 (Go)
    
    // ====================================================================
    // 5. Input Stream 전송 및 NPU Inference(Execution 모드) 시작
    // ====================================================================
    
    // 처리해야 할 Row 수를 명시 (NPU Stream에서 EOP를 생성할 때 쓰임)
    IOWR_32DIRECT(NPU_CTRL_0_BASE, NPU_SEQ_TOTAL_ROWS_REG, rows);

    // seq_mode = 1 (Execution 모드)로 설정 후 seq_start = 1 비트 발생
    // (seq_mode = 1 은 [2:1] 비트이므로 1 << 1 = 0x2, seq_start = 0x1 이므로 합 0x3)
    IOWR_32DIRECT(NPU_CTRL_0_BASE, NPU_SEQ_CTRL_REG, 0x00000003);

    // Input Stream을 NPU로 보내기 위해 MSGDMA RX 채널 Descriptor 세팅
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x00, (uint32_t)input_stream_addr); 
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x04, 0x00000000); 
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x08, in_bytes); 
    IOWR_32DIRECT(MSGDMA_RX_DESC_BASE, 0x0C, desc_ctrl); // 전송 시작! (Go)

    // ====================================================================
    // 6. NPU 연산 및 Output 수신 완료 대기 
    // ====================================================================
    
    // 1) RX(입력 전송)가 끝날 때까지 대기
    while (IORD_32DIRECT(MSGDMA_RX_CSR_BASE, 0x00) & 0x1) { }
    
    // 2) TX(결과 수신)가 끝날 때까지 대기
    while (IORD_32DIRECT(MSGDMA_TX_CSR_BASE, 0x00) & 0x1) { }
    
    // 3) NPU 상태 레지스터의 [1]번째 비트(Done)가 1이 되었는지 확인 (옵션)
    while (!(IORD_32DIRECT(NPU_CTRL_0_BASE, NPU_SEQ_STATUS_REG) & 0x2)) { }
    
    // 연산 완료: output_stream_addr에 NPU 결과가 메모리에 기록 완료됨.
}
```

---

## 부록 (Appendix): 하드웨어 접근 매크로 (Hardware Access Macros)

제공된 C 코드에서는 하드웨어 레지스터에 데이터를 읽고 쓰기 위해 다음과 같은 매크로를 정의하여 사용합니다.

```c
// ============================================================================
// Hardware Access Macros
// ============================================================================
#define IOWR_32DIRECT(base, offset, data)                                      \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)                                            \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))

#define IOWR(base, reg, data)                                                  \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)) = (data))
#define IORD(base, reg)                                                        \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)))
```

### 매크로 사용 이유 및 포인터 연산 과정 설명

일반적인 시스템 메모리(RAM)의 변수에 접근하는 것과 하드웨어 제어 레지스터(Memory-Mapped I/O)에 접근하는 것은 동작 원리가 다르므로 특별한 포인터 제어가 필요합니다.

1.  **`volatile` 키워드의 필요성 (캐싱 방지):** 
    하드웨어 레지스터의 상태 값(예: DMA 완료 플래그)은 소프트웨어의 제어 흐름과 무관하게 언제든 외부(하드웨어 내부 로직)에 의해 바뀔 수 있습니다. `volatile` 키워드를 사용하지 않으면, C 컴파일러(GCC 등)는 코드 최적화 과정에서 "반복문 내에서 이 변수를 C 코드로 수정한 적이 없으니 값이 그대로겠지"라고 판단하여 값을 새로 읽지 않고 캐시(Cache)되거나 CPU 레지스터에 기억해 둔 기존 값을 끝없이 재사용하는 버그(무한 루프 등)를 발생시킵니다.
    `volatile`은 컴파일러에게 "이 주소는 물리적인 하드웨어니 최적화를 하지 말고 접근 명령을 만날 때마다 **반드시 실제 메모리 버스(Bus)를 통해 물리 주소에서 새로 읽거(Read)나 쓰거(Write)라**" 라고 강제하는 지시어입니다.

2.  **`uint8_t *` 타입 캐스팅을 통한 안전한 주소 연산:**
    C 언어에서 포인터 연산(덧셈/뺄셈)은 포인터가 가리키는 자료형의 크기에 비례합니다.
    만약 `uint32_t` 포인터인 `ptr`에 대해 `ptr + 4`를 계산하면 주솟값은 4바이트가 늘어나는 것이 아니라 `4 * sizeof(uint32_t) = 16` 바이트가 늘어나 버립니다.
    이러한 문제를 완전히 방지하기 위해 매크로는 입력받은 `base` 포인터를 무조건 `(uint8_t *)`, 즉 1 Byte 크기 자료형의 포인터로 강제 형변환(Casting) 합니다.
    *   이렇게 1 Byte 포인터로 바꾼 뒤에 `+ (offset)`을 더하면, 프로그래머가 명시한 오프셋 수치와 정확히 1:1로 일치하는 바이트 단위 메모리 주소 이동이 가능해집니다.
    *   *예시: `IOWR_32DIRECT(0x1000, 4)`를 호출하면 `0x1000 + 4 Byte = 0x1004` 번지의 주솟값이 됩니다.*

3.  **데이터 쓰기/읽기를 위한 32-bit 포인터 변환 및 역참조 (Dereferencing):**
    *   위 과정을 통해 바이트 단위 연산을 마친 최종 주솟값 `((uint8_t *)(base) + (offset))`은 여전히 1바이트를 가리키는 형태입니다.
    *   이를 우리가 다룰 32-bit 하드웨어 레지스터로 취급하기 위해 밖에서 다시 한 번 `(volatile uint32_t *)` 타입으로 형변환합니다.
    *   마지막으로, 생성된 포인터의 가장 바깥쪽에 `*( ... )` 즉, 별표(Asterisk) 연산자를 씌워줌으로써, 단지 주소를 들고 있는 것이 아니라 해당 주소 메모리 공간이 담고 있는 **실제 32-bit 값 자체(Value)**를 끄집어냅니다 (Dereferencing).
    *   이를 통해 `=(data)`로 원하는 값을 대입해 하드웨어에 쓰거나, 그 값을 반환받아 하드웨어 상태를 읽을 수 있게 됩니다.

4.  **`DIRECT` (Byte Offset) vs 일반 매크로 (Register Index) 차이:**
    *   `IOWR_32DIRECT` / `IORD_32DIRECT`: 넘겨준 `offset` 파라미터 그대로 바이트 단위 주소 덧셈을 수행합니다. 데이터시트나 문서의 Byte Offset(예: 0x00, 0x04, 0x08) 항목을 보고 코딩할 때 유용합니다.
    *   `IOWR` / `IORD`: `(reg) * 4`를 연산식에 포함시켜 두었습니다. 우리가 흔히 문서의 Word Address 항목에 기재된 "N번째 레지스터"(예: reg = 0번, 1번, 2번...) 인덱스 번호를 넣으면 매크로가 바이트 단위 간격인 4의 배수로 내부에서 수식 스케일링을 대신 해줍니다. (예: 1번 레지스터 = `(1) * 4 = base + 4` Byte)
