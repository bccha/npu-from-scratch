# STUDY.md — NPU 설계 학습 노트

## 1. MAC (Multiply-Accumulate)

가장 기본 연산 유닛.

```
acc_out = acc_in + (a × b)   // 1클럭 등록 출력
```

- 입력: `a` (INT8), `b` (INT8), `acc_in` (INT32)
- 출력: `acc_out` (INT32)
- `valid_in` 게이팅: valid_in=0이면 acc_out 유지

---

## 2. Systolic Dot Product (systolic_dot4)

**4-element dot product**: `result = a0*b0 + a1*b1 + a2*b2 + a3*b3`

### 구조

```
valid_in → [MAC0] → v0 → [MAC1] → v1 → [MAC2] → v2 → [MAC3] → valid_out
              ↓               ↓               ↓               ↓
           acc0            acc1            acc2           result
```

ACC가 MAC 체인을 **왼쪽 → 오른쪽**으로 흘러간다.

### Input Skewing

각 MAC[i]는 `i`사이클 뒤에 valid_in을 받는다 (앞 MAC의 acc가 도착하는 타이밍).  
그 시점에 `a[i]`, `b[i]`도 도착해야 하므로 입력을 미리 딜레이시킨다.

| MAC | valid 타이밍 | 받는 데이터 |
|-----|-------------|------------|
| MAC0 | Cycle 0 | a0, b0, acc_in=0 |
| MAC1 | Cycle 1 | **a1_d1**, **b1_d1**, acc_in=a0*b0 |
| MAC2 | Cycle 2 | **a2_d2**, **b2_d2**, acc_in=a0*b0+a1*b1 |
| MAC3 | Cycle 3 | **a3_d3**, **b3_d3**, acc_in=합산(0..2) |

`_d1`, `_d2`, `_d3` = shift register로 1, 2, 3사이클 딜레이.

- **레이턴시**: valid_in → valid_out = 4사이클
- **처리량**: 매 클럭 back-to-back 가능

---

## 3. Systolic 4×4 Matrix Multiply (systolic_4x4)

**4×4 행렬곱**: `C = A @ B`  (C[i][j] = Σ A[i][k] * B[k][j], k=0..3)

### 구조

16개 MAC 셀이 [4][4] 그리드로 **병렬** 존재.  
각 셀[i][j]는 자기 자리에서 k=0~3을 **4회 누산**.

```
Cycle k: 모든 셀[i][j] 동시에 → c_reg[i][j] += A[i][k] * B[k][j]
```

### 입력 형식

매 사이클 k에 A의 k열과 B의 k행을 슬라이스로 제공:

```
a_col[31:0] = { A[3][k], A[2][k], A[1][k], A[0][k] }   // 8bit × 4
b_row[31:0] = { B[k][3], B[k][2], B[k][1], B[k][0] }   // 8bit × 4
```

### RTL for 문

```verilog
for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
        c_reg[i][j] <= c_reg[i][j] + a[i] * b[j];
```

소프트웨어 루프와 다르게, RTL for문은 합성 시 **16개 가산기가 동시에 동작하는 회로**가 된다.  
루프 변수는 코드 전개(unroll)를 위한 표기일 뿐, 실행 순서가 없다.

### dot4와 비교

| | systolic_dot4 | systolic_4x4 |
|--|--|--|
| 출력 | 1개 스칼라 | 16개 (4×4 행렬) |
| MAC 수 | 4개 직렬 | 16개 병렬 |
| 레이턴시 | 4사이클 | 5사이클 (valid_in 4회 + 1) |
| 입력 방식 | 벡터 1회 | k-슬라이스 4회 스트리밍 |

---

## 4. Nios II에서 Avalon-MM으로 NPU 제어 (npu_ctrl)

`npu_ctrl.v` 는 Avalon-MM Slave 레지스터 인터페이스를 통해 3개 모듈을 제어한다.

### 레지스터 맵 (8-bit word 주소)

| addr | 이름 | R/W | 비트 설명 |
|------|------|-----|----------|
| 0 | CTRL | W | `[3:2]`=mode, `[1]`=valid_in, `[0]`=start |
| 1 | STATUS | R | `[0]`=valid_out |
| 2 | A_DATA | W | `{a3,a2,a1,a0}` 각 8bit |
| 3 | B_DATA | W | `{b3,b2,b1,b0}` 각 8bit |
| 4 | DOT4_RESULT | R | dot4 결과 (32-bit signed) |
| 5~20 | C_OUT[0..15] | R | 4x4 결과: `C[i][j]` = addr `5+i*4+j` |

MODE 인코딩: `00`=dot4, `01`=4x4, `10`=4x4_inst

### Nios II C 코드 예시

```c
#define NPU_BASE 0xFF200000

#define REG_CTRL(base)        (*(volatile uint32_t*)((base) + 0x00))
#define REG_STATUS(base)      (*(volatile uint32_t*)((base) + 0x04))
#define REG_A_DATA(base)      (*(volatile uint32_t*)((base) + 0x08))
#define REG_B_DATA(base)      (*(volatile uint32_t*)((base) + 0x0C))
#define REG_DOT4(base)        (*(volatile  int32_t*)((base) + 0x10))
#define REG_C(base, n)        (*(volatile  int32_t*)((base) + 0x14 + (n)*4))

// 4x4 행렬곱 (MODE=01)
void npu_matmul_4x4(int8_t A[4][4], int8_t B[4][4], int32_t C[4][4]) {
    // 1. mode=01, start=1 동시에
    REG_CTRL(NPU_BASE) = (1<<2) | (1<<0);   // mode=01, start

    // 2. k=0..3 슬라이스 스트리밍
    for (int k = 0; k < 4; k++) {
        REG_A_DATA(NPU_BASE) = (uint8_t)A[0][k]        |
                               (uint8_t)A[1][k] << 8   |
                               (uint8_t)A[2][k] << 16  |
                               (uint8_t)A[3][k] << 24;
        REG_B_DATA(NPU_BASE) = (uint8_t)B[k][0]        |
                               (uint8_t)B[k][1] << 8   |
                               (uint8_t)B[k][2] << 16  |
                               (uint8_t)B[k][3] << 24;
        REG_CTRL(NPU_BASE) = (1<<2) | (1<<1);   // mode=01, valid_in
    }

    // 3. 완료 대기 (valid_out 폴링)
    while (!(REG_STATUS(NPU_BASE) & 1));

    // 4. 결과 읽기
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C[i][j] = REG_C(NPU_BASE, i*4 + j);
}
```

### 4.2 Linux (ARM Cortex-A9) mmap 기반 제어

운영체제가 올라간 ARM 프로세서 (또는 일반 x86 리눅스) 환경에서는 하드웨어 메모리에 직접 포인터로 접근할 수 없으므로, `/dev/mem`과 `mmap`을 사용해 가상 주소 공간으로 매핑해야 합니다.

```c
#include <fcntl.h>
#include <sys/mman.h>

// 1. 메모리 디바이스 열기 (O_SYNC 필수: CPU 캐시 우회)
int fd = open("/dev/mem", O_RDWR | O_SYNC);

// 2. 가상 주소로 브릿지 매핑
void *lw_bridge_map = mmap(NULL, 0x200000, PROT_READ | PROT_WRITE, 
                           MAP_SHARED, fd, 0xFF200000); // LWHPS2FPGA_BASE

// 3. 레지스터 포인터 연결
volatile uint32_t *npu_ctrl_ptr = (uint32_t *)(lw_bridge_map + 0x0 /* NPU_OFFSET */);

// 4. 레지스터 쓰기 및 Polling
*npu_ctrl_ptr = 3;  // Load Weight & Valid (bit 1, 0)
while ((*(npu_ctrl_ptr + 1 /* STATUS OFFSET */) & 1) != 0); // Polling (Busy 대기)
```

**핵심 차이점**: 
1. `O_SYNC`: ARM의 Out-of-Order 실행 및 캐시 정책을 하드웨어 I/O(Device Memory) 정책으로 강제 변경해, DMA 전송 버퍼가 DDR 레벨에서 즉각 일관성을 갖게(Cache Coherency) 만듭니다. O_SYNC와 `volatile` 포인터를 조합하면 `dsb` 같은 명시적인 배리어 명령어 없이도 순서가 보장됩니다.
2. `Polling`: 비순차적으로 어마어마한 속도로 실행되는 ARM CPU 로직을 50MHz FPGA 속도에 동기화시키기 위해, `usleep` 같은 임의의 타이머 대신 Status Register를 지속적으로 읽어(Polling) Busy 신호가 꺼질 때까지 `while` 대기하는 정밀한 핸드셰이킹을 요구합니다.

### 4.3 start + mode 같은 사이클 가능한 이유

RTL에서 `mode_now = ctrl_write ? writedata[3:2] : mode_reg` 로 처리.  
start/valid_in 게이팅이 mode_reg(이전 값)가 아닌 mode_now(현재 쓰는 값)를 사용하므로,  
mode와 start를 한 번의 write로 동시에 줄 수 있다.

