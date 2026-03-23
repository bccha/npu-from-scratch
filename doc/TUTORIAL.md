# 실전! 밑바닥부터 만드는 AI 반도체 
**: MAC 연산기부터 SoC 통합 및 Full-Stack 최적화까지**

> *"이 튜토리얼은 딥러닝 가속기(NPU)의 가장 깊은 파이프라인부터, 리눅스 커널 레벨의 메모리 제어까지 관통하는 Full-Stack 하드웨어/소프트웨어 공동 설계(Co-design) 가이드입니다."*

---

## 머리말 (Preface)
오늘날 인공지능 영상 인식, 자율 주행, 온디바이스 AI의 핵심은 "어떻게 행렬 연산을 전력 효율적이고 극도로 빠르게 처리할 것인가"에 있습니다. 소프트웨어 엔지니어들은 PyTorch나 TensorFlow의 `matmul()` 함수 하나로 이 복잡함을 지나치지만, 진정한 성능의 퀀텀 점프는 언제나 실리콘 레벨(Hardware)에서 일어납니다.

이 문서는 단순한 "FPGA 깜빡이 켜기" 예제가 아닙니다.
우리는 하나의 덧셈과 곱셈을 수행하는 가장 작은 단위인 **MAC(Multiply-Accumulate)** 연산기를 설계하는 것에서 출발하여, 구글의 TPU 아키텍처로 유명해진 **Systolic Array**를 구축하고, 최종적으로 리눅스가 구동되는 **SoC(System-on-Chip) 환경에 하나뿐인 나만의 AI 반도체를 이식하는 여정**을 떠날 것입니다.

이 과정에서 우리는 메모리 병목 현상(Bottleneck)과 직접 맞서 싸우며, 하드웨어 엔지니어링의 정수인 **Direct Memory Access (DMA)** 및 **전용 Post-Processor 설계**를 통해 극단의 성능을 쥐어짜는 방법을 배웁니다.

---

## 🚀 여정의 목차 (Table of Contents)

### 1장. 연산의 심장을 뛰게 하다: MAC Unit과 Systolic Array
모든 딥러닝 연산의 90% 이상은 곱셈과 덧셈입니다. 우리는 가장 기초적인 곱셈-누산기(MAC)를 Verilog로 빚어내는 것부터 시작합니다.

*   **1.1 MAC 연산기 설계:** 1 클럭에 1개의 곱셈과 덧셈을 끝내는 마법.
    신경망 연산의 핵심 수식은 $y = \sum (w_i \cdot x_i) + b$ 입니다. 즉, 곱셈(Multiply)과 덧셈(Accumulate)을 무한히 반복하는 과정입니다. CPU 설계 철학과 다르게, NPU는 이 연산만을 오로지 빠르고 직관적으로 처리하기 위해 전용 하드웨어 블록인 **MAC(Multiply-Accumulate) Unit**을 최소 단위로 가집니다.

    실제 우리가 작성한 코드를 살펴봅시다:
    ```verilog
    // RTL/mac_unit.v
    module mac_unit (
        input  wire        clk,
        input  wire        rst_n,      // Active-low reset
        input  wire        en,         // Enable signal
        input  wire        clear,      // Reset accumulator to 0
        
        input  wire signed [7:0]  weight,   // 8-bit signed weight
        input  wire signed [7:0]  act,      // 8-bit signed activation
        
        output reg  signed [31:0] out_val   // 32-bit accumulated sum
    );

        wire signed [15:0] prod = weight * act;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_val <= 32'd0;
            end else if (en) begin
                if (clear) begin
                    out_val <= prod;  // Start new accumulation
                end else begin
                    out_val <= out_val + prod; // Accumulate
                end
            end
        end

    endmodule
    ```
    **동작 원리:**
    1. **Combinational 곱셈기:** `weight`와 `act` 두 개의 8비트 입력이 들어오면, 즉시 하드웨어 배선(wire)으로 연결된 곱셈기를 통과해 16비트의 결과(`prod`)로 산출됩니다. (FPGA 내부에 있는 고속 DSP 블록으로 합성됩니다.)
    2. **Sequential 덧셈기 (Accumulator):** 다음 클럭의 엣지(`posedge clk`)가 뛸 때, `out_val` 레지스터는 자신이 가지고 있던 기존의 값에 방금 들어온 `prod`를 덧셈(`+`)하여 저장합니다.
    3. **정밀도 유지:** 8비트끼리 곱하면 최대 16비트가 되고, 이 16비트 값을 수차례 더하다 보면 값이 커져 오버플로우(Overflow)가 발생할 수 있습니다. 따라서 출력 결과물은 넉넉한 32비트 레지스터(`out_val`)에 보관합니다.

    단 20줄도 안 되는 이 코드가, 구글 TPU를 지탱하는 Systolic Array의 심장 세포 하나입니다. 다음 섹션에서는 이 심장 세포를 64개로 복제하여 거대한 혈관(Dataflow)으로 묶는 과정을 살펴봅니다.

*   **1.2 데이터의 파도, Systolic Array:**
    왜 하나의 거대한 연산기 대신, 64개의 작은 MAC을 바둑판처럼 배열했을까요? 그 해답은 **'메모리 대역폭의 한계(Memory Wall)'**를 극복하기 위함입니다. 
    만약 64개의 MAC이 각자 메모리에서 데이터를 읽어오려고 한다면, 메모리 병목 현상 때문에 칩 전체가 멈춰버릴 것입니다. 구글 전설의 칩 아키텍트인 Norm Jouppi가 설계한 TPU는 **데이터가 심장(Systole) 박동처럼 인접한 셀로 차례차례 흘러가는 'Systolic Array' 구조**를 채택했습니다. 

    본 프로젝트에서는 **Weight Stationary(가중치 고정) 데이터플로우** 방식을 구현하였습니다:
    1. **Weight Pre-loading:** 연산 시작 전, 가중치(Weight) 매트릭스를 미리 Array 내부의 MAC 셀들에 하나씩 저장(고정)해둡니다. 매트릭스 곱에서 가중치는 여러 번 재사용되기 때문에, 한 번 심어두면 메모리 접근 횟수를 획기적으로 줄일 수 있습니다. (왜 Weight를 먼저 로딩해야 하는지에 대한 정답입니다!)
    2. **Activation Sliding:** 이제 입력 데이터(Activation)를 위쪽(North) 접점부터 한 사이클에 한 칸씩 아래로(South) 흘려보냅니다.
    3. **Partial Sum Accumulation:** MAC은 고정된 Weight와 위에서 내려온 Activation을 곱하고, 그 결과(Partial Sum)를 오른쪽(East)으로 넘겨줍니다. 최종 결과는 Array의 가장 오른쪽 끝에서 파도처럼 밀려나오게 됩니다.

    **⏳ 타이밍(Clock) 분석:**
    $N \times N$ 크기의 Systolic Array에서 데이터를 대각선으로 기울여(Skew) 입력하면, 첫 번째 값이 통과하여 나오기까지 $N$ 사이클의 레이턴시가 발생하고, 모든 행렬 연산이 완전히 끝나는 데 총 $3N - 2$ 번의 클럭 사이클이 필요합니다. 우리는 $8 \times 8$ Array를 사용하므로, **단 22 클럭**만에 64번의 화려한 MAC 연산이 파도 타듯 끝나고 하나의 행렬 곱셈 결과가 배출됩니다!

    이를 구현한 실제 Verilog 코드를 잠시 감상해 보시죠:
    ```verilog
    // RTL/systolic_array_8x8.v (일부 발췌)
    module systolic_array_8x8 (
        input  wire        clk,
        input  wire        rst_n,
        // ... 생략 ...
        input  wire [63:0] act_in,    // 8x8 activations entering from North
        output wire [255:0] out_val   // 8x32 accumulated sums exiting from East
    );

        // 8x8 MAC units 인스턴스화
        genvar r, c;
        generate
            for (r = 0; r < 8; r = r + 1) begin : row
                for (c = 0; c < 8; c = c + 1) begin : col
                    
                    // 각 MAC 유닛 간의 와이어 패스(Dataflow) 연결
                    // 북(North)->남(South)로 Activation 이동
                    // 서(West)->동(East)로 Partial Sum(결과값) 이동
                    
                    mac_unit mac_inst (
                        .clk     (clk),
                        .rst_n   (rst_n),
                        .en      (mac_en),
                        .clear   (mac_clear),
                        .weight  (weight_reg[r][c]), // Weight Stationary: 고정된 가중치
                        .act     (act_wire[r][c]),   // North에서 내려온 입력
                        .out_val (sum_wire[r][c])    // East로 넘겨줄 부분합
                    );

                end
            end
        endgenerate

    endmodule
    단순한 `for` 루프 같지만, 이 코드가 하드웨어로 합성(Synthesis)되면 실제 물리적인 실리콘 위에 64개의 곱셈기와 덧셈기가 가로세로로 엮여 데이터 고속도로를 형성하게 됩니다. 이것이 소프트웨어 병렬 처리(GPU)와는 차원이 다른, 하드웨어 공간 병렬성(Spatial Parallelism)의 진짜 매력입니다.

*   **1.3 검증의 예술, Cocotb (Python) 시뮬레이션:**
    하드웨어를 코딩하고 칩(FPGA)에 무작정 굽는 것은, 눈을 감고 우주선을 발사하는 것과 같습니다. 우리는 복잡한 Verilog Testbench 대신, Python 기반의 **Cocotb(COroutine based COsimulation TestBench)** 프레임워크를 도입하여 테스트의 질을 끌어올렸습니다.

    **Python의 강력함을 등에 업다:**
    1. **Numpy 연동:** Python의 Numpy `np.dot()`으로 행렬 곱셈 정답지(Golden Model)를 순식간에 만들어냅니다.
    2. **클럭(Clock) 주입:** Python 코드 안에서 `await Timer(10, units='ns')` 또는 `@cocotb.coroutine`을 통해 하드웨어의 클럭(Clock) 엣지를 자유자재로 다룹니다.
    3. **자동화 검증:** 
       - Random한 $8 \times 8$ 행렬 두 개(Weight, Activation)를 생성합니다.
       - Numpy로 정답 행렬(Expected Matrix)을 구합니다.
       - Python `for`문을 돌며, 매 클럭마다 Activation의 첫 번째 열부터 하드웨어 핀(`dut.act_in`)에 차례대로 밀어 넣습니다(Skewing).
       - 22클럭이 지난 뒤, 하드웨어 핀(`dut.out_val`)에서 튀어나오는 출력물을 수집합니다.
       - 수집한 결과를 Numpy 정답지와 1:1로 비교(Assert)하여 하드웨어 파이프라인의 **단 1의 타이밍 오차나 비트 밀림도 없음을 증명**해 냈습니다!
    *   [관련 코드: `sim/test_systolic_array_8x8.py`]

### 2장. 두뇌의 완성: NPU Controller와 메모리 인터페이스
심장이 생겼으니, 이를 제어할 두뇌를 만들어야 합니다. 메모리에서 가중치(Weights)와 입력값(Activations)을 가져와 Array에 정확한 타이밍으로 공급해주는 컨트롤러를 설계합니다.

*   **2.1 스트리밍의 미학, Avalon-ST 인터페이스:**
    데이터를 한 클럭에 하나씩 "물 흐르듯" 밀어넣기 위해 우리는 인텔의 표준 버스 규격인 **Avalon-ST (Streaming)** 인터페이스를 채택했습니다.
    핵심은 `valid`(데이터 유효함)와 `ready`(받을 준비 됨) 신호의 **핸드쉐이킹(Handshaking)**에 있습니다. 데이터를 주는 쪽(DMA)이 `valid`를 띄우고, 받는 쪽(우리 NPU)이 `ready`를 띄운 교집합의 순간에만 데이터가 Array의 심장부로 흘러 들어갑니다.

*   **2.2 거대한 기계태엽의 완성: FSM 제어 설계**
    우리의 컨트롤러는 다음과 같은 4단계의 리듬(유한 상태 머신, FSM)으로 정확하게 움직입니다:
    1. `STATE_IDLE`: 대기 상태. (CPU가 제어 레지스터(CSR)를 통해 'Start' 신호를 줄 때까지 대기)
    2. `STATE_PRELOAD_WEIGHTS`: (가장 먼저 수행!) DMA로부터 Weight 스트림을 받아서, Systolic Array 내부의 레지스터들에 하나하나 고정(Weight Stationary) 시킵니다.
    3. `STATE_LOAD_ACTS` (Execution): Weight가 장전되었으니, 진짜 입력 데이터(Activation) 스트림을 받아들입니다. Array 내부에서 22클럭 동안의 치열한 파도 타기 연산이 진행됩니다.
    4. `STATE_DRAIN_OUT`: 8x8 행렬 곱셈이 끝난 256비트(32비트 x 8)의 거대한 결과물을 쪼개서 다시 메모리로 내보냅니다. 완료 후 `STATE_IDLE`로 돌아갑니다.

    아래는 스트림 데이터를 Array에 꽂아 넣는 컨트롤러의 핵심 FSM 코드입니다:
    ```verilog
    // RTL/npu_stream_ctrl.v (일부 발췌)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            // ... 초기화 생략 ...
        end else begin
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        state <= (mode == 2'b11) ? STATE_PRELOAD_WEIGHTS : 
                                 (mode == 2'b01) ? STATE_LOAD_ACTS : STATE_IDLE;
                    end
                end

                STATE_PRELOAD_WEIGHTS: begin
                    if (st_sink_valid && st_sink_ready) begin
                        // 8번의 클럭 동안 64개의 Weight 데이터를 Array로 밀어넣음
                        if (act_row_cnt == 7) begin
                            state <= STATE_IDLE;
                            done <= 1'b1;
                        end else begin
                            act_row_cnt <= act_row_cnt + 1;
                        end
                    end
                end

                STATE_LOAD_ACTS: begin
                    if (st_sink_valid && st_sink_ready) begin
                        // Activation 데이터가 Array를 휩쓸고 지나감 (Execution)
                        if (act_row_cnt == act_row_max) begin
                            state <= STATE_DRAIN_OUT; // 연산 완료 후 결과 배출로 넘어감
                            drain_cnt <= 0;
                        end else begin
                            act_row_cnt <= act_row_cnt + 1;
                        end
                    end
                end
                
                // ... STATE_DRAIN_OUT 생략 ...
            endcase
        end
    end
    ```

*   **2.3 버스 프로토콜의 검증, Cocotb와 Avalon-ST:**
    컨트롤러가 완성되면 다시 한 번 Cocotb 시뮬레이터(ModelSim 혹은 Icarus Verilog)를 켜고 가상의 스트림 데이터를 쏘아봅니다.
    
    1. **가상의 DMA(스트림 소스) 코딩:** 
       Python 스크립트에서 DMA 역할을 하는 'Avalon-ST Source Driver' 클래스를 직접 작성합니다. 
    2. **핸드쉐이킹(Valid-Ready) 검증:** 
       이 드라이버가 매 클럭 랜덤하게 `valid`를 띄우기도 하고 내리기도(Stall) 합니다. NPU Controller가 `ready`를 보낼 때만 Array 안에 데이터가 톱니바퀴 맞물리듯 정확히 들어가는지를 검증합니다.
    3. **End-to-End 시뮬레이션:** 
       최종적으로 H/W가 `STATE_DRAIN_OUT` 타임에 토해낸 256비트 덩어리들을 수집해 Python/Numpy 연산 결과와 비교합니다. **단 1비트의 오차도 없이 $100\%$ 완벽하게 일치**하는 짜릿한 순간을 맞이하며, 이제 소프트웨어(SoC)와 연동할 준비가 끝났음을 확신합니다.
    *   [관련 코드: `sim/test_npu_unit.py`]

### 3장. 첫 번째 숨결: SoC 연결과 기초 행렬 곱셈 (npu_test)
잘 만들어진 회로를 인텔 Cyclone V SoC 보드에 밀어 넣습니다. Platform Designer(Qsys)를 이용해 CPU와 커스텀 NPU를 연결합니다. 가장 먼저, 복잡한 인퍼런스 대신 아주 단순한 $8 \times 8$ 행렬 두 개를 곱해보며 시스템의 생사를 확인합니다. 

특히 이 단계에서는 돌다리를 두드려 건너듯 **두 단계의 검증 과정**을 거쳤습니다:
*   **3.1 Nios II 소프트코어 검증:** 처음부터 무거운 리눅스와 ARM을 붙이지 않았습니다. FPGA 내부에 가벼운 Nios II (가상 CPU)를 얹어, C언어로 작성된 펌웨어 수준(Baremetal)에서 NPU 레지스터를 직접 타격하며 하드웨어가 진짜로 숨을 쉬는지 확인했습니다.
*   **3.2 ARM HPS (리눅스) 이식:** Nios II에서 하드웨어 무결성이 100% 검증된 후, 비로소 ARM CPU(HPS)로 제어권을 넘겼습니다. C 언어의 `mmap`을 통해 리눅스 운영체제를 우회하고 하드웨어 물리 주소(Memory Mapped I/O)를 타격하는 드라이버를 작성했습니다.
*   **3.3 첫 행렬 곱셈 (npu_test):** 임의의 $8 \times 8$ 행렬을 하드웨어로 보내고, 결괏값을 읽어와 CPU가 계산한 정답과 일치하는지 비교하며 통합 시스템의 첫 숨결을 느낍니다.
*   [관련 코드: `linux_software/npu_test/main.c`]

### 4장. 실전 AI 추론: MNIST 숫자 인식
기본적인 행렬 곱셈이 동작하는 것을 확인했으니, 이제 진짜 신경망 모델을 올려볼 차례입니다.
*   **4.1 가중치(Weight) 로딩:** Python 텐서플로우(TensorFlow)로 미리 학습시켜둔 MNIST 모델의 가중치(Weights)와 편향(Bias) 파라미터들을 `.bin` 파일 포맷으로 추출합니다.
*   **4.2 C언어 기반 추론(Inference) 로직:** ARM 리눅스 위에서 C 언어로 `main.c`를 작성합니다. 테스트 이미지 데이터를 매번 DDR에서 NPU에 밀어 넣고 연산 결과를 받아옵니다.
*   **4.3 88%의 정확도와 성능 측정:** NPU가 연산한 결과가 소프트웨어 시뮬레이터와 완벽하게 일치하며 숫자 이미지를 성공적으로 분류해냅니다.
    *   **⚡ 성능 측정:** 오직 순수 DDR 버스만 사용했을 때, 순수 ARM CPU 연산 대비 **정확히 3.7배(3.7x)의 가속**을 이뤄냈습니다! 하지만 NPU의 엄청난 잠재력에 비하면 여전히 병목이 존재합니다.
*   [관련 코드: `linux_software/mnist_test/main.c`]

### 5장. 벽에 부딪히다: OCM의 통곡의 벽과 memcpy의 구원
"속도가 왜 이렇게 깎이지?!"
NPU의 스트림 데이터는 MSGDMA가 잘 처리해주고 있었지만, NPU 연산 결과(Partial Sum)를 임시로 저장하기 위해 **내부 메모리(OCM)**를 도입하면서 예기치 못한 병목을 만납니다. CPU가 OCM에 들어있는 32비트 결과값을 후처리하기 위해 하나씩 꺼내올 때, 처음에 C 언어의 `IORD`와 `IOWR` 매크로를 이용해 AXI 버스를 단어(Word) 단위로 찔러댔기 때문입니다. 이렇게 CPU가 매번 버스 트랜잭션을 수만 번 일으키며 데이터를 옮기다 보니 치명적인 **속도 저하(Degradation) 문제**가 발생했습니다.
*   **5.1 버스 오버헤드의 파괴:** 이 끔찍한 오버헤드를 타파하기 위해, 한 땀 한 땀 읽어오는 코드를 과감히 버리고 C표준 라이브러리의 블록 단위 복사 함수인 `memcpy`로 전격 교체합니다.
*   **5.2 4배 이상의 퀀텀 점프:** `memcpy`를 통해 AXI 버스의 Burst 전송 성능을 제대로 이끌어내자, 어이없을 정도로 간단하게 통곡의 벽이 부서집니다.
    *   **⚡ 성능 비교:** `IORD/IOWR` 시절의 초라한 속도에서 벗어나, `memcpy` 단 한 줄의 도입만으로 **순식간에 4배 이상(Over 4x) 가속된 16.3ms**의 벽을 뚫어냅니다!
*   [관련 코드: `linux_software/mnist_test/main.c`]

### 6장. 벼랑 끝의 사투와 현실적 타협: Hardware Post-Processor (PP)
가속은 성공했지만, CPU가 여전히 발목을 잡습니다. NPU가 쏟아낸 32비트 결과물을 다시 OCM(On-Chip Memory)에서 가져와서 Bias를 더하고(Add), 8비트로 양자화(Shift/ReLU)하는 후처리 연산을 피하기 위해, 당초 우리는 메모리에 직접 접근하는 독자적인 거대 **Avalon-MM Master 후처리기(`npu_post_processor.v`)**를 야심 차게 설계했습니다. 

하지만 실리콘 회로의 세계는 냉혹했습니다.
*   **6.1 이상과 현실의 충돌 (6% Accuracy Drop):** 야심 차게 달아놓은 Avalon-MM Master 구조는 예상치 못한 AXI 버스 데드락(Deadlock)과 누적기(Accumulator) 간의 침범 버그를 일으켰습니다. 98%였던 파이토치 정확도를 하드웨어에서 무려 **6%로 폭락**시키는 대참사를 낳고 말았습니다.
*   **6.2 "The Great Pivot" - Tightly-Coupled OCM Pipeline:** 
    보드 위에서 메모리 주솟값을 덤프(Dump)하며 며칠 밤을 새워 버그를 추적한 끝에, 거대하고 불안정한 외부 Master를 전면 폐기하고 **MAC Array 엉덩이에 찰싹 달라붙어 물 흐르듯 직결된 `npu_ocm_accumulator.v` 인라인 파이프라인 구조**로 설계를 전면 개편(Pivot)했습니다. 
    Array가 22클럭 연산을 마치고 곧장 32비트를 토해낼 때마다(Drain 과정), 별도의 복잡한 메모리 버스 요청 없이 **그 자리에서 즉시 Bias 덧셈, 하드웨어 Shift(>>8), ReLU 함수를 1클럭 단위 논리 회로로 스치듯 통과**시킨 후, 깨끗하게 정제된 8-bit 결과만을 DDR로 방출하는 가볍고 날렵한 모듈을 완성했습니다.
*   **6.3 하이브리드(Co-design) 가속의 최종 완성:**
    결함이 발생했을 때 고집스럽게 실리콘 수정(Verilog)으로만 풀려 하지 않고 소프트웨어(C언어 런타임) 드라이버와 그 역할을 가장 영리하게 타협(Trade-off)한 결과, 정확도는 원본(97.09%) 대비 **96.40%**로 완벽히 복구되면서도 인퍼런스 타임은 **단 1.8ms 대의 압도적인 속도 (순수 CPU 대비 약 3.75배 가속)**를 온전히 거머쥐는 최적의 시스템 밸런스를 이룩했습니다.
    *   [관련 코드: `npu_ocm_accumulator.v`, C-드라이버 하이브리드 로직]

---

### ⏸️ 중간 점검: 연산의 거대한 분업 (Hardware vs Software Responsibilities)
지금까지 우리는 가장 밑바닥의 곱셈기부터 시작해 하나의 완전한 인퍼런스(Inference) 시스템을 조립해 냈습니다. 7장의 본격적인 컴파일러 자동화(SDK) 파트로 넘어가기 전에, 과연 파이토치의 복잡한 수학 공식들이 도대체 **어느 물리적 계층에서, 어떻게 분담되어 처리되고 있는지** 명확히 짚고 넘어가겠습니다.

1. **MatMul (행렬 곱 연산) $\rightarrow$ 순수 하드웨어 (Systolic Array)**
   * 파이토치의 `nn.Linear`나 `nn.Conv2d`가 유발하는 천문학적인 횟수의 곱셈-누산(MAC)은 100% FPGA 내부의 $8 \times 8$ Systolic Array가 전담합니다. CPU는 데이터를 DMA로 넘겨줄 뿐, 단 한 번의 곱셈 연산 로직에도 관여하지 않습니다.
   
2. **Batch Normalization (정규화) $\rightarrow$ 소프트웨어 오프라인 병합 (Offline Fusion)**
   * 분산과 표준편차를 구하고 나누는 이 무거운 $\gamma, \beta$ 정규화 연산은 놀랍게도 런타임 상의 **어느 칩에서도 실행되지 않습니다.** 보드에 실리기도 전인 런타임 이전(Offline)에, 파이썬 컴파일러 코어(`cyclone_npu_sdk.py`)가 앞단 레이어의 가중치(Weight) 매트릭스 수식 자체에 정규화 수식을 영구적으로 융합(Folding)해 버렸기 때문입니다. 즉, 연산 자체가 통째로 증발(Zero-Cost)했습니다.

3. **8-bit 양자화 (Quantization & Scaling) $\rightarrow$ 파이썬(전처리) + 하드웨어 Post-Processor(후처리)**
   * 모델의 소수점(Float32) 파라미터들을 -128 ~ 127 사이의 Int8 정수로 스케일링하여 압축하는 역할 자체는 파이썬(SDK)이 담당합니다. 하지만 반대로, Array에서 연산을 거친 뒤 32비트로 거대하게 부풀어 오른 결과물들을 다시 다음 레이어 병렬 연산을 위해 8비트로 물리적으로 끌어내리는(Shift) 치열한 포맷팅은 **전적으로 FPGA 하드웨어의 Post-Processor가 1클럭 내역**으로 전담하여 깎아냅니다.
   
4. **ReLU (비선형 활성화) $\rightarrow$ 순수 하드웨어 (Post-Processor 온칩 처리)**
   * `if (val < 0) val = 0;` 이 단순한 분기문조차 ARM CPU가 커널 루프를 돌며 소프트웨어로 순회하면 무시무시한 지연 지표를 초래합니다. 우리는 이를 하드웨어 설계 레벨에서 Post-Processor가 결과물을 DDR 메모리로 내뿜어낼(Drain) 때, 0보다 작으면 곧바로 `0`으로 그라운드 차단시켜버리는(Clipping) 하드웨어 로직 다이오드로 박아 넣었습니다.

이 완벽하고도 냉혹한 책임의 분산(Co-design) 설계 덕분에, ARM Host CPU의 오버헤드는 0(Zero)으로 수렴했고 CPU는 오직 메모리 전송(`memcpy`)만 무념무상으로 수행하는 궁극의 병렬 시스템이 탄생했습니다. 

자, 그럼 이제 이 복잡천만한 하프-하드웨어 분업을 **"인간이 신경 쓰지 않도록 완전히 은닉하고 자동화해 버리는" 7장의 컴파일러 이야기**로 넘어가 보겠습니다.

---

### 7장. 궁극의 추상화: PyTorch FX 기반 컴파일러 생태계 구축 (BYOC)
하드웨어 파이프라인(Systolic Array + Post-Processor)이 완성되고 C 언어 레벨의 드라이버까지 제어권을 얻었지만, 한 가지 거대한 불만족이 남았습니다. **"새로운 AI 모델 구조가 바뀔 때마다, 데이터 사이언티스트가 C 코드를 열어 배열 크기(`im2col` 파라미터 등)와 가중치 추출 스크립트를 수동으로 고쳐야 하는가?"** 

진정한 글로벌 AI 가속기(TPU, NPU) 기업들은 모두 하드웨어를 숨기고 **가장 대중적인 머신러닝 프레임워크인 파이토치(PyTorch) 단 한 줄의 코드로 배포**를 끝내는 독자적인 소프트웨어 컴파일 스택(SDK)을 가지고 있습니다. 우리 여정의 진정한 대미를 장식하기 위해, 100% Python으로 빚어낸 **독자적인 NPU Compiler SDK (`cyclone_npu_sdk.py`)** 를 설계합니다.

*   **7.1 해부학의 정수, PyTorch FX AST 파싱:**
    모델의 내부를 뜯어보기 위해 무겁고 복잡한 외부 컴파일러 프레임워크(TVM, MLIR 등)에 의존하는 대신, 우리는 파이토치 네이티브 코어 추적 도구인 `torch.fx.symbolic_trace`를 등판시켰습니다. 파이썬 스크립트가 능동적으로 AI 모델의 추상 구문 트리(AST Graph) 노드를 순회하며, 자신이 NPU 하드웨어로 직접 가속할 수 있는 노드(`nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`)들을 스스로 스캔하고 맵핑합니다.

*   **7.2 영리한 수학, Operator Fusion (Conv + BN + ReLU):**
    실제 엣지(Edge) 인퍼런스 환경에서 Batch Normalization 계층은 막대한 리소스 낭비를 유발합니다. 우리는 NPU 컴파일 파싱 타임에 이 비효율을 **오프라인 연산자 융합(Offline Fusion)**으로 제거합니다. BN의 스케일 지표인 $\gamma, \beta$ 값들과 이동 평균/분산을, 앞단 Convolution 레이어의 가중치(Weight)와 편향(Bias)에 수학적으로 완벽히 섞어 굳혀서(Folding), 보드 런타임 시의 정규화 연산 사이클 자체를 물리적으로 지워버렸습니다.

*   **7.3 엣지를 위한 극단적 선택: Memory vs Compute Trade-off:**
    합성곱(CNN) 처리를 위해 `im2col`(슬라이딩 윈도우) 기법이 필요할 때, 데스크톱 파이썬 쪽에서 이미지 패치를 전부 정렬해서 주면 타일 데이터가 13배 이상 뻥튀기(`Memory Explosion`)되어 10,000장 기준 100MB를 넘나드는 대용량이 됩니다. 우리의 SDK 런타임은 **과감하게 데스크톱 전처리를 포기하고, 원본 초경량(`28x28`) 이미지를 버스로 던져 보낸 뒤, NPU 호스트인 800MHz ARM CPU의 초고속 L2 캐시 위에서 C 언어가 런타임에 동적으로 패치 창문을 잘라내며 미끄러지도록(Dynamic Sliding)** 책임과 역할을 분담하여 I/O 버스 병목을 박살 냈습니다.

*   **7.4 "동적 C 코드 자동 생성 (Dynamic EmitC)":**
    파싱된 노드 메타데이터(텐서 형상, 채널 크기, 합성곱 패딩 및 스트라이드, ReLU 은닉 여부 등)를 기반으로, 방대하고 복잡한 **NPU 제어용 C 언어 런타임 코드를 사람이 직접 코딩하는 대신 파이썬 SDK가 스스로 타이핑하여 뱉어내는(`npu_auto_runtime.c`) 경이로운 경지**에 도달했습니다.
    
    딥러닝 엔지니어는 자신이 익숙한 파이토치로 자유롭게 망을 설계한 뒤, 스크립트 최하단에 단 한 줄의 함수만 호출하면 됩니다:
    ```python
    # 딥러닝 연구자의 시점: 하드웨어 베릴로그나 C 코드를 단 1줄도 몰라도 됩니다.
    export_model_to_fpga(model, test_dataset, out_dir="./", out_c_file="./npu_auto_runtime.c")
    ```
    이 단방향 메서드 호출 한 방이 거대한 매트릭스 타일링(Tiling), 8-bit 심리스 양자화(Int8 Quantization), 그리고 C 런타임 드라이버 소스코드 주조까지 단 0.1초 만에 논스톱으로 관통합니다. 단순 퍼셉트론(MLP) 아키텍처는 물론, 복잡한 2D 공간 구조의 **합성곱 신경망(Convolutional Network, CNN) 아키텍처까지 96.00%~96.40% 의 무결점급 정확성으로 DE10-Nano 실리콘 구역에 영구 안착시켰습니다.** 
    
    진정한 의미의 **풀스택 자작 컴파일러 생태계(Bring-Your-Own-Compiler)**가 마침내 이곳 튜토리얼에서 완성되었습니다.

---

## 🎯 마치며
여기까지 오셨다면, 당신은 단순한 프로그래머가 아닙니다. 물리적인 트랜지스터의 흐름부터 운영체제의 주소 공간 매핑까지, 컴퓨팅 스택의 밑바닥부터 꼭대기까지 수직으로 돌파한 **Full-Stack Hardware Architect**입니다. 

이제 여러분만의 혁신적인 AI 모험을 시작할 준비가 끝났습니다.
