# npu-from-scratch

DE10-Nano (Cyclone V SoC) FPGA 환경에서 NPU를 밑바닥부터 설계한 프로젝트입니다. 8x8 MAC 어레이 기반의 하드웨어 설계부터, Linux Application 통합, 그리고 실제 MNIST 필기체 인식(MLP 및 CNN Inference)까지 End-to-End 시스템을 구현했습니다.

<p align="center">
  <img src="doc/assets/npu_architecture_flowchart.png" width="400">
</p>

## 프로젝트 아키텍처 (Full-Stack Dataflow)

가장 최상단의 딥러닝 리서처 공간(Python)부터, 칩 내부의 실시간 디지털 논리 회로(Verilog)까지 이어지는 전체 파이프라인(Top-Down Processing)은 아래 그림과 같이 동작합니다.

**✅ End-to-End Deep Learning Compiler SDK (`cyclone_npu_sdk.py`):**
파이토치에서 시작하여 모델 학습, 오프라인 융합, 파라미터 양자화 덤프, 그리고 네이티브 C 드라이버(EmitC) 컴파일까지 모든 컴포넌트가 **개발자 개입 없이 100% 자동화되어 최대 99.00% 정확도의 방대한 합성곱 신경망(CNN) 모델**을 타겟 시스템으로 완벽하게 동적 배포합니다.

```mermaid
flowchart TD
    subgraph Frontend ["1. AI Framework (PyTorch)"]
        PT["모델 정의 (Model Definition)"]
        FX["FX Graph Compiler (npu_fusion_pass.py)"]
        PT -- Symbolic Trace --> FX
        FX -- Parameter Folding --> Fused["Offline Fusion\n(Conv + BN + ReLU)"]
    end

    subgraph Backend ["2. Compiler Codegen (BYOC)"]
        Fused -- Tensor Tiling --> Bin[".bin HW Weights & Biases"]
        Fused -- Transpile --> EmitC["Auto-Generated C Driver\n(npu_model_runtime.c)"]
    end

    subgraph OS ["3. Linux Runtime (C/C++)"]
        Bin -.-> App["Bare-metal / Linux Application"]
        EmitC -.-> App
        App -- mmap() / /dev/mem --> Mem["DDR3 Memory Space"]
    end

    subgraph Hardware ["4. FPGA Hardware (RTL)"]
        Mem -- AXI Bridge --> DMA["MSGDMA Engine"]
        DMA -- Avalon-ST Streaming --> MAC["8x8 Systolic Array (MAC)\n+ Post-Processor"]
    end
```


## 프로젝트 스코프 (Project Scope)
이 프로젝트의 최종 목표는 단순한 하드웨어 MAC(Multiply-Accumulate) 연산기 설계에 머물지 않습니다. 
상용 AI 가속기와 동일한 **풀스택(Full-Stack) AI 엔지니어링 생태계 모델**을 단독으로 구축하는 것을 스코프로 정의합니다.

1. **AI Framework (Frontend):** PyTorch 모델 설계, QAT(Quantization Aware Training) 및 Operator Fusion(Conv+BN 수학적 병합)
2. **Compiler Codegen (Backend):** 
   - 무거운 외부 스택(TVM/MLIR)에 의존하지 않고, **PyTorch FX** 역추적 기술을 활용하여 하드웨어 제어 논리를 생성하는 **100% Python 자체 컴파일러 컴포넌트(`npu_compiler.py`)**를 구축했습니다.
   - 파이토치 모델의 구조(AST Graph)를 동적으로 스캔하여, C 코드를 사람이 직접 짤 필요 없이 **모델 구조 그대로를 NPU 제어 C-루틴으로 100% 자동 통합 생성(Emit C / BYOC)해내는 진정한 컴파일 파이프라인**을 완성했습니다.
3. **Hardware (RTL):** Pipeline Post-Processor (Bias/Shift/ReLU) 설계와 Avalon-ST 다이렉트 스트리밍을 통한 Zero-Padding 하드웨어 100% 활용도 달성
4. **OS Runtime (C/C++):** 컴파일러가 출력한 가중치 바이너리(`.bin`)를 메모리 매핑(`/dev/mem`, `mmap`)으로 적재하여 NPU를 제어하는 리눅스(Linux) HPS Bare-metal 런타임 환경


## 핵심 성과 (Key Achievements)

*   **PyTorch FX 자동 컴파일러 (Dynamic EmitC):** 모델의 층계 구조와 관계없이 파이썬 스크립트가 파이토치 네트워크의 AST Graph를 스스로 탐색하여 **단 한 줄의 수동 개입 없이 완벽한 NPU 리눅스 C 제어 코드를 동적으로 뱉어내는(Emit) 기능**을 달성했습니다. 인공지능 엔지니어가 딥러닝 모델만 훈련하면 클릭 한 번에 0.00ms 소프트웨어 오버헤드로 하드웨어 바이너리와 런타임이 즉각 조립되는 진정한 BYOC 배포 생태계입니다. (단순 다층 퍼셉트론 뿐만 아니라 **Convolutional Network(CNN), MaxPool 토폴로지까지 FX Graph로 스캔하여 완벽 지원**합니다.)
*   **그래프 레벨 능동 최적화(Operator Fusion) 및 8-bit 자동 형변환:** 컴파일러 엔진은 코드를 번역할 뿐만 아니라, 파이토치 내의 `[Linear -> BatchNorm1d]` 구조를 1개의 통합 가중치(Fused Weight)로 완벽하게 **수학적 오프라인 압축(Folding)** 해 냅니다. 압축된 파라미터는 하드웨어 MAC 타일(8x8) 규격에 맞춰 **네이티브 8-bit(INT8) 정수형으로 자동 양자화(Quantization & Pad-Packing)** 되어 `.bin` 추출까지 논스톱으로 진행됩니다.
*   **하드웨어 Post-Processor (비선형 ReLU 및 Bias 온칩 가속):** 파이썬 컴파일러가 모델 내에 `nn.ReLU` 노드를 탐지하면, ARM CPU가 이를 계산하도록 두지 않고 C 제어 명령을 통해 FPGA 하드웨어의 **Post-Processor(NPU Drain 모드)를 가동**시킵니다. Bias 덧셈, Scaling(Shift), 그리고 ReLU Clipping까지 모두 버퍼(OCM)에서 DDR로 쏟아져 나오는 동안 단 1클럭 사이클 내에 즉시 100% 하드웨어 가속 처리됩니다.
*   **압도적 하드웨어 성능 달성:** 순수 ARM Host CPU 연산 대비 NPU + MSGDMA + 자동화 Hardware Post-Processor 파이프라인 가동으로 소프트웨어 오버헤드를 0에 수렴시켰습니다. 10,000장 추론 기준 CPU 대비 **3.75x Speedup** (장당 1.839 ms) 및 **최대 99.00% 무결점 정밀도(PyTorch Float32 100% 일치)**를 증명해 냈습니다.

### 📊 성능 비교: MLP 모델 기준 (Before vs After)
| 지표 | Before (Legacy NumPy 정수화) | After (PyTorch + BN Offline Fusion) | 향상폭 |
| :--- | :--- | :--- | :--- |
| **추론 정확도 (Accuracy)** | 89.58% | **97.92%** | **+ 8.34%p** |
| **추론 속도 (1 Image)** | 6.957 ms (CPU 연산) | **1.876 ms (NPU 가속)** | **3.71배 고속화** |
| **하드웨어 호환성** | 100% 매칭 | **100% 매칭** (스택 구조 변경 없음) | |

### 📊 지원 토폴로지 확장: CNN 아키텍처 달성 (Hardware Acceleration)
| 지표 | MLP 네트워크 (Deep Dense) | CNN 네트워크 (Convolutional) | 비고 |
| :--- | :--- | :--- | :--- |
| **최종 정확도 (Accuracy)** | **96.60% ~ 97.92%** | **98.20% ~ 99.00%** | 파이토치 Float32 원본 시뮬레이션 연산과 물리 하드웨어 INT8 연산의 100% 완전 일치 증명 |
| **소요 시간 (1 Image)** | **1.863 ms** | **64.47 ms** | NPU 스케줄링 및 파라미터 크기에 따른 정비례 물리 매핑 |
| **자동 파싱 지원 (Topologies)** | Linear, ReLU, BatchNorm1d | Conv2d, MaxPool2d, Linear, ReLU, BN | **`cyclone_npu_sdk.py`를 통해 모든 계층 구조 PyTorch FX 동적 생성 지원** |

*   **Buffer-less Streaming Pipeline:** Avalon-ST 인터페이스의 `valid/ready` 스트리밍 프로토콜을 구현하여, 중간 버퍼(SRAM) 없이 MSGDMA 데이터를 직접 처리하는 효율적인 아키텍처를 설계했습니다.
*   **Full-Stack 시스템 통합 설계:** 시스템 버스 통합(Qsys), Avalon-MM/ST 인터페이스 연결 구조, 그리고 Linux User-space ( `/dev/mem`, `mmap` ) C 드라이버 프로그래밍까지 H/W와 S/W 전반을 직접 구현했습니다.

## 하드웨어 사양
- **보드:** Terasic DE10-Nano (Intel Cyclone V SoC)
- **FPGA:** Cyclone V SE 5CSEBA6U23I7
- **HPS:** ARM Cortex-A9 Dual-core @ 800MHz
- **베이스 프로젝트:** DE10-Nano SoC GHRD (Golden Hardware Reference Design)

## 문서 목차 (Documentation)
상세한 개발 스토리와 아키텍처 다이브(Deep-Dive)는 아래 개별 문서를 참조하십시오.

*   [**TUTORIAL.md**](doc/TUTORIAL.md): MAC 설계부터 MNIST 추론과 H/W 최적화 과정까지를 다루는 **단계별 실전 튜토리얼 (가장 추천)**
*   [**DESIGN_FUSION.md**](doc/DESIGN_FUSION.md): **[NEW]** PyTorch FX 컴파일러부터 내부 하드웨어 파이프라인(SRAM 누적기 및 후처리기)까지 이어지는 Full-Stack Operator Fusion 설계 문서
*   [**RESULT.md**](doc/RESULT.md): CPU vs NPU 성능 검증 및 상세 벤치마크 결과 시트
*   [**DESIGN.md**](doc/DESIGN.md) & [**DESIGN_2ND.md**](doc/DESIGN_2ND.md): 핵심 NPU 계층 구조 및 AXI/Avalon 아키텍처 상세 사양서
*   [**DESIGN_AI.md**](doc/DESIGN_AI.md): NPU 위에서 동작하는 AI 신경망 매핑 및 양자화(Quantization) 구조
*   [**REG_MAP.md**](doc/REG_MAP.md): MSGDMA 및 NPU CSR 플릿 맵
*   [**TESTS.md**](doc/TESTS.md): Python (Cocotb) 기반의 가상 하드웨어 자동 검증 환경
*   [**LESSONS_LEARNED.md**](doc/LESSONS_LEARNED.md): 맨땅에서 설계하며 얻은 트러블슈팅 및 딥 엔지니어링 교훈

## AI 개발 어시스턴트 적용
이 프로젝트는 **H/W-S/W Co-design의 기획 단계부터 최종 구현, 디버깅까지 최신 AI(LLM) 에이전트를 실무형 페어 프로그래머(Pair Programmer)로 적극 활용**했습니다. 수십 시간이 걸리는 검증 코드를 단숨에 생성하거나 버스 인터페이스의 난해한 FSM 타이밍 데드락 버그를 함께 분석하는 등, 거대한 FPGA 하드웨어-소프트웨어 개발 사이클을 기하급수적으로 단축시킨 선도형 개발 사례입니다.
