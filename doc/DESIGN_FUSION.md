# Full-Stack NPU Fusion Compiler Roadmap

이 문서는 PyTorch 컴파일러(Frontend) 레벨의 Operator Fusion 처리부터 하드웨어 NPU(Backend)의 전용 Post-Processor까지 이어지는 **풀스택 AI 가속기(Full-Stack AI Accelerator)** 구현 파이프라인과 그 단계를 구조화한 핵심 설계서입니다.

---

## 🎯 핵심 목표 (Core Objective)
단순한 FPGA 행렬곱 가속기를 넘어, 상용 신경망 처리 장치(TPU, TensorRT, Apple Neural Engine)와 같이 **컴파일러가 NPU 하드웨어 구조를 인지하고(HW-Aware) 계산 그래프를 최적화하여 100% 성능을 뽑아내는 완전체 모델**을 구현합니다.

* 핵심 패러다임: **`[Conv2d] ➔ [BatchNorm2d] ➔ [ReLU]` 노드의 단일 Custom Node 압축 (Operator Fusion)** 및 **하드웨어 배수에 맞춘 필터 타일링 (NAS)**

---

## 🛠 구현 로드맵 (Bottom-Up Approach)

아키텍처 붕괴를 막고 검증을 완벽히 수행하기 위해, 물리적 레이어(H/W)부터 최상위 컴파일러(MLIR) 레이어 방향으로 구축합니다.

### Step 1. H/W Post-Processor (PP) RTL 모듈 삽입 및 검증
가장 먼저 도착지가 될 하드웨어 파이프라인 구조를 정밀하게 완성합니다.
* **작업:** `post_processor.v` (Bias 덧셈, Shift 스케일링, ReLU 클리핑) 파이프라인 작성.
* **통합:** `npu_unit.v` 내부의 `systolic_core` 출력부와 `npu_stream_ctrl` 입력부 사이(Stream-to-Stream)에 연결.
* **검증:** C 드라이버(`main.c`)에서 기존 CPU 연산 루프(SW Accum/ReLU) 삭제 후, DMA 데이터 스트리밍 속도 프로파일링 및 8-bit 결과 정확도 일치 증명.
> **📂 수정 타겟 파일:**
> * `[신규생성]` `rtl/post_processor.v`
> * `[수정]` `rtl/npu_unit.v` (PP 인스턴스화 및 Avalon-ST 256-bit -> 64-bit 스트림 체인 연결)
> * `[수정]` `linux_software/cnn_test/main.c` (CPU Accum/ReLU 루프 제거 및 DMA 8-bit 수신 로직 반영)

### Step 2. 파이썬 수학적 오프라인 퓨전 (Offline Fusion) 검증
컴파일러 백엔드를 연동하기 전에, 수식 최적화가 모델 정확도를 해치지 않는지 파이썬에서 증명합니다.
* **작업:** `cnn_train_and_export.py`에 Batch Norm 레이어를 추가하여 QAT(Quantization Aware Training) 기반 학습 수행.
* **통합:** 모델 Export 시점에 Batch Norm 파라미터($\gamma, \beta, \mu, \sigma^2$)를 Conv Layer의 Weight와 Bias에 수학적으로 병합(Fold)하는 함수 작성.
* **검증:** 수식 퓨전 이후 추출된 8-bit 파라미터로 NPU H/W 추론 시 98%+ 시뮬레이션 정확도(Accuracy) 유지 확인.
> **📂 수정 타겟 파일:**
> * `[수정]` `linux_software/cnn_test/cnn_train_and_export.py` (`train_cnn()`에 BatchNorm 추가, `export_binaries()`에 파라미터 퓨전 수식 반영)

### Step 3. 하드웨어 맞춤형(HW-Aware) NAS 형태 매핑 적용
컴파일러가 타겟팅할 최적화된 자료구조(타일링 포맷)를 설계합니다.
* **작업:** 기존 패딩 오버헤드가 발생하는 3x3 커널 매핑 로직을 소각하고, **4x4(16 element)** 단위로 확장.
* **통합:** 8x8 MAC 어레이에 빈틈없이 100% 맵핑되도록 메모리 정렬(Alignment) 구조 변경.
> **📂 수정 타겟 파일:**
> * `[수정]` `linux_software/cnn_test/cnn_train_and_export.py` (`K_H`, `K_W`를 4x4로 변경, `im2col` 패딩 사이즈 16 align 최적화)
> * `[수정]` `linux_software/cnn_test/main.c` (CNN Dimension 매크로 파라미터를 4x4 구조로 업데이트)

### Step 4. PyTorch FX 기반 기초 컴파일러 프로토타입 (Level 1)
MLIR이라는 거대한 시스템에 진입하기 전, Native 파이썬 환경에서 가장 빠르고 직관적인 초기 컴파일러를 구축합니다.
* **작업:** `torch.fx.symbolic_trace`를 사용해 모델 그래프 캡처 및 `[Conv2d] ➔ [BatchNorm2d] ➔ [ReLU]` 노드 패턴 파싱.
* **통합:** 찾아낸 3개의 노드를 압축하여 1개의 커스텀 노드(예: `NPU_Fused_Conv2d`)로 파이썬 단에서 강제 치환 및 가중치 추출.
> **📂 수정 타겟 파일:**
> * `[신규생성]` `linux_software/compiler/npu_fusion_pass.py` (PyTorch FX 기반 그래프 파싱 및 단순 Subgraph Pattern Matching)

### Step 5. MLIR 인프라 도입 및 TableGen (.td) 패턴 매칭 (Level 2)
구글/LLVM 진영의 산업 표준 프레임워크인 MLIR을 도입하여, 완벽한 선언적(Declarative) 컴파일러 아키텍처로 진화합니다.
* **작업:** PyTorch 모델을 `torch-mlir` 다리를 통해 MLIR(Linalg/TOSA 방언)로 변환(Lowering)하는 스크립트 작성.
* **통합 (`.td` 파일):** 파이썬 if-else 탐색을 폐기하고, **TableGen Description (`my_npu_fusion.td`)** 파일을 정의. `Pat<(ReLU (BatchNorm (Conv2d $x, $w, $b))), (MyNPU_Fused_Conv $x, $w, $b)>` 와 같이 직관적인 패턴 매칭 룰을 선언.
> **📂 수정 타겟 파일:**
> * `[신규생성]` `linux_software/compiler/mlir/my_npu_fusion.td` (NPU 전용 백엔드 그래프 퓨전 룰 선언 파일)
> * `[신규생성]` `linux_software/compiler/mlir/npu_dialect_converter.cpp` (TableGen이 자동 생성한 패턴 매칭을 구동시키는 C++ 패스)

### Step 6. MLIR EmitC 기반 C 코드 방출기 (Ultimate Codegen)
직접 기계어 어셈블러를 작성하는 LLVM 백엔드의 고통을 피하고, MLIR의 범용성(EmitC)을 활용해 DE10-Nano의 C 코드를 텍스트로 뽑아냅니다.
* **작업:** 매칭이 끝난 커스텀 노드(`MyNPU_Fused_Conv`)를 마주하면 MLIR `EmitC` 방언(Dialect)으로 번역하는 동작을 정의.
* **통합 (BYOC):** 최종 결과물로 `npu_load_weights(...)`, `npu_start_dma(...)` 등 **런타임 C 소스코드(.c) 자체를 MLIR이 텍스트로 기계식 출력(Codegen)** 하도록 설계. 뱉어진 C 파일은 DE10-Nano의 기존 GCC로 완벽하게 빌드(Compilation) 됨.
> **📂 수정 타겟 파일:**
> * `[신규생성]` `linux_software/compiler/mlir/npu_emit_c.cpp` (MLIR 단일 노드를 NPU Driver C 언어 문자열 템플릿으로 변환하여 `.c` 파일을 자동 출력(Print)하는 제너레이터)

---

## 📈 예상 산출물 (Expected Outcomes)
1. **Zero-Overhead Inference:** 기존에 밀리초 단위로 CPU 발목을 잡던 Bias/ReLU 연산시간 증발.
2. **100% Hardware Utilization:** 패딩과 오정렬로 인한 낭비 없이 64개의 MAC 어레이가 쉬지 않고 행렬 스칼라곱 수행.
3. **MLIR-Based BYOC (Bring Your Own Codegen):** 무거운 파이토치 환경 없이, MLIR 엔진이 던져준 최적화된 "C 실행 코드"와 ".bin 덩어리"만으로 임베디드 칩(DE10-Nano)이 무지막지한 속도로 엣지(Edge) 추론을 달성하는 상용급 파이프라인 완성.

---

## 🏆 현재까지 완료된 작업 (Completed Work)

### 1️⃣ Phase 1: 하드웨어 퓨전 파이프라인 완성 (RTL Level)
가장 시급했던 하드웨어 내부 직결(Hardwired) 구조 개편을 성공적으로 마쳤습니다. 메모리를 왕복하며 낭비되던 클럭을 제거하고, 연산 결과가 나오는 즉시 `후처리기(Post-Processor)`로 꽂히도록 **모든 데이터패스가 원-웨이(One-way) 퓨전 터널로 연결**되었습니다.

* **`npu_ocm_accumulator.v` 재설계**: 시스톨릭 어레이의 곱셈 결과(`pe_valid_out`)와 완벽하게 동기화되어 256-bit 출력을 1클럭 이내에 캐치합니다. 내부 256-bit FIFO(깊이 8)를 도입하여 어레이에서 쏟아지는 데이터 폭주 시에도 단 한 픽셀의 Drop 없이 32-bit OCM 포트에 안전하게 `Read-Modify-Write`를 수행합니다.
* **`npu_post_processor.v` 신규 개발**: OCM 누적이 끝나고 `DRAIN` 모드가 켜지는 즉시, 물 흐르듯 3가지 연산을 1클럭 파이프라인으로 수행합니다.
    1. **Bias 연산**: `npu_ctrl`에 신설된 2KB Bias SRAM 포트에서 Fused Bias를 읽어 덧셈 (+ $\beta$)
    2. **스케일링**: 8-bit 오른쪽 시프트연산 (Requantization)
    3. **활성화 함수**: ReLU (음수 클리핑)
* **결과**: `test_fusion.py`를 통해 파이썬 시뮬레이션(Numpy)과 완벽하게 100% 동일한 비트 단위 일치도(Bit-exact Accuracy)를 얻어냈으며, 모든 레거시 연산 코드들 환경에서도 Regression Testing(ALL PASS)을 증명했습니다.

### 2️⃣ Phase 2: PyTorch FX 소프트웨어 컴파일러 완성 (Python Level)
PyTorch 환경의 딥러닝 리서처가 하드웨어 동작 방식을 몰라도, 클릭 한 번에 퓨전된 파라미터 바이너리를 얻어낼 수 있는 브릿지 컴파일러(`npu_fusion_pass.py`)를 개발했습니다.

#### 💡 왜 무거운 외부 컴파일러(TVM/MLIR) 대신 PyTorch FX를 도입했는가?
과거에는 파이토치를 하드웨어에 매핑하기 위해 반드시 모델을 ONNX로 빼낸 뒤, TVM이나 MLIR 같은 거대 C++ 컴파일러 스택을 거쳐야 했습니다. 하지만 본 프로젝트에서는 **최신 PyTorch FX (Symbolic Tracing)** 엔진을 사용하여 이러한 무거운 중간 단계를 완전히 폐기(Bypass)하고 압도적인 생산성을 달성했습니다.
* **TVM의 한계:** 수백만 줄의 C++ 코드베이스를 파악하여, 우리만의 독자적인 NPU(8x8 어레이)에 맞게 퓨전 규칙이나 타일링 패스를 끼워넣는 작업은 개발 오버헤드가 극도로 큽니다.
* **PyTorch FX의 장점:** 파이썬 환경 안에서 곧바로 계산 그래프(Graph)를 뜯어보고 조작할 수 있습니다. 단 200여 줄의 파이썬 코드만으로 "그래프 스캔 $\to$ 파라미터 병합 $\to$ 하드웨어 맞춤형 8x8 텐서 타일링 $\to$ C 런타임 드라이버 방출"까지 이어지는 **독자기반 BYOC(Bring Your Own Codegen) 시스템**을 초고속으로 구축할 수 있었습니다.

* **FX Subgraph 매칭**: `torch.fx.symbolic_trace`를 통해 `[Conv2d -> BatchNorm2d -> ReLU]` 가 체인으로 엮인 서브그래프를 순식간에 찾아냅니다.
* **오프라인 파라미터 퓨전**: 찾아낸 배치 정규화(Batch Norm)의 파라미터($\gamma, \beta, \mu, \sigma^2$)를 수학적으로 가중치와 바이어스에 분배 흡수(Fold)시켜 무거운 BN 노드를 흔적도 없이 증발시켰습니다.
* **하드웨어 커스텀 노드 치환 (`NPU_Conv2d`)**: 흡수된 퓨전 네트워크를 기존 Conv 노드 자리에 강제로 끼워 넣어, PyTorch 상에서 즉시 NPU 하드웨어 에뮬레이션(`im2col`, `Tiling`, `Shift`, `ReLU Clipping`)을 통한 추론 단위 테스트 검증을 가능케 했습니다.
* **DMA용 바이너리 타일링 추출기 (`export_hw_params`)**: NPU 내부 8x8 MAC 어레이의 특이한 Z그재그 배선 구조와 MSGDMA의 1D 스트림 전송 조건에 맞추어, 4차원 파라미터 텐서를 정교한 **수학적 패딩(Multiple of 8) 및 타일링 매트릭스**로 썰어 `.bin` 덤프 파일로 직출력하는 기능을 갖추었습니다.
