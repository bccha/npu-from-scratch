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
