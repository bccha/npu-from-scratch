# NPU AI 프로그래밍 디자인 가이드 (DESIGN_AI.md)

이 문서는 8x8 하드웨어 행렬 승산기(MAC Array) 위에서 실행되는 **신경망(Neural Network) 소프트웨어 매핑 아키텍처**를 설명합니다.

현재 NPU가 검증 완료한 딥러닝 아키텍처인 **MNIST 2-Layer MLP (Multi-Layer Perceptron)**의 레이어별 사양은 다음과 같습니다.

---

## 1. 2-Layer MLP (MNIST 필기체 인식) 아키텍처

MNIST 데이터셋은 $28 \times 28$ 크기의 흑백 이미지를 사용합니다. 이를 1차원 벡터로 강제 전개(Flatten)하여 784개의 픽셀 데이터를 NPU로 흘려보냅니다.

### 전체 파이프라인 구조
```text
Input Image (28x28 = 784) 
      ↓
[Layer 1] Fully Connected (Dense) + ReLU Activation
      ↓
Hidden Features (64)
      ↓
[Layer 2] Fully Connected (Dense) -> Argmax
      ↓
Output Classes (10: 0~9 확률)
```

---

### Layer 1 상세 분석 (입력층 $\rightarrow$ 은닉층)

첫 번째 신경망 레이어는 원본 이미지의 특징을 추출하여 64차원의 중간 벡터(Hidden State)로 변환합니다.

*   **Input ($X$):** `[1, 784]` (하나의 이미지가 들어있는 1D 벡터)
*   **Weights ($W_1$):** `[784, 64]`
*   **Bias ($B_1$):** `[1, 64]`
*   **수식:** $H = \text{ReLU}(X \times W_1 + B_1)$

**[NPU 통신 오프로드 (MSGDMA)]**
NPU는 $8 \times 8$ 타일로만 연산이 가능합니다. C 드라이버(`mnist_test/main.c`)는 소프트웨어적으로 `[1, 784]` 벡터를 행렬곱이 가능하도록 형태를 맞추고, $W_1$ 트랜잭션을 8개 단위로 끊어서 MSGDMA로 밀어 넣습니다.
*   연산이 끝나면 NPU는 `[1, 64]` 크기의 중간 결과를 HPS 메모리로 뱉어냅니다. 
*   C CPU 코드는 이 메모리 배열을 읽어, 값이 0 이하인 경우 0으로 쳐내는 **소프트웨어 ReLU (Activation)** 연산을 수행합니다.

---

### Layer 2 상세 분석 (은닉층 $\rightarrow$ 출력층)

두 번째 신경망 레이어는 Layer 1에서 추출한 64차원 특징을 바탕으로 최종 0~9 번호표 레이블을 도출합니다.

*   **Input ($H$):** `[1, 64]` (ReLU를 거친 1D 벡터)
*   **Weights ($W_2$):** `[64, 10]`
*   **Bias ($B_2$):** `[1, 10]`
*   **수식:** $Y = H \times W_2 + B_2$

**[차원 패딩 (Zero-Padding)의 중요성]**
출력 클래스는 숫자 0부터 9까지 총 10개이므로, 최종 가중치 행렬 $W_2$는 `[64, 10]` 크기를 갖습니다.
하지만 NPU는 하드웨어적으로 **무조건 8의 배수 단위(8x8 블록)**로만 통신합니다. 따라서 8의 배수인 `16`으로 강제 확장이 필요합니다.
*   Python 스크립트(`train_and_export.py`) 단계에서 $W_2$를 **`[64, 16]`** 크기로 우측을 0(Zero)으로 채워 넣은 뒤 `.bin` 파일로 추출합니다.
*   NPU 연산 후 최종 배열은 `[1, 16]` 크기의 포인터 배열로 수신되며, 이 중 인덱스 0번부터 9번까지만 꺼내어 `argmax` (가장 확률이 높은 값)를 검출하면 최종 필기체 예측이 완료됩니다.

---

## 2. 8비트 양자화 모델 제약 및 QAT 도입 필요성

현재 위 아키텍처의 정확도는 CPU Float32 기준 99%에 달하지만, NPU 탑재용으로 변환한 결과 88.08%의 정확도를 기록했습니다. 이 차이는 하드웨어 한계에서 비롯됩니다.

*   **PTQ (Post-Training Quantization)의 한계:** NPU는 내부적으로 효율적인 Int8 (8비트 정수) 곱셈만을 지원합니다. 현재 소프트웨어가 학습을 마친 가중치(Float32)를 강제로 8비트 범위(-128~127)로 변환하고 소수점을 버리는(Clipping) 단방향 변환 구조를 갖추었기 때문에 미세한 데이터 손실이 누적됩니다.
*   **QAT (Quantization-Aware Training) 솔루션:** 구글(Google Edge TPU) 등의 업계 표준 방식입니다. 이를 극복하려면 Python 학습(`train_and_export.py`) 단계의 Forward 파이프라인 내부에서부터, H/W의 `Target Scale` 나누기 연산 및 `Clip` 현상을 모델이 시뮬레이션(Fake Quantization)하면서 가중치를 업데이트 하도록 재작성해야 합니다.

---

### 다음 단계: CNN (합성곱 신경망) 아키텍처

CNN(합성곱 신경망)은 특징 맵(Feature Map)을 유지하면서 필터(커널)를 슬라이딩시키는 구조입니다. $8 \times 8$ GEMM 구조의 단일 Matrix H/W에서 이를 처리하기 위해서는, 입력 이미지를 강제로 퍼즐 조각 내듯 길게 전개시키는 **`im2col` (Image-to-Column)** 기술이 S/W 드라이버 계층에 추가로 설계되어야 합니다. (진행 예정)
