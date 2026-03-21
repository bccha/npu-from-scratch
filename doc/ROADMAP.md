# NPU 학습 로드맵

FPGA 기반 NPU 설계 경험을 쌓기 위한 단계별 로드맵.

---

## Phase 1: Systolic Array & DMA Offload — 행렬곱 가속기 통신 완성 (완료)

**목표:** 8×8 MAC 어레이 RTL 구현 및 Linux/Python 행렬곱 벤치마크 완벽 구동

- [x] 8×8 Systolic Array MAC 어레이 RTL 구현 (Weight Stationary)
- [x] Avalon 컨트롤 및 스트리밍 제어 분리 (npu_ctrl, npu_stream_ctrl)
- [x] 패킷 송수신 신호 정밀화 (EOP Batch Streaming) 보완
- [x] Linux `/dev/mem` 및 Python `mmap`으로 Numpy/C-Driver 결과 직접 검증 및 벤치마크 (완료)

**결과물:** "행렬곱 가속기 스택 (H/W + S/W)" 조기 완성. Bare-metal(Nios II) 및 리눅스(ARM Cortex-A9) 완벽 호환 구현 달성.

---

## Phase 2: MNIST & CNN 성능 한계 돌파 (진행 중)

**목표:** 최신 NPU 기술인 Quantization Aware Training (QAT) 도입 및 합성곱 연산 가속화

- [ ] Python 학습 파이프라인에서 QAT를 통한 int8/int32 Symmetric Quantization 오차 최소화
- [ ] Python Export 스크립트에 Batch Norm 오프라인 Fusion (Weights/Bias 병합) 로직 추가
- [ ] 하드웨어 Post-Processor(PP) RTL 파이프라인 설계 및 Avalon-ST Stream 체인 연동 (Bias, Shift, ReLU 1클럭 처리)
- [ ] C 드라이버(`main.c`)의 CPU Accumulation 병목 루프 삭제 및 DMA 8-bit 정밀 수신 최적화
- [ ] HW-Aware NAS 적용: 3x3 필터를 4x4(16 elements)로 확장하여 8x8 MAC 어레이 100% 활용 달성 (Padding 오버헤드 제로화)
- [ ] 2-Layer 모델 최적화 배포 (실제 정확도 추론 98%+ 달성 및 NPU H/W 100% 가속)

**결과물:** "실전 추론 NPU 딥러닝 정확도 및 성능 최적화 파이프라인" 완성

---

## Phase 3: TVM/MLIR 분석 — NPU 컴파일러 광속 연동

**목표:** TVM 코드베이스 분석 및 FPGA 백엔드 연결 실험 (가속기 통합 연동)

- [ ] TVM 설치 + Relay IR → TIR 변환 단계 집중 분석
- [ ] Operator Fusion 패스 하나 직접 추가
- [ ] BYOC(Bring Your Own Codegen)로 Python 행렬곱 가속 API 백엔드 붙이기 실험
- [ ] LLVM 경험과 연결

**결과물:** "NPU 컴파일러 코드베이스 기반 AI 연동" 완성

---

## Phase 4: 구조 고도화 — 하드웨어 및 NPU 역량 확장

**목표:** SRAM 버퍼 구조 확장 및 Activation 추가 지원

- [ ] 대용량 Global Buffer (SRAM) 온칩 확장 및 DMA 제어 구조 최적화
- [ ] Activation (ReLU, Sigmoid) 하드웨어 로직 파이프라인 연계
- [ ] Pooling/Conv2D 스케줄링 확장 고려

**결과물:** "상용 NPU 급 연산 유닛 (Conv, Act)" 베이스라인 추가 경험 달성

---

## 전체 흐름

```
RTL 설계 (Systolic Array)
    ↓
Python 인터페이스 (Offload API)
    ↓
컴파일러 레이어 (TVM/MLIR)
```

실제 NPU 스택의 bottom-up 학습 경로와 일치.
