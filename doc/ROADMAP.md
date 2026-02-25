# NPU 학습 로드맵

FPGA 기반 NPU 설계 경험을 쌓기 위한 단계별 로드맵.

---

## 2월: Systolic Array & DMA Offload — 행렬곱 가속기 통신 완성

**목표:** 8×8 MAC 어레이 RTL 구현 및 Linux/Python 행렬곱 벤치마크 완벽 구동

- [x] 8×8 Systolic Array MAC 어레이 RTL 구현 (Weight Stationary)
- [x] Avalon 컨트롤 및 스트리밍 제어 분리 (npu_ctrl, npu_stream_ctrl)
- [x] 패킷 송수신 신호 정밀화 (EOP Batch Streaming) 보완
- [x] Linux `/dev/mem` 및 Python `mmap`으로 Numpy/C-Driver 결과 직접 검증 및 벤치마크 (완료)

**결과물:** "행렬곱 가속기 스택 (H/W + S/W)" 조기 완성. Bare-metal(Nios II) 및 리눅스(ARM Cortex-A9) 완벽 호환 구현 달성.

---

## 3~4월: TVM/MLIR 분석 — NPU 컴파일러 경험 연동

**목표:** TVM 코드베이스 분석 및 FPGA 백엔드 연결 실험 (가속기 통합 연동)

- [ ] TVM 설치 + Relay IR → TIR 변환 단계 집중 분석
- [ ] Operator Fusion 패스 하나 직접 추가
- [ ] BYOC(Bring Your Own Codegen)로 Python 행렬곱 가속 API 백엔드 붙이기 실험
- [ ] LLVM 경험과 연결

**결과물:** "NPU 컴파일러 코드베이스 기반 AI 연동" 완성

---

## 5~6월: 구조 고도화 — 하드웨어 및 NPU 역량 확장

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
