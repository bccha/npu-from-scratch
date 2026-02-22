# NPU 학습 로드맵

FPGA 기반 NPU 설계 경험을 쌓기 위한 단계별 로드맵.

---

## 3~4월: Systolic Array — 행렬곱 가속기

**목표:** 4×4 MAC 어레이 RTL 구현 및 행렬곱 검증

- [ ] 4×4 Systolic Array MAC 어레이 RTL 구현
  - Dataflow 방식 결정: weight stationary vs. output stationary
- [ ] Tiling 로직 구현 (ARM Python에서 (M,K)@(K,N) 분할)
- [ ] BRAM weight 로드 (기존 DMA 재활용)
- [ ] Python numpy로 결과 검증

**결과물:** "행렬곱 가속기" 완성

---

## 5~6월: Python Offload — FPGA 코프로세서

**목표:** Python에서 `A @ B` 호출 시 내부적으로 FPGA가 계산하는 인터페이스 구현

- [ ] Python C Extension 또는 pybind11/cffi로 래핑
- [ ] numpy 스타일 API 설계 (`A @ B` → FPGA DMA 전송 → 결과 수신)
- [ ] DMA 전송 레이턴시를 의식한 설계
- [ ] CPU vs FPGA 성능 벤치마크

**결과물:** "FPGA 코프로세서" 완성

---

## 7~8월: TVM/MLIR 분석 — NPU 컴파일러 경험

**목표:** TVM 코드베이스 분석 및 FPGA 백엔드 연결 실험

- [ ] TVM 설치 + Relay IR → TIR 변환 단계 집중 분석
- [ ] Operator Fusion 패스 하나 직접 추가
- [ ] BYOC(Bring Your Own Codegen)로 FPGA 백엔드 붙이기 실험
- [ ] LLVM 경험과 연결

**결과물:** "NPU 컴파일러 코드베이스 경험" 추가

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
