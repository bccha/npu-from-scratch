# npu-from-scratch

DE10-Nano(Cyclone V SoC) FPGA 위에서 NPU를 밑바닥부터 설계하는 프로젝트.

## 로드맵

| 기간 | 단계 | 목표 |
|------|------|------|
| 3~4월 | Systolic Array | 4×4 MAC 어레이 RTL + 행렬곱 검증 |
| 5~6월 | Python Offload | `A @ B` → FPGA 계산 API |
| 7~8월 | TVM/MLIR | NPU 컴파일러 코드베이스 경험 |

자세한 내용 → [doc/ROADMAP.md](doc/ROADMAP.md)

## 하드웨어

- **보드:** Terasic DE10-Nano (Intel Cyclone V SoC)
- **FPGA:** Cyclone V SE 5CSEBA6U23I7
- **HPS:** ARM Cortex-A9 Dual-core @ 800MHz
- **베이스 프로젝트:** DE10-Nano SoC GHRD (Golden Hardware Reference Design)

## 프로젝트 구조

```
├── ip/                  # 커스텀 IP
├── nios_software/       # Nios II 소프트웨어
├── soc_system.qsys      # Qsys 시스템 설계
├── DE10_NANO_SoC_GHRD.v # 최상위 RTL
└── doc/                 # 문서
```
