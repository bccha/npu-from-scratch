# npu-from-scratch

DE10-Nano(Cyclone V SoC) FPGA 위에서 NPU를 밑바닥부터 설계하는 프로젝트.
<p align="center">
  <img src="doc/assets/image.png" width="400">
</p>

## 로드맵

| 구분 | 단계 | 목표 | 
|------|------|------|
| Phase 1 | Systolic Array & Linux DMA | 8×8 MAC 어레이 RTL 구현, Linux C/Python 벤치마크 시스템 오프로딩 **[완료]** | 
| Phase 2 | TVM/MLIR 컴파일러 연동 | NPU 컴파일러 코드베이스 경험 및 커스텀 오퍼레이터 매핑 |
| Phase 3 | 하드웨어 고도화 | SRAM 버퍼 크기 증가 및 지원 연산 (Activation, Pooling 등) 확장 설계 |

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
├── linux_software/      # Linux ARM HPS 소프트웨어 (벤치마크)
├── rtl/                 # NPU 하드웨어 로직 (Verilog)
├── sim/                 # Python 베이스 Cocotb 시뮬레이션
├── soc_system.qsys      # Qsys 시스템 설계
└── doc/                 # 문서 및 설계 자료
```
## 문서 목차 (Documentation Index)

프로젝트와 관련된 상세 기획, 구조, 맵 구조, 트러블슈팅 내역은 아래 문서들을 참고해 주십시오.

*   [**ROADMAP.md**](doc/ROADMAP.md): 프로젝트 전체 로드맵 및 단계별 세부 구현 목표.
*   [**DESIGN.md**](doc/DESIGN.md): (초기) NPU 아키텍처 및 기본 설계 방향 문서.
*   [**DESIGN_2ND.md**](doc/DESIGN_2ND.md): (현재) Avalon-ST (Streaming) 기반의 Bufferless NPU 아키텍처 및 MSGDMA 직렬화 파이프라인 최신 설계 사양.
*   [**REG_MAP.md**](doc/REG_MAP.md): NPU(`npu_ctrl`) 및 DMA(MSGDMA CSR/Descriptor) 관련 Memory-mapped 레지스터 맵 및 C 코드 제어 예제.
*   [**RESULT.md**](doc/RESULT.md): CPU vs NPU (FPGA) 간의 성능 검증 벤치마크 결과 및 프로파일링 데이터.
*   [**LESSONS_LEARNED.md**](doc/LESSONS_LEARNED.md): 설계 과정 및 디버깅 중 얻은 기술적 교훈 (Quartus, SoC, MSGDMA 관련).
*   [**STUDY.md**](doc/STUDY.md): 딥러닝 가속기(Systolic Array 등) 구조 및 관련 개념 학습 정리.
*   [**ISSUE_4x4_DUPLICATION.md**](doc/ISSUE_4x4_DUPLICATION.md): 초기 NPU 파이프라인에서 발생했던 출력 중복(Duplication) 하드웨어 버그 및 그 해결 과정 상세.

## 핵심 기술 성과 (Key Technical Achievements)

이 프로젝트는 단순한 "시뮬레이션용 행렬 곱셈기"를 넘어, 실제 **고성능 시스템 반도체(SoC) 환경에서 요구되는 풀스택 아키텍처 현업 설계 기법**을 토대로 개발되었습니다.

### 1. Decoupled & Buffer-less Streaming Pipeline (Avalon-ST)
*   일반적인 Memory-Mapped (대용량 Block RAM 핑퐁 버퍼) 구조의 한계를 탈피하고, NPU 제어(Avalon-MM)와 데이터 전송(Avalon-ST)을 완전히 분리했습니다.
*   연산 코어(`systolic_core.v`) 내부에 거대한 SRAM을 두지 않고, MSGDMA로부터 유입되는 데이터 스트림을 실시간으로 소모하는 **순수 연산 중심의 Buffer-less 아키텍처**를 구현하여 면적 효율(Area Efficiency)을 극대화했습니다. 
*   하위 DMA의 `ready=0` 패킷 밀림 상황에서도 데이터 유실이나 중복이 발생하지 않는 엄격한 `valid/ready` 백프레셔(Backpressure) 핸드셰이킹 로직을 100% 지원합니다.

### 2. 하드웨어 타이밍(CDC/FSM) 및 출력 중복 버그 완전 디버깅
*   256-bit (8x8 가로축) 병렬 연산 결과를 시스템 버스 격인 64-bit 플릿(Flit) 4개로 쪼개어(Serialization) 전송하는 `npu_stream_ctrl.v`를 독자 개발했습니다.
*    초기 4x4 및 8x8 파이프라인에서 시뮬레이터는 통과했으나 실제 보드 환경의 타이밍 밀림으로 인해 출력 데이터가 중복 캡처되던 치명적인 H/W 버그를 겪었습니다. 이를 FSM 상태 전이 타이밍과 Shift Register의 파이프라인 정렬(Alignment)을 클럭 레벨에서 재설계하여 하드웨어 단에서 원천 차단했습니다.
*   다중 행렬(Batch) 처리 시 가장 마지막 프레임에만 정확히 EOP(End of Packet) 플래그를 발생시켜, MSGDMA가 조기 종료되지 않고 1000개 이상의 트랜잭션을 병목 없이 끊김없이 고속 스트리밍(Direct Memory Access)하는 데 성공했습니다.

### 3. Full-Stack End-to-End System Integration (Verilog to Linux S/W)
*   단순히 Verilog RTL 단위 설계에 그치지 않고, Qsys(Platform Designer)를 통한 AXI 버스 연결 아키텍처를 주도적으로 구성했습니다.
*   **Linux ARM HPS (Cortex-A9)** 환경에서 물리 메모리(`0x20000000`)와 LWH2F Avalon 버스 브릿지(`0xFF200000`)를 `/dev/mem` 및 `mmap()`을 활용하여 가상 메모리(Virtual Memory)로 끌어와 IP를 직접 제어하는 **User-space C 드라이버(API)를 스크래치부터 구현**했습니다.
*   **성능 실측 (Benchmarking):** CPU(`gcc -O2` 최적화 3중 for문)와 FPGA 하드웨어 (MSGDMA 오프로드) 상에서 동일한 4000 Batch 연산을 수행하고 `gettimeofday` 단위로 엄밀하게 비교한 결과, **50MHz NPU가 800MHz 듀얼코어 프로세서 대비 약 4.64배 (4.64x) 빠른 압도적인 실행 속도**를 냄을 실제 환경에서 정량적으로 증명했습니다. (DMA 세팅 오버헤드 포함)

## AI-Assisted Development Journey (단 4일간의 여정)

**🚀 개발 기간: 2026년 2월 22일 ~ 2026년 2월 25일 (총 4일)**

이 프로젝트는 처음부터 끝까지 **LLM(AI 에이전트) 기반의 애자일(Agile)한 Hardware-Software Co-design** 방법론을 한껏 활용하여, 단 4일 만에 초기 아키텍처 구상부터 Linux 벤치마크 완료까지 진행되었습니다. 단순한 AI 코드 생성을 넘어, AI를 '시니어 페어 프로그래머(Pair Programmer)'로 삼아 다음과 같은 고난이도 엔지니어링 맹점들을 돌파했습니다.

1.  **Cocotb 기반의 Python 시뮬레이션 환경 구축 및 컴파일 시간의 획기적 단축:**
    기존 FPGA 워크플로우의 가장 큰 병목인 **"수십 분이 걸리는 Quartus 전체 컴파일타임"**을 최소화하는 것이 핵심이었습니다. AI를 활용해 Python 기반 **Cocotb** 테스트 코드를 대거 자동 생성하여, NPU의 입력 행렬 주입과 `Numpy` 골든 모델(Golden Model) 기대값(Expected) 비교를 Python 코드로 완벽하게 자동화했습니다. 결과적으로, 번거로운 보드 합성 과정 없이 회로 로직의 결함을 초 단위 시뮬레이션으로 즉각 수백 번 반복 테스트하며 **개발 사이클을 기하급수적으로 단축**할 수 있었습니다.
2.  **클럭 단위의 치열한 RTL FSM 디버깅 제안:**
    위에서 언급한 Data Duplication 버그 발생 당시, 캡처된 Waveform과 클럭 사이의 델타 사이클 증상을 AI에게 묘사하고 함께 시나리오를 구성했습니다. AI는 Serialization FSM 단계에서 Ready가 떨어지기도 전에 FIFO 포인터가 넘어가버리는 현상을 정확히 짚어냈고, 이를 기반으로 상태 머신 제어 코드를 즉각 리팩터링할 수 있었습니다.
3.  **MSGDMA 공식 스펙 분석 및 C S/W 드라이버 매크로 구현:**
    수천 페이지에 달하는 Intel/Altera MSGDMA 및 SoC 매뉴얼 사이에서 헤맬 필요 없이, AI에게 목적(SOP/EOP 생성 및 폴링 방법 등)을 제시하고 핵심 Register Map과 Bit 필드 정보를 빠르고 정확하게 도출했습니다. 이를 통해 Linux C 코드의 포인터 오프셋과 `mmap` 제어 코드를 단숨에 완성했습니다.

본 프로젝트는 전통적 시스템 반도체 설계에 AI(Agentic Workflow)를 결합했을 때, 1인 개발자가 H/W Architecture 설계부터 S/W System Integration까지 얼마나 강력하고 완벽한 풀스택 결과물을 딥(Deep)하게 만들어낼 수 있는지를 보여주는 최상의 사례입니다.
