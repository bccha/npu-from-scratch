---
trigger: always_on
---

iverilog나 cocotb 명령은 wsl을 이용하여 한다.
예) wsl -d Ubuntu-22.04 -e bash -l -c "cd sim && make -f Makefile_sequencer"

wire를 연결할때는 end to end 확인을 꼭 한다.

cocotb benchmark를 단위 module마다 작성한다