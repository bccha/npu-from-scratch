---
trigger: always_on
---

iverilog나 cocotb 명령은 wsl을 이용하여 한다.
예) wsl -d Ubuntu-22.04 -e bash -l -c "cd sim && make -f Makefile_sequencer" 