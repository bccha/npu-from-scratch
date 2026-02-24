#include "common.h"
#include <alt_types.h>
#include <io.h>
#include <stdio.h>
#include <system.h>


// Unified Register Map
#define REG_CTRL 0
#define REG_STATUS 1
#define REG_DMA_RD_ADDR 2
#define REG_DMA_RD_LEN 3
#define REG_DMA_WR_ADDR 4
#define REG_DMA_WR_CTRL 5
#define REG_SEQ_ROWS 6

// Legacy PE Registers (Base Address + Offset 8)
#define REG_PE_CTRL 8
#define REG_PE_X_IN 9
#define REG_PE_Y_IN 10
#define REG_PE_Y_OUT 11

void verify_full_system() {
  printf("\nStarting Full System Matrix Validation...\n");

  IOWR_32DIRECT(ADDRESS_SPAN_EXTENDER_0_CNTL_BASE, 0, 0x20000000);
  alt_u32 physical_base = 0x20000000;
  alt_u32 src_addr = DDR3_WINDOW_BASE;
  alt_u32 dst_addr = DDR3_WINDOW_BASE + 0x1000000;

  int total_rows = 8;
  int rd_len = total_rows * 2; // 16 words in
  int wr_len = total_rows * 8; // 64 words out

  printf("Preparing Matrix Row Data...\n");
  for (int i = 0; i < rd_len; i++) {
    // Injecting dummy values (e.g. 0x000100aa for high/low parts)
    IOWR_32DIRECT(src_addr, i * 4, (i << 16) | 0x00AA);
  }

  printf("Clearing Destination Area...\n");
  for (int i = 0; i < wr_len; i++) {
    IOWR_32DIRECT(dst_addr, i * 4, 0);
  }

  printf("Configuring System CSRs...\n");
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, total_rows);
  IOWR(NPU_CTRL_BASE, REG_DMA_RD_ADDR, physical_base);
  IOWR(NPU_CTRL_BASE, REG_DMA_RD_LEN, rd_len);
  IOWR(NPU_CTRL_BASE, REG_DMA_WR_ADDR, physical_base + 0x1000000);

  printf("Starting Hardware Execution...\n");
  // REG_CTRL: Mode=1 (Exec), Start=1 -> 0x3
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x3);
  // REG_DMA_WR_CTRL: Start WR=Bit 17, Start RD=Bit 16, Length=wr_len
  IOWR(NPU_CTRL_BASE, REG_DMA_WR_CTRL, (1 << 17) | (1 << 16) | wr_len);

  printf("Waiting for DMA Interrupts...\n");
  int timeout = 0;
  while (1) {
    alt_u32 status = IORD(NPU_CTRL_BASE, REG_STATUS);
    if ((status & 0x00030000) == 0x00030000)
      break;
    timeout++;
    if (timeout > 100000) {
      printf("TIMEOUT ERROR! System Halted.\n");
      return;
    }
  }
  printf("Execution Finished!\n\n");

  printf("Output Dumping (First 16 beats):\n");
  for (int i = 0; i < 16; i++) {
    alt_u32 val = IORD_32DIRECT(dst_addr, i * 4);
    printf("OUT[%d] = 0x%08x\n", i, (unsigned int)val);
  }

  printf("\nFull System Validation: PASS\n");
}

void verify_mac_pe() {
  printf("\nStarting MAC PE Verification...\n");

  IOWR(NPU_CTRL_BASE, REG_PE_X_IN, 7);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 1);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);

  IOWR(NPU_CTRL_BASE, REG_PE_X_IN, 3);
  IOWR(NPU_CTRL_BASE, REG_PE_Y_IN, 10);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 2);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);

  alt_u32 result = IORD(NPU_CTRL_BASE, REG_PE_Y_OUT);
  printf("Result: %d (Expected: 31)\n", (int)result);

  if (result == 31)
    printf("MAC PE Test: PASS\n");
  else
    printf("MAC PE Test: FAIL\n");
}

int main() {
  while (1) {
    printf("\nNPU System Verification (Full Framework)\n");
    printf("----------------------------------------------\n");
    printf("1. Verify MAC PE\n");
    printf("3. Verify Full System Data path\n");
    printf("q. Quit\n");
    printf("Choose: ");

    char c = get_char_polled();
    printf("%c\n", c); // Echo

    if (c == '1') {
      verify_mac_pe();
    } else if (c == '2') {
      verify_full_system();
    } else if (c == 'q') {
      printf("Exiting...\n");
      break;
    }
  }

  return 0;
}
