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
  printf("\nStarting Full System Matrix Validation (4x4)...\n");

  IOWR_32DIRECT(ADDRESS_SPAN_EXTENDER_0_CNTL_BASE, 0, 0x20000000);
  alt_u32 physical_base = 0x20000000;
  alt_u32 weights_addr = DDR3_WINDOW_BASE;
  alt_u32 inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  alt_u32 dst_addr = DDR3_WINDOW_BASE + 0x2000;

  int total_rows = 4;
  int rd_len = total_rows * 1; // 4 32-bit beats (1 per row)
  int wr_len = total_rows * 4; // 16 32-bit beats (4 per row: 4*32 / 32)

  // C Test: Weights Matrix = Identity Matrix (Diagonal 1s) for 4x4
  // 32-bit beat = [Col3, Col2, Col1, Col0] (8-bit each)
  alt_u32 weights_32[4];
  weights_32[0] = 0x01000000; // Col 3 (Row 0)
  weights_32[1] = 0x00010000; // Col 2
  weights_32[2] = 0x00000100; // Col 1
  weights_32[3] = 0x00000001; // Col 0

  // Input Matrix = 1, 2, 3... 16
  alt_u32 inputs_32[4] = {
      0x04030201, // row 0: 1-4
      0x08070605, // row 1: 5-8
      0x0C0B0A09, // row 2: 9-12
      0x100F0E0D  // row 3: 13-16
  };

  printf("Preparing Matrix Row Data into DDR3...\n");
  for (int i = 0; i < 4; i++) {
    IOWR_32DIRECT(weights_addr, i * 4, weights_32[i]);
    IOWR_32DIRECT(inputs_addr, i * 4, inputs_32[i]);
  }

  printf("Clearing Destination Area...\n");
  for (int i = 0; i < 16; i++) {
    IOWR_32DIRECT(dst_addr, i * 4, 0);
  }

  printf("Phase 1: Loading Weights...\n");
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, (alt_u32)total_rows);
  IOWR(NPU_CTRL_BASE, REG_DMA_RD_ADDR, (alt_u32)physical_base);
  IOWR(NPU_CTRL_BASE, REG_DMA_RD_LEN, (alt_u32)rd_len);

  // REG_CTRL: Mode=0 (Load Weight), Start=1 -> 0x1
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000001);
  // REG_DMA_WR_CTRL: Start RD=Bit 16
  IOWR(NPU_CTRL_BASE, REG_DMA_WR_CTRL, 0x00010000);

  int timeout = 0;
  while (1) {
    alt_u32 status = IORD(NPU_CTRL_BASE, REG_DMA_WR_CTRL);
    if ((status & 0x00010000) == 0x00010000) // RD Done bit is 16
      break;
    timeout++;
    if (timeout > 100000) {
      printf("TIMEOUT ERROR! Weight Load Halted.\n");
      return;
    }
  }
  printf("Weights Loaded! DMA Status = 0x%08x\n",
         (unsigned int)IORD(NPU_CTRL_BASE, REG_DMA_WR_CTRL));

  printf("Phase 2: Execution...\n");
  IOWR(NPU_CTRL_BASE, REG_DMA_RD_ADDR, physical_base + 0x1000);
  IOWR(NPU_CTRL_BASE, REG_DMA_WR_ADDR, physical_base + 0x2000);

  // REG_CTRL: Mode=1 (Exec), Start=1 -> 0x3
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x3);
  // REG_DMA_WR_CTRL: Start WR=Bit 17, Start RD=Bit 16, Length=wr_len
  IOWR(NPU_CTRL_BASE, REG_DMA_WR_CTRL, (1 << 17) | (1 << 16) | wr_len);

  timeout = 0;
  while (1) {
    alt_u32 status = IORD(NPU_CTRL_BASE, REG_DMA_WR_CTRL);
    if ((status & 0x00030000) == 0x00030000) // RD/WR Done (bits 16, 17)
      break;
    timeout++;
    if (timeout > 100000) {
      printf("TIMEOUT ERROR! Execution Halted.\n");
      return;
    }
  }
  printf("Execution Finished! DMA Status = 0x%08x\n\n",
         (unsigned int)IORD(NPU_CTRL_BASE, REG_DMA_WR_CTRL));

  int errors = 0;

  printf("Dumping Full 4x4 Output Matrix:\n");
  for (int r = 0; r < 4; r++) {
    printf("Row %d: [", r);
    for (int c = 0; c < 4; c++) {
      alt_u32 hw_val = IORD_32DIRECT(dst_addr, (r * 4 + c) * 4);
      printf("%3d", (int)hw_val);
      if (c < 3)
        printf(", ");
    }
    printf("]\n");
  }

  printf("\nVerifying Output (Expecting Y=X if Weights=Identity)...\n");
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      alt_u32 hw_val = IORD_32DIRECT(dst_addr, (r * 4 + c) * 4);
      alt_u32 np_val = r * 4 + c + 1; // 1 to 16
      if (hw_val != np_val) {
        printf("Mismatch at [%d, %d]: HW=%d, Expected=%d\n", r, c, (int)hw_val,
               (int)np_val);
        errors++;
      }
    }
  }

  if (errors == 0) {
    printf("\nFull System Validation: PASS! All %d elements matched.\n", 16);
  } else {
    printf("\nFull System Validation: FAIL (%d errors)\n", errors);
  }
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
    printf("2. Verify Full System Data path\n");
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
