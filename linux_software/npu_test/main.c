#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>


// DE10-Nano LWHPS2FPGA Bridge Base
#define LWHPS2FPGA_BASE 0xFF200000
#define LWHPS2FPGA_SPAN 0x00200000

// NPU Offset (from system.h)
#define NPU_OFFSET 0x00030000

// Register Offsets (Word indices)
#define REG_CTRL 0
#define REG_STATUS 1
#define REG_A_DATA 2
#define REG_B_DATA 3
#define REG_ACC_IN 4
#define REG_RES 5
#define REG_C_BASE 16

volatile uint32_t *npu_regs = NULL;

void verify_mac(int8_t a, int8_t b, int32_t acc_in) {
  int32_t expected = acc_in + (a * b);
  printf("[MAC] %d * %d + %d\n", a, b, acc_in);

  npu_regs[REG_A_DATA] = (uint32_t)a;
  npu_regs[REG_B_DATA] = (uint32_t)b;
  npu_regs[REG_ACC_IN] = (uint32_t)acc_in;
  npu_regs[REG_CTRL] = (0 << 2) | (1 << 1); // Mode 00, valid_in

  int32_t result = (int32_t)npu_regs[REG_RES];
  printf("  -> Result: %d (Exp: %d) [%s]\n", result, expected,
         (result == expected) ? "PASS" : "FAIL");
}

void verify_dot4(uint32_t a_vec, uint32_t b_vec, int32_t acc_in) {
  int8_t a0 = (int8_t)(a_vec & 0xFF), a1 = (int8_t)((a_vec >> 8) & 0xFF),
         a2 = (int8_t)((a_vec >> 16) & 0xFF),
         a3 = (int8_t)((a_vec >> 24) & 0xFF);
  int8_t b0 = (int8_t)(b_vec & 0xFF), b1 = (int8_t)((b_vec >> 8) & 0xFF),
         b2 = (int8_t)((b_vec >> 16) & 0xFF),
         b3 = (int8_t)((b_vec >> 24) & 0xFF);
  int32_t expected = acc_in + (int32_t)a0 * b0 + (int32_t)a1 * b1 +
                     (int32_t)a2 * b2 + (int32_t)a3 * b3;

  printf("[Dot4] A:0x%08X, B:0x%08X\n", a_vec, b_vec);
  npu_regs[REG_A_DATA] = a_vec;
  npu_regs[REG_B_DATA] = b_vec;
  npu_regs[REG_ACC_IN] = (uint32_t)acc_in;
  npu_regs[REG_CTRL] = (1 << 2) | (1 << 1);

  while (!(npu_regs[REG_STATUS] & 1))
    ;

  int32_t result = (int32_t)npu_regs[REG_RES];
  printf("  -> Result: %d (Exp: %d) [%s]\n", result, expected,
         (result == expected) ? "PASS" : "FAIL");
}

void verify_matmul_4x4() {
  printf("[4x4 MatMul] Random Matrix Test\n");
  int8_t A[4][4], B[4][4];
  int32_t C_sw[4][4];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      A[i][j] = (rand() % 21) - 10;
      B[i][j] = (rand() % 21) - 10;
      C_sw[i][j] = 0;
    }
  }

  // SW model
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++)
        C_sw[i][j] += (int32_t)A[i][k] * B[k][j];
    }
  }

  // NPU HW
  npu_regs[REG_CTRL] = (1 << 0); // Start (Clear)
  for (int k = 0; k < 4; k++) {
    uint32_t a_col = (uint8_t)A[0][k] | ((uint8_t)A[1][k] << 8) |
                     ((uint8_t)A[2][k] << 16) | ((uint8_t)A[3][k] << 24);
    uint32_t b_row = (uint8_t)B[k][0] | ((uint8_t)B[k][1] << 8) |
                     ((uint8_t)B[k][2] << 16) | ((uint8_t)B[k][3] << 24);
    npu_regs[REG_A_DATA] = a_col;
    npu_regs[REG_B_DATA] = b_row;
    npu_regs[REG_CTRL] = (2 << 2) | (1 << 1);
  }

  while (!(npu_regs[REG_STATUS] & 1))
    ;

  int pass_count = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int32_t res = (int32_t)npu_regs[REG_C_BASE + (i * 4 + j)];
      if (res == C_sw[i][j])
        pass_count++;
    }
  }
  printf("  -> Result: %d/16 Elements Correct [%s]\n", pass_count,
         (pass_count == 16) ? "PASS" : "FAIL");
}

int main() {
  int fd;
  void *virtual_base;

  if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1) {
    perror("Error: could not open /dev/mem");
    return 1;
  }

  virtual_base = mmap(NULL, LWHPS2FPGA_SPAN, (PROT_READ | PROT_WRITE),
                      MAP_SHARED, fd, LWHPS2FPGA_BASE);
  if (virtual_base == MAP_FAILED) {
    perror("Error: mmap() failed");
    close(fd);
    return 1;
  }

  npu_regs = (uint32_t *)((uint8_t *)virtual_base + NPU_OFFSET);

  printf("--- Linux NPU Full Verification ---\n");
  verify_mac(7, -6, 100);
  verify_dot4(0x01020304, 0x01010101, 10);
  verify_matmul_4x4();

  if (munmap(virtual_base, LWHPS2FPGA_SPAN) != 0) {
    perror("Error: munmap() failed");
  }
  close(fd);
  return 0;
}
