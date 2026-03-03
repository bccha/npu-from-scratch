#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LWHPS2FPGA_BASE 0xFF200000
#define LWHPS2FPGA_SPAN 0x00200000

#define NPU_CTRL_OFFSET 0x00030000
#define OCM_OFFSET_LW 0x00040000

// The address the PP (FPGA master) uses to access OCM
#define OCM_FPGA_ADDR 0x00040000

#define IOWR_32DIRECT(base, offset, data)                                      \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)                                            \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))

#define IOWR(base, reg, data)                                                  \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)) = (data))
#define IORD(base, reg)                                                        \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)))

void npu_post_process(volatile uint8_t *npu_ctrl_base, uint32_t src_addr,
                      uint32_t dst_addr, uint32_t bias_addr,
                      uint32_t num_elements, uint32_t shift_val) {
  // select_pp corresponds to address[7:4] == 4'h1, so base offset is 0x10.
  // The registers are 32-bit (4 bytes), so the byte offsets are:
  // 4'h0 -> 0x10 * 4 = 16 * 4 = 64
  // 4'h1 -> 0x11 * 4 = 17 * 4 = 68
  // 4'h2 -> 0x12 * 4 = 18 * 4 = 72
  // 4'h3 -> 0x13 * 4 = 19 * 4 = 76
  // 4'h4 -> 0x14 * 4 = 20 * 4 = 80
  // 4'h5 -> 0x15 * 4 = 21 * 4 = 84
  // 4'h6 -> 0x16 * 4 = 22 * 4 = 88

  // We are using IOWR which multiplies the register index by 4,
  // so we can pass the register index directly:
  IOWR(npu_ctrl_base, 18, src_addr);
  IOWR(npu_ctrl_base, 19, dst_addr);
  IOWR(npu_ctrl_base, 20, bias_addr);
  IOWR(npu_ctrl_base, 21, num_elements);
  IOWR(npu_ctrl_base, 22, shift_val);

  // Diagnostic Readback
  printf("  [NPU_CTRL Check] Reading back PP configuration:\n");
  printf("    src_addr (reg 18): 0x%08X (Expected: 0x%08X)\n",
         IORD(npu_ctrl_base, 18), src_addr);
  printf("    dst_addr (reg 19): 0x%08X (Expected: 0x%08X)\n",
         IORD(npu_ctrl_base, 19), dst_addr);
  printf("    bias_addr(reg 20): 0x%08X (Expected: 0x%08X)\n",
         IORD(npu_ctrl_base, 20), bias_addr);
  printf("    num_elems(reg 21): %d (Expected: %d)\n", IORD(npu_ctrl_base, 21),
         num_elements);
  printf("    shift_val(reg 22): %d (Expected: %d)\n", IORD(npu_ctrl_base, 22),
         shift_val);

  // Start (Offset 0x10 = 16)
  IOWR(npu_ctrl_base, 16, 1);

  // Wait for Done (Offset 0x11 = 17, bit 0 is pp_done)
  int timeout = 0;
  while ((IORD(npu_ctrl_base, 17) & 0x01) == 0) {
    timeout++;
    if (timeout > 1000000) {
      printf("\n[TIMEOUT] PP is stuck! STATUS=0x%08X\n",
             IORD(npu_ctrl_base, 17));
      break;
    }
  }
}

int main() {
  printf("=== NPU Post Processor (PP) Test ===\n\n");

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open /dev/mem");
    return EXIT_FAILURE;
  }

  // Map Lightweight Bridge
  uint8_t *lw_bridge_map =
      (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, LWHPS2FPGA_BASE);
  if (lw_bridge_map == MAP_FAILED) {
    perror("mmap lw_bridge_map failed");
    close(fd);
    return EXIT_FAILURE;
  }

  volatile uint8_t *npu_ctrl_base = lw_bridge_map + NPU_CTRL_OFFSET;
  volatile uint8_t *ocm_base = lw_bridge_map + OCM_OFFSET_LW;

  // Emulate data for PP:
  // Let's test 16 elements.
  const uint32_t NUM_ELEMENTS = 16;
  const uint32_t SHIFT_VAL = 8;

  // Choose offsets within OCM
  uint32_t z_offset = 0x0000;
  uint32_t bias_offset = 0x0100;
  uint32_t dst_offset = 0x0200;

  // Set up source Z data and Biases in OCM via CPU
  printf("1. Initializing Source (Z) and Bias arrays in OCM...\n");
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    int32_t z_val =
        (i - 8) * 1000; // Generate some negative and positive values
    int32_t bias_val = 500;

    // Each element is 32-bit (4 bytes).
    IOWR_32DIRECT(ocm_base, z_offset + i * 4, (uint32_t)z_val);
    IOWR_32DIRECT(ocm_base, bias_offset + i * 4, (uint32_t)bias_val);
  }

  // Clear Destination area (16 elements = 16 bytes = 4 words)
  printf("2. Clearing Destination area in OCM...\n");
  for (int i = 0; i < (NUM_ELEMENTS / 4) + 1; i++) {
    IOWR_32DIRECT(ocm_base, dst_offset + i * 4, 0x00000000);
  }

  // Trigger Hardware PP
  printf("3. Triggering Hardware Post Processor...\n");
  npu_post_process(npu_ctrl_base, OCM_FPGA_ADDR + z_offset,
                   OCM_FPGA_ADDR + dst_offset, OCM_FPGA_ADDR + bias_offset,
                   NUM_ELEMENTS, SHIFT_VAL);
  printf("4. Post Processor Execution Done!\n\n");

  // Read back and verify
  printf("=== Verification ===\n");
  int errors = 0;

  // PP writes packed 8-bit output, 4 elements per 32-bit word
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    // Read the expected values we wrote
    // i * 4 bytes per element
    int32_t z_val = (int32_t)IORD_32DIRECT(ocm_base, z_offset + i * 4);
    int32_t bias_val = (int32_t)IORD_32DIRECT(ocm_base, bias_offset + i * 4);

    // CPU Expected PP logic
    int32_t raw_sum = z_val + bias_val;
    int32_t shifted = raw_sum >> SHIFT_VAL;
    int32_t expected = shifted;
    if (expected < 0)
      expected = 0;
    else if (expected > 127)
      expected = 127;

    // Read hardware output (PP writes 4 bytes per word, byte-addressable)
    uint32_t word_idx = i / 4;
    uint32_t byte_idx = i % 4; // Little-endian, byte 0 is lowest 8 bits.
    uint32_t hw_word = IORD_32DIRECT(ocm_base, dst_offset + word_idx * 4);
    uint8_t hw_val = (hw_word >> (byte_idx * 8)) & 0xFF;

    printf("  Element %2d | Z: %6d, Bias: %4d | Expected: %3d, HW: %3d ", i,
           z_val, bias_val, expected, hw_val);

    if ((uint8_t)expected == hw_val) {
      printf("[PASS]\n");
    } else {
      printf("[FAIL]\n");
      errors++;
    }
  }

  if (errors == 0) {
    printf("\n>>> RESULT: ALL PASSED! <<<\n");
  } else {
    printf("\n>>> RESULT: %d ERRORS DETECTED! <<<\n", errors);
  }

  // Cleanup
  munmap(lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);

  return 0;
}
