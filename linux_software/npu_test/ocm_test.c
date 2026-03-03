#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define LWHPS2FPGA_BASE 0xFF200000
#define LWHPS2FPGA_SPAN 0x00200000

#define HPS2FPGA_AXI_BASE 0xC0000000
#define HPS2FPGA_AXI_SPAN 0x20000000

#define OCM_OFFSET_LW 0x00040000

#define IOWR_32DIRECT(base, offset, data)                                      \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)                                            \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))

void test_memory_access(const char *name, volatile uint8_t *base_addr,
                        uint32_t offset) {
  printf("Testing %s (Offset: 0x%08x)...\n", name, offset);

  // Write test pattern
  uint32_t test_val1 = 0xDEADBEEF;
  uint32_t test_val2 = 0x12345678;

  printf("  Writing 0x%08x to offset 0x00...\n", test_val1);
  IOWR_32DIRECT(base_addr, offset + 0x00, test_val1);

  printf("  Writing 0x%08x to offset 0x04...\n", test_val2);
  IOWR_32DIRECT(base_addr, offset + 0x04, test_val2);

  // Read back and verify
  uint32_t read_val1 = IORD_32DIRECT(base_addr, offset + 0x00);
  uint32_t read_val2 = IORD_32DIRECT(base_addr, offset + 0x04);

  printf("  Read back from offset 0x00: 0x%08x\n", read_val1);
  printf("  Read back from offset 0x04: 0x%08x\n", read_val2);

  if (read_val1 == test_val1 && read_val2 == test_val2) {
    printf("  [PASS] %s Access Successful!\n\n", name);
  } else {
    printf("  [FAIL] %s Verification Failed!\n\n", name);
  }
}

int main() {
  printf("=== ARM to OCM Access Test ===\n\n");

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open /dev/mem");
    return EXIT_FAILURE;
  }

  // 1. Map Lightweight Bridge
  uint8_t *lw_bridge_map =
      (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, LWHPS2FPGA_BASE);
  if (lw_bridge_map == MAP_FAILED) {
    perror("mmap lw_bridge_map failed");
    close(fd);
    return EXIT_FAILURE;
  }

  // 2. Map Heavyweight Bridge (AXI H2F) 0xC0000000
  uint8_t *h2f_bridge_map =
      (uint8_t *)mmap(NULL, HPS2FPGA_AXI_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, HPS2FPGA_AXI_BASE);
  if (h2f_bridge_map == MAP_FAILED) {
    perror("mmap h2f_bridge_map failed (Checking LW only)");
  }

  // --- Test 1: LWHPS2FPGA ---
  printf(">> Test Case 1: Lightweight HPS-to-FPGA Bridge\n");
  test_memory_access("OCM over LW Bridge", lw_bridge_map, OCM_OFFSET_LW);

  // --- Test 2: H2F AXI ---
  if (h2f_bridge_map != MAP_FAILED) {
    printf(">> Test Case 2: Heavyweight HPS-to-FPGA AXI Bridge\n");
    // OCM offset on H2F AXI might be different depending on Qsys mapping.
    // If it's mapped at address 0x00000000 on the AXI bridge:
    test_memory_access("OCM over AXI Bridge", h2f_bridge_map, 0x00000000);
    munmap(h2f_bridge_map, HPS2FPGA_AXI_SPAN);
  }

  // Cleanup
  munmap(lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);

  printf("Tests complete.\n");
  return 0;
}
