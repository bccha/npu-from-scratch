#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// ==========================================
// Bridge Offsets (Fixed Architecture)
// ==========================================
#define LWHPS2FPGA_BASE 0xFF200000
#define LWHPS2FPGA_SPAN 0x00200000

#define HPS_FPGA_RAM_BASE 0x20000000
#define HPS_FPGA_RAM_SPAN 0x01000000 // 16MB Window

// ==========================================
// Component Offsets (Extracted from Qsys)
// ==========================================
#define DDR_READ_ST_CSR_OFFSET 0x31000
#define DDR_READ_ST_DESC_OFFSET 0x31040
#define DDR_WRITE_ST_CSR_OFFSET 0x31020
#define DDR_WRITE_ST_DESC_OFFSET 0x31050
#define NPU_CTRL_OFFSET 0x00030000

// ============================================================================
// Hardware Access Macros
// ============================================================================
#define IOWR_32DIRECT(base, offset, data)                                      \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)                                            \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))

#define IOWR(base, reg, data)                                                  \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)) = (data))
#define IORD(base, reg)                                                        \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)))

// ============================================================================
// Global Pointers & Base Addresses
// ============================================================================
volatile uint8_t *lw_bridge_map = NULL;
volatile uint8_t *ddr_map = NULL;

volatile uint8_t *NPU_CTRL_BASE;
volatile uint8_t *DDR_READ_ST_CSR_BASE;
volatile uint8_t *DDR_READ_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR_WRITE_ST_CSR_BASE;
volatile uint8_t *DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR3_WINDOW_BASE;

// ============================================================================
// MSGDMA Helpers
// ============================================================================
void msgdma_init(volatile uint8_t *csr_base) {
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);
  IOWR_32DIRECT(csr_base, 0x04, 0x00000000);
}

void msgdma_read_stream_push(volatile uint8_t *descriptor_base,
                             uint32_t src_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x8C000000);
}

void msgdma_write_stream_push(volatile uint8_t *descriptor_base,
                              uint32_t dst_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80C00000);
}

// ============================================================================
// Terminal Helper
// ============================================================================
char get_char_polled() {
  char ch;
  // Note: space before %c ignores whitespace/newlines left in standard input
  if (scanf(" %c", &ch) != 1) {
    return 'q'; // Default to quit if EOF or error reading
  }
  return ch;
}

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

#define NPU_MAT_SIZE 8
#define NPU_MAT_BYTES (NPU_MAT_SIZE * 8)
#define NPU_OUT_BYTES (NPU_MAT_SIZE * 32)

// ==========================================
// Data Formatting API
// ==========================================

void npu_format_inputs(volatile uint8_t *dst_addr,
                       signed char src_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    uint32_t low_32 = 0;
    uint32_t high_32 = 0;

    for (int c = 0; c < 4; c++) {
      low_32 |= (((uint32_t)(unsigned char)src_matrix[r][c]) << (c * 8));
      high_32 |= (((uint32_t)(unsigned char)src_matrix[r][c + 4]) << (c * 8));
    }

    IOWR_32DIRECT(dst_addr, r * 8 + 0, low_32);
    IOWR_32DIRECT(dst_addr, r * 8 + 4, high_32);
  }
}

void npu_format_weights(volatile uint8_t *dst_addr,
                        signed char src_matrix[8][8]) {
  for (int t = 0; t < 8; t++) {
    int c = 7 - t;
    uint32_t low_32 = 0;
    uint32_t high_32 = 0;

    for (int r = 0; r < 4; r++) {
      low_32 |= (((uint32_t)(unsigned char)src_matrix[r][c]) << (r * 8));
      high_32 |= (((uint32_t)(unsigned char)src_matrix[r + 4][c]) << (r * 8));
    }

    IOWR_32DIRECT(dst_addr, t * 8 + 0, low_32);
    IOWR_32DIRECT(dst_addr, t * 8 + 4, high_32);
  }
}

void npu_parse_output(volatile uint8_t *src_addr, uint32_t dst_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      int hw_c = c ^ 1;
      uint32_t raw = IORD_32DIRECT(src_addr, (r * 8 + hw_c) * 4);
      dst_matrix[r][c] = __builtin_bswap32(raw);
    }
  }
}

// ==========================================
// NPU Control API
// ==========================================

void npu_load_weights(uint32_t weights_addr, int num_matrices) {
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000003);

  msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, weights_addr,
                          NPU_MAT_BYTES * num_matrices);

  // Wait for MSGDMA Read Status to be Idle (Bit 0 != 1)
  while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0) {
  }

  // Wait for NPU Sequencer Busy Flag (REG_STATUS Bit 0) to be idle.
  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
  }

  IOWR(NPU_CTRL_BASE, 7, 1);
  IOWR(NPU_CTRL_BASE, 7, 0);
}

void npu_get_matrix(uint32_t dst_addr, int num_matrices) {
  msgdma_write_stream_push(DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE, dst_addr,
                           NPU_OUT_BYTES * num_matrices);
}

void npu_load_matrix(uint32_t inputs_addr, int num_matrices) {
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000001);
  msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, inputs_addr,
                          NPU_MAT_BYTES * num_matrices);
}

void npu_wait_execution() {
  // Wait for MSGDMA Write Status to be Idle
  while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0) {
  }

  // Wait for NPU Sequencer Busy Flag
  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
  }
}

// ==========================================
// System Validation
// ==========================================

void verify_full_system() {
  printf("\nStarting Full System Matrix Validation (Fixed 8x8 HW with 4x4 "
         "submatrix)...\n");

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  IOWR(NPU_CTRL_BASE, REG_CTRL, 0);
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0);

  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
  }

  uint32_t physical_base = 0x20000000;
  volatile uint8_t *weights_addr = DDR3_WINDOW_BASE;
  volatile uint8_t *inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  volatile uint8_t *dst_addr = DDR3_WINDOW_BASE + 0x2000;

  printf("Clearing Memories...\n");
  for (int i = 0; i < 64; i++) {
    IOWR_32DIRECT(weights_addr, i * 4, 0);
    IOWR_32DIRECT(inputs_addr, i * 4, 0);
    IOWR_32DIRECT(dst_addr, i * 4, 0);
  }

  printf("Preparing 8x8 Identity Weight Matrix...\n");

  signed char test_weights[8][8] = {0};
  signed char test_inputs[8][8] = {0};

  for (int i = 0; i < 8; i++) {
    test_weights[i][i] = 1;
  }

  int val = 1;
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      test_inputs[r][c] = val++;
    }
  }

  npu_format_weights(weights_addr, test_weights);
  npu_format_inputs(inputs_addr, test_inputs);

  printf("Phase 1: Loading Weights via MSGDMA API...\n");
  npu_load_weights(physical_base, 1);
  printf("Weights Loaded!\n");

  printf("Phase 2: Execution via MSGDMA API...\n");
  npu_get_matrix(physical_base + 0x2000, 1);
  npu_load_matrix(physical_base + 0x1000, 1);

  npu_wait_execution();
  printf("Execution Finished!\n\n");

  int errors = 0;

  printf("Verifying Output (Expecting Y=X for 8x8 matrix)...\n");

  uint32_t hw_matrix[8][8];
  npu_parse_output(dst_addr, hw_matrix);

  printf("\n=== Hardware Output Matrix ===\n");
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      printf("%3d ", (int)hw_matrix[r][c]);
    }
    printf("\n");
  }

  printf("\n=== Expected Output Matrix ===\n");
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      printf("%3d ", (int)(r * 8 + c + 1));
    }
    printf("\n");
  }
  printf("\n");

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      uint32_t hw_val = hw_matrix[r][c];
      uint32_t np_val = r * 8 + c + 1;

      if (hw_val != np_val) {
        printf("Mismatch at [%d, %d]: HW=0x%08x, Expected=0x%08x\n", r, c,
               (int)hw_val, (int)np_val);
        errors++;
      }
    }
  }

  if (errors == 0) {
    printf(
        "\nFull System Validation: PASS! All 64 elements matched correctly.\n");
  } else {
    printf("\nFull System Validation: FAIL (%d errors)\n", errors);
  }
}

void verify_streaming_batch() {
  printf("\nStarting Streaming Batch Test (10 Matrices)....\n");

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  uint32_t physical_base = 0x20000000;
  volatile uint8_t *weights_addr = DDR3_WINDOW_BASE;
  volatile uint8_t *inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  volatile uint8_t *outputs_addr = DDR3_WINDOW_BASE + 0x8000;

  signed char weight_matrix[8][8];
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      weight_matrix[r][c] = (r == c) ? 1 : 0;
    }
  }
  npu_format_weights(weights_addr, weight_matrix);

  for (int i = 0; i < 10; i++) {
    signed char in_mat[8][8];
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        in_mat[r][c] = (signed char)(((i * 10 + r * 8 + c) % 256) - 128);
      }
    }
    npu_format_inputs(inputs_addr + i * NPU_MAT_BYTES, in_mat);
  }

  printf("Clearing Memories...\n");
  for (int i = 0; i < (10 * NPU_OUT_BYTES) / 4; i++) {
    IOWR_32DIRECT(outputs_addr, i * 4, 0);
  }

  printf("Loading Weights...\n");
  npu_load_weights(physical_base, 1);

  printf("Firing 10-Batch Streaming Pipeline...\n");

  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 10 * 8);

  npu_get_matrix(physical_base + 0x8000, 10);
  npu_load_matrix(physical_base + 0x1000, 10);

  npu_wait_execution();

  int total_errors = 0;
  for (int i = 0; i < 10; i++) {
    uint32_t hw_matrix[8][8];
    npu_parse_output(outputs_addr + i * NPU_OUT_BYTES, hw_matrix);

    int errors = 0;
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        uint32_t hw_val = hw_matrix[r][c];

        int expected_signed =
            (int)((signed char)(((i * 10 + r * 8 + c) % 256) - 128));
        uint32_t expected = (uint32_t)expected_signed;

        if (hw_val != expected) {
          if (errors < 5) {
            printf("Batch %d Mismatch [%d, %d]: HW=0x%08x, Exp=0x%08x\n", i, r,
                   c, (unsigned int)hw_val, (unsigned int)expected);
          }
          errors++;
        }
      }
    }
    if (errors == 0) {
      printf("Batch %d: PASS\n", i);
    } else {
      printf("Batch %d: FAIL (%d errors)\n", i, errors);
      total_errors += errors;
    }
  }

  if (total_errors == 0) {
    printf("\nStreaming Validation: PASS! All 10 batches successfully fully "
           "matched.\n");
  } else {
    printf("\nStreaming Validation: FAIL (%d total errors)\n", total_errors);
  }
}

void verify_mac_pe() {
  printf("\nStarting MAC PE Verification...\n");

  IOWR(NPU_CTRL_BASE, REG_PE_X_IN, 7);
  usleep(1);
  // NEW RTL needs valid=1 AND load_weight=1. 11(binary) = 3
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 3);
  usleep(1);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);
  usleep(1);

  // CRITICAL FIX: The PE is double-buffered.
  // We must trigger `weight_latch_en` (NPU_CTRL offset 7) to move the weight
  // from `shadow_weight_reg` to `active_weight_reg`.
  // Otherwise, active_weight_reg remains 0, and the PE calculates 0 * 3 + 10
  // = 10.
  IOWR(NPU_CTRL_BASE, 7, 1);
  usleep(1);
  IOWR(NPU_CTRL_BASE, 7, 0);
  usleep(1);

  IOWR(NPU_CTRL_BASE, REG_PE_X_IN, 3);
  usleep(1);
  IOWR(NPU_CTRL_BASE, REG_PE_Y_IN, 10);
  usleep(1);

  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 2);
  usleep(1);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);

  // MAC PE는 npu_core 내부에서만 동작하며 상태 레지스터가 연결되어 있지 않은
  // 모듈 테스트입니다. ARM 프로세서 속도가 매우 빨라서, NPU_CTRL_BASE에 값이
  // 쓰인 후 FPGA가 연산을 마치기도 전에 바로 Y_OUT을 가져오려고 하면 이전
  // 상태값(10)을 반환하게 됩니다. 따라서 Status 레지스터 폴링이 적용되지 않는
  // 이 테스트에서는 usleep으로 물리적 시간을 보장해야 합니다.
  usleep(1); // 1us (50 Cycles at 50MHz)

  uint32_t result = IORD(NPU_CTRL_BASE, REG_PE_Y_OUT);
  printf("Result: %d (Expected: 31)\n", (int)result);

  if (result == 31)
    printf("MAC PE Test: PASS\n");
  else
    printf("MAC PE Test: FAIL\n");
}

// ==========================================
// Performance Comparison (CPU vs NPU)
// ==========================================

// CPU Reference 8x8 Matrix Multiplication
void cpu_matmul_8x8(signed char A[8][8], signed char B[8][8], int32_t C[8][8]) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      int32_t sum = 0;
      for (int k = 0; k < 8; k++) {
        sum += (int32_t)A[i][k] * (int32_t)B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

// Timing helper
double get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

void verify_performance_cpu_vs_npu(int batch_count) {
  if (batch_count <= 0) {
    printf("Invalid batch count. Must be > 0.\n");
    return;
  }

  printf(
      "\nStarting CPU vs NPU Performance Comparison (%d Batches of 8x8)...\n",
      batch_count);

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  uint32_t physical_base = 0x20000000;
  volatile uint8_t *weights_addr = DDR3_WINDOW_BASE;
  volatile uint8_t *inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  volatile uint8_t *outputs_addr = DDR3_WINDOW_BASE + 0x8000;

  // 1. Generate Random Data
  signed char weight_matrix[8][8];

  // Dynamically allocate to avoid stack overflow on large batch counts
  signed char (*input_matrices)[8][8] =
      malloc(batch_count * sizeof(*input_matrices));
  int32_t (*cpu_output)[8][8] = malloc(batch_count * sizeof(*cpu_output));

  if (!input_matrices || !cpu_output) {
    printf("Failed to allocate memory for matrices.\n");
    if (input_matrices)
      free(input_matrices);
    if (cpu_output)
      free(cpu_output);
    return;
  }

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      weight_matrix[r][c] = (rand() % 256) - 128;
    }
  }

  for (int b = 0; b < batch_count; b++) {
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        input_matrices[b][r][c] = (rand() % 256) - 128;
      }
    }
  }

  // Formatting and loading to DDR for NPU
  npu_format_weights(weights_addr, weight_matrix);
  for (int b = 0; b < batch_count; b++) {
    npu_format_inputs(inputs_addr + b * NPU_MAT_BYTES, input_matrices[b]);
  }

  // 2. Profile CPU Execution Time
  double cpu_start = get_time_us();
  for (int b = 0; b < batch_count; b++) {
    // initialize cpu output to 0 since malloc doesn't zero memory
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        cpu_output[b][r][c] = 0;
      }
    }
    cpu_matmul_8x8(input_matrices[b], weight_matrix, cpu_output[b]);
  }
  double cpu_end = get_time_us();
  double cpu_duration = cpu_end - cpu_start;

  // 3. Profile NPU Execution Time (Include DMA Setup overhead)
  npu_load_weights(physical_base, 1); // pre-load once

  double npu_start = get_time_us();
  // Batch Execution Request
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, batch_count * 8);
  npu_get_matrix(physical_base + 0x8000, batch_count);
  npu_load_matrix(physical_base + 0x1000, batch_count);

  // Wait for completion
  npu_wait_execution();
  double npu_end = get_time_us();
  double npu_duration = npu_end - npu_start;

  // 4. Verify Correctness
  int total_errors = 0;
  for (int b = 0; b < batch_count; b++) {
    uint32_t hw_matrix[8][8];
    npu_parse_output(outputs_addr + b * NPU_OUT_BYTES, hw_matrix);

    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        uint32_t hw_val = hw_matrix[r][c];
        uint32_t cpu_val = (uint32_t)cpu_output[b][r][c];
        if (hw_val != cpu_val) {
          if (total_errors < 5) {
            printf("Mismatch Batch %d [%d,%d] - NPU: %08x, CPU: %08x\n", b, r,
                   c, (int)hw_val, (int)cpu_val);
          }
          total_errors++;
        }
      }
    }
  }

  // 5. Report Results
  printf("\n=== Performance Results (%d Batches) ===\n", batch_count);
  if (total_errors == 0) {
    printf("Verification: PASS (NPU output perfectly matches CPU)\n");
  } else {
    printf("Verification: FAIL (%d errors detected)\n", total_errors);
  }

  printf("CPU Time : %.3f us\n", cpu_duration);
  printf("NPU Time : %.3f us (Includes DMA Setup overhead)\n", npu_duration);

  if (npu_duration > 0) {
    printf("Speedup  : %.2f x\n", cpu_duration / npu_duration);
  }

  free(input_matrices);
  free(cpu_output);
}

// ----------------------------------------------------------------------------
// mmap Entry Point
// ----------------------------------------------------------------------------
int main() {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open /dev/mem");
    return EXIT_FAILURE;
  }

  lw_bridge_map = (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, LWHPS2FPGA_BASE);
  if (lw_bridge_map == MAP_FAILED) {
    perror("mmap lw_bridge_map");
    close(fd);
    return EXIT_FAILURE;
  }

  ddr_map = (uint8_t *)mmap(NULL, HPS_FPGA_RAM_SPAN, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, HPS_FPGA_RAM_BASE);
  if (ddr_map == MAP_FAILED) {
    perror("mmap ddr_map");
    munmap((void *)lw_bridge_map, LWHPS2FPGA_SPAN);
    close(fd);
    return EXIT_FAILURE;
  }

  // Base addresses setup mapping
  NPU_CTRL_BASE = lw_bridge_map + NPU_CTRL_OFFSET;
  DDR_READ_ST_CSR_BASE = lw_bridge_map + DDR_READ_ST_CSR_OFFSET;
  DDR_READ_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_READ_ST_DESC_OFFSET;
  DDR_WRITE_ST_CSR_BASE = lw_bridge_map + DDR_WRITE_ST_CSR_OFFSET;
  DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_WRITE_ST_DESC_OFFSET;
  DDR3_WINDOW_BASE = ddr_map;

  while (1) {
    printf("\nNPU System Verification (Full Framework)\n");
    printf("----------------------------------------------\n");
    printf("1. Verify MAC PE\n");
    printf("2. Verify Full System Data path\n");
    printf("3. Verify Streaming Pipeline (N Batches)\n");
    printf("4. CPU vs NPU Performance Comparison\n");
    printf("q. Quit\n");
    printf("Choose: ");

    char c = get_char_polled();
    printf("%c\n", c); // Echo

    if (c == '1') {
      verify_mac_pe();
    } else if (c == '2') {
      verify_full_system();
    } else if (c == '3') {
      verify_streaming_batch();
    } else if (c == '4') {
      int batches = 0;
      printf("Enter number of batches (e.g., 10, 100, 1000): ");
      if (scanf("%d", &batches) == 1) {
        verify_performance_cpu_vs_npu(batches);
      } else {
        printf("Invalid input.\n");
        // Clear invalid input from standard input buffer
        while (getchar() != '\n')
          ;
      }
    } else if (c == 'q') {
      printf("Exiting...\n");
      break;
    }
  }

  munmap((void *)ddr_map, HPS_FPGA_RAM_SPAN);
  munmap((void *)lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);

  return 0;
}