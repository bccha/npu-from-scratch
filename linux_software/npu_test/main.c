#define _POSIX_C_SOURCE 199309L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "hw_addresses.h"

#define REG_CTRL 0

#define NPU_MAT_SIZE 8
#define NPU_MAT_BYTES (NPU_MAT_SIZE * 8) // 64 bytes per input/weight matrix
#define NPU_OUT_BYTES                                                          \
  (NPU_MAT_SIZE * 32) // 256 bytes per output matrix (8 rows * 256-bit)

// Global Pointers
volatile uint32_t *npu_ctrl = NULL;
volatile uint32_t *ddr_read_st_csr = NULL;
volatile uint32_t *ddr_write_st_csr = NULL;
volatile uint32_t *ddr_read_st_desc = NULL;
volatile uint32_t *ddr_write_st_desc = NULL;
volatile uint32_t *fpga_ram = NULL;

// ==========================================
// MSGDMA Helper API
// ==========================================

void msgdma_init(volatile uint32_t *csr_base) {
  // Reset Dispatcher (bit 1 of Control Register)
  csr_base[1] = 0x00000002;
  while (csr_base[1] & 0x00000002)
    ;

  // Clear status register (W1C bits)
  csr_base[0] = 0xFFFFFFFF;
  // Clear Stop Dispatcher, Disable Interrupts
  csr_base[1] = 0x00000000;
}

void msgdma_read_stream_push(volatile uint32_t *descriptor_base,
                             uint32_t src_addr, uint32_t length) {
  descriptor_base[0] = src_addr;
  descriptor_base[1] = 0x00000000;
  descriptor_base[2] = length;
  // GO, Gen SOP, Gen EOP -> 0x8C000000 (Bit 31, 27, 26)
  descriptor_base[3] = 0x8C000000;
}

void msgdma_write_stream_push(volatile uint32_t *descriptor_base,
                              uint32_t dst_addr, uint32_t length) {
  descriptor_base[0] = 0x00000000;
  descriptor_base[1] = dst_addr;
  descriptor_base[2] = length;
  // Bit 31: GO
  // Bit 23: End on Length
  // Bit 22: End on EOP
  descriptor_base[3] = 0x80C00000;
}

// ==========================================
// Data Formatting API
// ==========================================

void npu_format_inputs(uint32_t ram_offset, signed char src_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    uint32_t low_32 = 0;
    uint32_t high_32 = 0;

    for (int c = 0; c < 4; c++) {
      low_32 |= (((uint32_t)(unsigned char)src_matrix[r][c]) << (c * 8));
      high_32 |= (((uint32_t)(unsigned char)src_matrix[r][c + 4]) << (c * 8));
    }

    // Write 64-bit word (8 bytes per row) to FPGA RAM space
    fpga_ram[(ram_offset / 4) + (r * 2) + 0] = low_32;
    fpga_ram[(ram_offset / 4) + (r * 2) + 1] = high_32;
  }
}

void npu_format_weights(uint32_t ram_offset, signed char src_matrix[8][8]) {
  for (int t = 0; t < 8; t++) {
    int c = 7 - t;
    uint32_t low_32 = 0;
    uint32_t high_32 = 0;

    for (int r = 0; r < 4; r++) {
      low_32 |= (((uint32_t)(unsigned char)src_matrix[r][c]) << (r * 8));
      high_32 |= (((uint32_t)(unsigned char)src_matrix[r + 4][c]) << (r * 8));
    }

    fpga_ram[(ram_offset / 4) + (t * 2) + 0] = low_32;
    fpga_ram[(ram_offset / 4) + (t * 2) + 1] = high_32;
  }
}

void npu_parse_output(uint32_t ram_offset, uint32_t dst_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      // HPS AXI bridge outputs sequential 32-bit DWORDS,
      // but Big-Endian to Little-Endian byte swap is required for the 32-bit
      // MAC results.
      // Additionally, the 64-bit AXI bus writes the two 32-bit DWORDS in
      // reverse order (Col 1, Col 0).
      int hw_c = c ^ 1;
      uint32_t raw = fpga_ram[(ram_offset / 4) + r * 8 + hw_c];
      dst_matrix[r][c] = __builtin_bswap32(raw);
    }
  }
}

// ==========================================
// NPU Control API
// ==========================================

void npu_load_weights(uint32_t weights_physical_addr, int num_matrices) {
  // NPU를 Load Weight 모드로 설정 (0x3 = seq_mode 1, seq_start 1)
  npu_ctrl[REG_CTRL] = 0x00000003;
  // MSGDMA Read 큐잉 (메모리 -> st_sink)
  msgdma_read_stream_push(ddr_read_st_desc, weights_physical_addr,
                          NPU_MAT_BYTES * num_matrices);
  int timeout = 100000000;
  while (((ddr_read_st_csr[0] & 0x01) != 0) && timeout > 0) {
    timeout--;
  }
  if (timeout == 0)
    printf("[ERROR] npu_load_weights DMA Read Timeout!\n");

  // Wait for the slow 50MHz FPGA internal systolic processing to complete
  // before latching the data! (seq_busy bit is bit 0 in NPU_CTRL offset 1).
  timeout = 100000000;
  while ((npu_ctrl[1] & 0x01) && timeout > 0) {
    timeout--;
  }
  if (timeout == 0)
    printf("[ERROR] npu_load_weights Sequencer Busy Timeout!\n");

  // 64개의 PE에 흩어진 Shadow Weight들을 한 번에 동작 레지스터(Active Weight)로
  // Latch
  npu_ctrl[7] = 1;
  npu_ctrl[7] = 0;
}

void npu_get_matrix(uint32_t dst_phys_addr, int num_matrices) {
  msgdma_write_stream_push(ddr_write_st_desc, dst_phys_addr,
                           NPU_OUT_BYTES * num_matrices);
}

void npu_load_matrix(uint32_t inputs_physical_addr, int num_matrices) {
  // NPU를 Execute 모드로 설정 (0x1 = seq_mode 0, seq_start 1)
  npu_ctrl[REG_CTRL] = 0x00000001;
  // MSGDMA Read 큐잉 (입력 행렬)
  msgdma_read_stream_push(ddr_read_st_desc, inputs_physical_addr,
                          NPU_MAT_BYTES * num_matrices);
}

void npu_wait_execution() {
  int timeout = 100000000;
  while (((ddr_write_st_csr[0] & 0x01) != 0) && timeout > 0) {
    timeout--;
  }
  if (timeout == 0)
    printf("[ERROR] npu_wait_execution DMA Write Timeout!\n");

  // Wait for NPU calculation to be totally idle on the FPGA fabric
  timeout = 100000000;
  while ((npu_ctrl[1] & 0x01) && timeout > 0) {
    timeout--;
  }
  if (timeout == 0)
    printf("[ERROR] npu_wait_execution Sequencer Busy Timeout!\n");
}

// ==========================================
// System Validation / Benchmarking
// ==========================================

#define BATCH_SIZE 100

int32_t hw_weights[8][8];
int32_t sw_inputs[BATCH_SIZE][8][8];
int32_t sw_outputs[BATCH_SIZE][8][8] = {0};

uint64_t get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

void run_benchmark() {
  printf("=========================================\n");
  printf("   ARM Cortex-A9 vs NPU Matrix Benchmark \n");
  printf("=========================================\n\n");
  fflush(stdout);

  // IMPORTANT: For HPS-FPGA, the MSGDMA Master targets the HPS DDR explicitly
  // at 0x20000000 over the f2h_sdram bridge (unlike Nios II which uses a 32-bit
  // Span Extender at 0x08000000).
  uint32_t fpga_physical_base = 0x20000000;
  uint32_t weights_offset = 0x0000;
  uint32_t inputs_offset =
      0x1000; // 100 matrices * 64 bytes = 6400 bytes (0x1900)
  uint32_t dst_offset =
      0x4000; // 100 matrices * 256 bytes = 25600 bytes (0x6400)

  printf("Allocating and Initializing data for %d matrices...\n", BATCH_SIZE);

  // Initialize weights context: identity matrix
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      hw_weights[r][c] = (r == c) ? 1 : 0;
    }
  }

  // Initialize inputs context: random values 1 to 10
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        sw_inputs[b][r][c] =
            (int32_t)((signed char)(((r * 8 + c + b) % 256) - 128));
      }
    }
  }

  // Formatting for NPU hardware representations

  // Clear existing DDR memory for weights to avoid stale data
  for (int i = 0; i < NPU_MAT_BYTES / 4; i++) {
    fpga_ram[(weights_offset / 4) + i] = 0;
  }

  signed char n_weights[8][8] = {0};
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      n_weights[r][c] = (signed char)hw_weights[r][c];
    }
  }
  npu_format_weights(weights_offset, n_weights);

  for (int b = 0; b < BATCH_SIZE; b++) {
    signed char n_inputs[8][8] = {0};
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        n_inputs[r][c] = (signed char)sw_inputs[b][r][c];
      }
    }
    npu_format_inputs(inputs_offset + (b * NPU_MAT_BYTES), n_inputs);
  }

  // ---------------------------------------------------------
  // 1. ARM CPU Benchmark (O3 Optimization)
  // ---------------------------------------------------------
  printf("\n[1] Starting ARM CPU Benchmark (%d matrices)... ", BATCH_SIZE);
  uint64_t cpu_start = get_time_us();

  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        int32_t sum = 0;
        for (int k = 0; k < 8; k++) {
          sum += sw_inputs[b][r][k] * hw_weights[k][c];
        }
        sw_outputs[b][r][c] = sum;
      }
    }
  }

  uint64_t cpu_end = get_time_us();
  uint64_t cpu_time = cpu_end - cpu_start;
  printf("Done in %llu us\n", (unsigned long long)cpu_time);

  // ---------------------------------------------------------
  // 2. Hardware NPU Benchmark
  // ---------------------------------------------------------

  // Aggressive Hard-Reset of Hardware state for consecutive runs:
  // 1. Force NPU Sequencer to IDLE state
  npu_ctrl[REG_CTRL] = 0;
  npu_ctrl[REG_CTRL] = 0; // Double write to ensure pipe flush

  // 2. Clear out any trapped bytes in the Avalon-ST FIFOs and MSGDMA queues
  msgdma_init(ddr_read_st_csr);
  msgdma_init(ddr_write_st_csr);

  int idle_timeout = 100000000;
  while ((npu_ctrl[1] & 0x01) && idle_timeout > 0) {
    idle_timeout--;
  }

  // Clear existing DDR memory for outputs to avoid stale data
  for (int i = 0; i < (BATCH_SIZE * NPU_OUT_BYTES) / 4; i++) {
    fpga_ram[(dst_offset / 4) + i] = 0;
  }

  // Pre-load weights (not included in strict batch execution time)
  npu_load_weights(fpga_physical_base + weights_offset, 1);

  printf("[2] Starting FPGA NPU Streaming Benchmark (%d matrices)... ",
         BATCH_SIZE);

  uint64_t npu_start = get_time_us();

  // Configure total sequence rows for Avalon-ST EOP generation
  // (Must be set AFTER loading weights, right before inputs begin).
  npu_ctrl[6] = BATCH_SIZE * 8; // REG_SEQ_ROWS = index 6

  // Dispatch MSGDMA Descriptor pairs for 100 matrices simultaneously
  npu_get_matrix(fpga_physical_base + dst_offset, BATCH_SIZE);
  npu_load_matrix(fpga_physical_base + inputs_offset, BATCH_SIZE);

  npu_wait_execution();

  uint64_t npu_end = get_time_us();
  uint64_t npu_time = npu_end - npu_start;
  printf("Done in %llu us\n\n", (unsigned long long)npu_time);

  // ---------------------------------------------------------
  // Verify Correctness
  // ---------------------------------------------------------
  printf("Starting Verification...\n");
  int errors = 0;
  for (int b = 0; b < BATCH_SIZE; b++) {
    uint32_t hw_matrix_out[8][8];
    npu_parse_output(dst_offset + (b * NPU_OUT_BYTES), hw_matrix_out);
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        uint32_t hw_val = hw_matrix_out[r][c];
        int32_t expected_signed = sw_outputs[b][r][c];
        uint32_t expected = (uint32_t)expected_signed;

        if (hw_val != expected) {
          if (errors == 0) { // Print the full matrices on first error
            printf("\n--- FIRST MISMATCH IN BATCH %d ---\n", b);
            printf("Expected ARM Output:\n");
            for (int pr = 0; pr < 8; pr++) {
              for (int pc = 0; pc < 8; pc++) {
                printf("%08x ", (uint32_t)sw_outputs[b][pr][pc]);
              }
              printf("\n");
            }
            printf("\nHardware NPU Output:\n");
            for (int pr = 0; pr < 8; pr++) {
              for (int pc = 0; pc < 8; pc++) {
                printf("%08x ", hw_matrix_out[pr][pc]);
              }
              printf("\n");
            }
            printf("------------------------------------\n\n");
          }
          if (errors < 5) { // Limit error spam
            printf("Batch %d Mismatch [%d, %d]: HW=0x%08x, Exp=0x%08x\n", b, r,
                   c, hw_val, expected);
          }
          errors++;
        }
      }
    }
  }

  if (errors) {
    printf("Verification Failed with %d mismatches.\n", errors);
  } else {
    printf("Verification Passed! Hardware outputs match ARM CPU.\n");
    // Show Speedup
    double speedup = (double)cpu_time / (double)npu_time;
    printf("\n>>> NPU Speedup: %.2fx Faster than ARM Cortex-A9! <<<\n",
           speedup);
  }
}

int main(int argc, char **argv) {
  int fd;
  void *virtual_base;
  void *hps_ram_base;

  if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1) {
    perror("Error: could not open /dev/mem");
    return 1;
  }

  // Map LWHPS2FPGA Bridge (contains NPU Ctrl and MSGDMA CSRs)
  virtual_base = mmap(NULL, LWHPS2FPGA_SPAN, (PROT_READ | PROT_WRITE),
                      MAP_SHARED, fd, LWHPS2FPGA_BASE);
  if (virtual_base == MAP_FAILED) {
    perror("Error: mmap() failed for LWHPS2FPGA");
    close(fd);
    return 1;
  }

  // Map HPS2FPGA Bridge (contains RAM used by NPU - 0x20000000 HPS DDR3)
  hps_ram_base = mmap(NULL, HPS_FPGA_RAM_SPAN, (PROT_READ | PROT_WRITE),
                      MAP_SHARED, fd, HPS_FPGA_RAM_BASE);
  if (hps_ram_base == MAP_FAILED) {
    perror("Error: mmap() failed for HPS FPGA RAM");
    munmap(virtual_base, LWHPS2FPGA_SPAN);
    close(fd);
    return 1;
  }

  // virtual_base is already mapped to LWHPS2FPGA_BASE (0xFF200000)
  // Therefore, we only add the component offset.
  npu_ctrl = (uint32_t *)((uint8_t *)virtual_base + NPU_CTRL_OFFSET);
  ddr_read_st_csr =
      (uint32_t *)((uint8_t *)virtual_base + DDR_READ_ST_CSR_OFFSET);
  ddr_write_st_csr =
      (uint32_t *)((uint8_t *)virtual_base + DDR_WRITE_ST_CSR_OFFSET);
  ddr_read_st_desc =
      (uint32_t *)((uint8_t *)virtual_base + DDR_READ_ST_DESC_OFFSET);
  ddr_write_st_desc =
      (uint32_t *)((uint8_t *)virtual_base + DDR_WRITE_ST_DESC_OFFSET);

  fpga_ram = (uint32_t *)((uint8_t *)hps_ram_base);

  printf("[DEBUG] Init ddr_read_st_csr... Done\n");
  printf("[DEBUG] Init ddr_write_st_csr... Done\n");
  fflush(stdout);

  run_benchmark();

  if (munmap(virtual_base, LWHPS2FPGA_SPAN) != 0) {
    perror("Error: munmap() failed for LWHPS2FPGA");
  }
  if (munmap(hps_ram_base, HPS_FPGA_RAM_SPAN) != 0) {
    perror("Error: munmap() failed");
  }

  close(fd);
  return 0;
}
