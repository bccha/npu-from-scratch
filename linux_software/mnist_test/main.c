#include "hw_addresses.h"
#include "model_params.h"
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define IOWR_32DIRECT(base, offset, data)                                      \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)                                            \
  (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))

#define IOWR(base, reg, data)                                                  \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)) = (data))
#define IORD(base, reg)                                                        \
  (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)))



#define REG_CTRL 0
#define REG_STATUS 1
#define REG_SEQ_ROWS 6

#define NPU_MAT_SIZE 8
#define NPU_MAT_BYTES (NPU_MAT_SIZE * 8)
#define NPU_OUT_BYTES 64

volatile uint8_t *NPU_CTRL_BASE;
volatile uint8_t *DDR_READ_ST_CSR_BASE;
volatile uint8_t *DDR_READ_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR_WRITE_ST_CSR_BASE;
volatile uint8_t *DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *NPU_DDR_CNTL_BASE;
volatile uint8_t *lw_bridge_map;

void msgdma_init(volatile uint8_t *csr_base) {
  IOWR_32DIRECT(csr_base, 0x04, (1 << 1));
  while (IORD_32DIRECT(csr_base, 0x00) & (1 << 6))
    ;
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);
  IOWR_32DIRECT(csr_base, 0x04, 0x00000000);
}

void msgdma_read_stream_push(volatile uint8_t *descriptor_base,
                             uint32_t src_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000300);
}

void msgdma_write_stream_push(volatile uint8_t *descriptor_base,
                              uint32_t dst_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000000);
}

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

int load_binary_file(const char *filename, uint8_t *dest, size_t max_size) {
  FILE *f = fopen(filename, "rb");
  if (!f)
    return 0;
  size_t bytes = fread(dest, 1, max_size, f);
  fclose(f);
  return bytes;
}

void npu_format_inputs(volatile uint8_t *dst_addr, signed char src_matrix[8][8]) {
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

void npu_format_weights(volatile uint8_t *dst_addr, signed char src_matrix[8][8]) {
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

double get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

// Memory mapping configurations
uint32_t phys_ddr_base;
uint8_t *virt_ddr_base;
// OCM is rigorously locked to the Lightweight Bridge at offset 0x40000
uint32_t phys_ocm_base;
volatile uint8_t *virt_ocm_base;
uint32_t npu_ddr_base = 0x20000000;
uint32_t dma_ocm_base = 0x00040000;

// Base Offsets within DDR
uint32_t inputs_offset = 0x000000;
uint32_t weights_l1_offset = 0x800000;
uint32_t weights_l2_offset = 0x840000;
uint32_t npu_out_ddr_offset = 0x900000;

int32_t bias_l1[64];
int32_t bias_l2[16];
int32_t labels[10000];

void cpu_mac_8x8(uint8_t *w_base, uint32_t w_offset, uint8_t *x_base,
                 uint32_t x_offset, int32_t Z[8][8]) {
  int8_t *W = (int8_t *)(w_base + w_offset);
  int8_t *X = (int8_t *)(x_base + x_offset);
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      int32_t sum = 0;
      for (int k = 0; k < 8; k++) {
        // CPU emulates exactly the array alignment
        int8_t w_val = W[(7 - c) * 8 + k];
        int8_t x_val = X[r * 8 + k];
        sum += x_val * w_val;
      }
      Z[r][c] += sum;
    }
  }
}

void run_inference(int num_batches, bool use_npu, double *time_ms,
                   double *accuracy) {
  int correct_predictions = 0;
  double start_time = get_time_us();

  // HW OCM Local pointers (H2F AXI Master mapped directly)
  volatile uint8_t *ocm_dst_addr = virt_ocm_base + dma_ocm_base + 0x8000;
  uint8_t *l1_drain_buf = virt_ddr_base + npu_out_ddr_offset;

  for (int b = 0; b < num_batches; b++) {
    int8_t H[8][64]; // Layer 1 Hidden outputs
    memset(H, 0, sizeof(H));

    if (b % 100 == 0) {
      if (use_npu)
        printf("    [INFO] Processing Batch %d (HW)...\n", b);
      else
        printf("    [INFO] Processing Batch %d (SW)...\n", b);
    }

    // ---------------- Layer 1: MAC + Bias + Shift + ReLU ----------------
    for (int j = 0; j < 8; j++) {
      if (use_npu) {
        if (b == 0 && j == 0) {
          printf("\n[DEBUG] === NPU Hardware Pipeline Diagnostics ===\n");
          printf("[DEBUG] [1] DDR Fetch Check. Input[0]=%d, Weight_L1[0]=%d\n", 
                 (int8_t)*(virt_ddr_base + inputs_offset),
                 (int8_t)*(virt_ddr_base + weights_l1_offset));
        }

        // HW ACCUMULATOR: Zero out 32-bit partial sums container (256 bytes per
        // matrix)
        for (int i = 0; i < 256 / 4; i++) {
          IOWR_32DIRECT(ocm_dst_addr, i * 4, 0);
        }
        // ARCHITECTURAL FIX: Force a blocking dummy read to flush the
        // Heavyweight AXI posting buffers. Without this, the 64 zero-writes
        // linger in the AXI bridge and eventually overwrite the NPU's
        // mathematically accumulated values asynchronously!
        volatile uint32_t dummy = IORD_32DIRECT(ocm_dst_addr, 0);
        (void)dummy;

        if (b == 0 && j == 0) {
          printf("[DEBUG] [2] OCM Zero Check (Should be 0): %d\n", 
                 (int32_t)IORD_32DIRECT(ocm_dst_addr, 0));
        }

        IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 8);

        // Accumulate exactly 98 matrix patches
        for (int t = 0; t < 98; t++) {
          uint32_t curr_w1_offset = weights_l1_offset + (t * 8 + j) * 64;
          uint32_t curr_x_offset = inputs_offset + (b * 98 + t) * 64;

          npu_load_weights(npu_ddr_base + curr_w1_offset, 1);

          IOWR(NPU_CTRL_BASE, 0x23, dma_ocm_base + 0x8000);
          IOWR(NPU_CTRL_BASE, 0x25, 64);
          IOWR(NPU_CTRL_BASE, 0x20, 1); // accum_start

          npu_load_matrix(npu_ddr_base + curr_x_offset, 1);
          npu_wait_execution();
          while ((IORD(NPU_CTRL_BASE, 0x21) & 1) == 0)
            ; // Wait accum_done
        }

        if (b == 0 && j == 0) {
            printf("[DEBUG] [4] Pre-DRAIN Sequence OCM Check (Fully accumulated 98 tiles):\n");
            printf("[DEBUG] [Row 0] ");
            for(int i=0; i<8; i++) {
                printf("%d ", (int32_t)IORD_32DIRECT(ocm_dst_addr, i*4));
            }
            printf("\n");
        }

        // LOAD BIAS for these 8 output nodes (Neuron chunk)
        for (int f = 0; f < 8; f++) {
          IOWR(NPU_CTRL_BASE, 0x200 + f, bias_l1[j * 8 + f]);
        }

        // HW DRAIN PORTION (Triggers Bias+, Shr>>, ReLU and writes 8-bit
        // Output)
        IOWR(NPU_CTRL_BASE, REG_CTRL, 4); // seq_mode=2
        IOWR(NPU_CTRL_BASE, 0x23, dma_ocm_base + 0x8000);
        IOWR(NPU_CTRL_BASE, 0x25, 64);
        IOWR(NPU_CTRL_BASE, 0x20, 1); // accum_start

        msgdma_write_stream_push(DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE,
                                 npu_ddr_base + npu_out_ddr_offset, 64);
        while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0)
          ;
        while ((IORD(NPU_CTRL_BASE, 0x21) & 1) == 0)
          ;

        if (b == 0 && j == 0) {
            printf("[DEBUG] [5] Post-DRAIN DDR Buffer Check (l1_drain_buf via CPU):\n");
            printf("[DEBUG] %d %d %d %d\n", 
              (int8_t)l1_drain_buf[0], (int8_t)l1_drain_buf[1], (int8_t)l1_drain_buf[2], (int8_t)l1_drain_buf[3]);
        }

        // Fetch DRAIN 8-bit stream from DDR, compensating for Big-Endian HW
        // Mapping
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            H[r][j * 8 + c] = (int8_t)l1_drain_buf[r * 8 + (7 - c)];
          }
        }
      } else {
        // CPU S/W Layer 1 Target
        int32_t Z[8][8] = {0};
        for (int t = 0; t < 98; t++) {
          uint32_t curr_w1_offset = weights_l1_offset + (t * 8 + j) * 64;
          uint32_t curr_x_offset = inputs_offset + (b * 98 + t) * 64;
          cpu_mac_8x8(virt_ddr_base, curr_w1_offset, virt_ddr_base,
                      curr_x_offset, Z);
        }
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            int32_t val = (Z[r][c] + bias_l1[j * 8 + c]) / 256;
            if (val < 0)
              val = 0;
            if (val > 127)
              val = 127;
            H[r][j * 8 + c] = (int8_t)val;
          }
        }
      }

      if (b == 0 && j == 0) {
        if (use_npu)
          printf("\n--- NPU Layer 1 (j=0) Output ---\n");
        else
          printf("\n--- CPU Layer 1 (j=0) Output ---\n");
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            printf("%4d ", H[r][c]);
          }
          printf("\n");
        }
      }
    }

    // ---------------- Layer 2: Final Inference (No ReLU) ----------------
    int32_t Y_final[8][16] = {0};
    uint32_t h_ddr_offset = npu_out_ddr_offset + 0x1000;
    uint8_t *h_ddr_ptr = virt_ddr_base + h_ddr_offset;

    for (int j = 0; j < 2; j++) {
      if (use_npu) {
        // HW ACCUMULATOR: Zero out buffer
        for (int i = 0; i < 256 / 4; i++) {
          IOWR_32DIRECT(ocm_dst_addr, i * 4, 0);
        }
        // ARCHITECTURAL FIX: Force a blocking dummy read to flush the AXI posting buffers.
        volatile uint32_t dummy = IORD_32DIRECT(ocm_dst_addr, 0);
        (void)dummy;

        // Reformat Intermediate Output into natively blocked HW sequence
        for (int t = 0; t < 8; t++) {
          for (int r = 0; r < 8; r++) {
            uint32_t low = 0, high = 0;
            for (int c = 0; c < 4; c++) {
              low |= (((uint32_t)(uint8_t)H[r][t * 8 + c]) << (c * 8));
              high |= (((uint32_t)(uint8_t)H[r][t * 8 + c + 4]) << (c * 8));
            }
            IOWR_32DIRECT(h_ddr_ptr, r * 8 + 0, low);
            IOWR_32DIRECT(h_ddr_ptr, r * 8 + 4, high);
          }
          // Enforce SDRAM write buffer flushing before firing MSGDMA Read
          volatile uint32_t ddr_dummy = IORD_32DIRECT(h_ddr_ptr, 0);
          (void)ddr_dummy;

          uint32_t curr_w2_offset = weights_l2_offset + (t * 2 + j) * 64;

          npu_load_weights(npu_ddr_base + curr_w2_offset, 1);
          IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 8);
          IOWR(NPU_CTRL_BASE, 0x23, dma_ocm_base + 0x8000);
          IOWR(NPU_CTRL_BASE, 0x25, 64);
          IOWR(NPU_CTRL_BASE, 0x20, 1);

          npu_load_matrix(npu_ddr_base + h_ddr_offset, 1);
          npu_wait_execution();
          while ((IORD(NPU_CTRL_BASE, 0x21) & 1) == 0)
            ;
        }

        // INTELLIGENT BYPASS:
        // Do NOT trigger DRAIN `seq_mode=2` since Logits (Layer 2 Output)
        // should NOT be ReLUClipped! ArgMax heavily relies on preserving
        // negative states. Instead, pull raw 32-bit Integer sums directly out
        // of HW memory via Linux HPS Bridge!
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            uint32_t raw_Z2 =
                IORD_32DIRECT(ocm_dst_addr, (r * 8 + c) * 4);
            Y_final[r][j * 8 + c] =
                (int32_t)raw_Z2 + bias_l2[j * 8 + c];
          }
        }
      } else {
        // CPU S/W Layer 2
        int32_t Z2[8][8] = {0};
        for (int t = 0; t < 8; t++) {
          // Flatten SW matrix locally
          uint8_t sw_h_buf[64];
          for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
              sw_h_buf[r * 8 + c] = H[r][t * 8 + c];
            }
          }
          uint32_t curr_w2_offset = weights_l2_offset + (t * 2 + j) * 64;
          cpu_mac_8x8(virt_ddr_base, curr_w2_offset, sw_h_buf, 0, Z2);
        }
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++)
            Y_final[r][j * 8 + c] = Z2[r][c] + bias_l2[j * 8 + c];
        }
      }
    }

    if (b == 0) {
      if (use_npu)
        printf("\n--- NPU Layer 2 (10-Class Logits) ---\n");
      else
        printf("\n--- CPU Layer 2 (10-Class Logits) ---\n");
      for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 10; c++) {
          printf("%6d ", Y_final[r][c]);
        }
        printf("\n");
      }
    }

    // ArgMax Classification Output
    for (int r = 0; r < 8; r++) {
      int best_class = 0;
      int32_t max_val = Y_final[r][0];
      for (int c = 1; c < 10; c++) {
        if (Y_final[r][c] > max_val) {
          max_val = Y_final[r][c];
          best_class = c;
        }
      }
      if (best_class == labels[b * 8 + r])
        correct_predictions++;
    }
  }

  double end_time = get_time_us();
  *time_ms = (end_time - start_time) / 1000.0;
  *accuracy = (double)correct_predictions / (num_batches * 8) * 100.0;
}

int main(int argc, char **argv) {
  int num_test_batches = 125; // Default 1000 images (8 images / batch)
  if (argc > 1) {
    num_test_batches = atoi(argv[1]) / 8;
  }

  printf("=== MNIST Inference Benchmark ===\n");
  printf("Images:   %d\n", num_test_batches * 8);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open /dev/mem");
    return EXIT_FAILURE;
  }
  uint8_t *lw_bridge_map =
      (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, LWHPS2FPGA_BASE);
  uint8_t *ddr_map =
      (uint8_t *)mmap(NULL, HPS_FPGA_RAM_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, HPS_FPGA_RAM_BASE);

  if (lw_bridge_map == MAP_FAILED || ddr_map == MAP_FAILED) {
    perror("mmap failed");
    return EXIT_FAILURE;
  }

  NPU_CTRL_BASE = lw_bridge_map + NPU_CTRL_OFFSET;
  DDR_READ_ST_CSR_BASE = lw_bridge_map + DDR_READ_ST_CSR_OFFSET;
  DDR_READ_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_READ_ST_DESC_OFFSET;
  DDR_WRITE_ST_CSR_BASE = lw_bridge_map + DDR_WRITE_ST_CSR_OFFSET;
  DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_WRITE_ST_DESC_OFFSET;

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  phys_ddr_base = HPS_FPGA_RAM_BASE;
  virt_ddr_base = ddr_map;

  phys_ocm_base = LWHPS2FPGA_BASE;
  virt_ocm_base = lw_bridge_map;

  printf("\nLoading Binaries...\n");
  load_binary_file("inputs.bin", virt_ddr_base + inputs_offset, 8000000);
  load_binary_file("weights_l1.bin", virt_ddr_base + weights_l1_offset, 200000);
  load_binary_file("weights_l2.bin", virt_ddr_base + weights_l2_offset, 200000);
  load_binary_file("bias_l1.bin", (uint8_t *)bias_l1, 256);
  load_binary_file("bias_l2.bin", (uint8_t *)bias_l2, 64);
  load_binary_file("labels.bin", (uint8_t *)labels, 40000);

  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 8);
  double cpu_time_ms = 0, cpu_acc = 0;
  double npu_time_ms = 0, npu_acc = 0;

  printf("\n[1] Running S/W (CPU) Inference...\n");
  run_inference(num_test_batches, false, &cpu_time_ms, &cpu_acc);
  printf("    CPU Accuracy : %.2f%%\n", cpu_acc);
  printf("    CPU Time     : %.2f ms (%.3f ms/img)\n", cpu_time_ms,
         cpu_time_ms / (num_test_batches * 8));

  printf("\n[2] Running H/W (NPU) Inference...\n");
  run_inference(num_test_batches, true, &npu_time_ms, &npu_acc);
  printf("    NPU Accuracy : %.2f%%\n", npu_acc);
  printf("    NPU Time     : %.2f ms (%.3f ms/img)\n", npu_time_ms,
         npu_time_ms / (num_test_batches * 8));

  printf("\n=== Speedup ===\n");
  if (npu_time_ms > 0) {
    printf("    H/W Acceleration: %.2f x\n", cpu_time_ms / npu_time_ms);
  }

  munmap((void *)ddr_map, HPS_FPGA_RAM_SPAN);
  munmap((void *)lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);
  return 0;
}
