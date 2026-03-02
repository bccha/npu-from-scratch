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

#define HPS2FPGA_AXI_BASE 0xC0000000
#define HPS2FPGA_AXI_SPAN 0x20000000

// Unified Register Map
#define REG_CTRL 0
#define REG_STATUS 1
#define REG_SEQ_ROWS 6

#define NPU_MAT_SIZE 8
#define NPU_MAT_BYTES (NPU_MAT_SIZE * 8)
#define NPU_OUT_BYTES (NPU_MAT_SIZE * 32)

// Pointers
volatile uint8_t *NPU_CTRL_BASE;
volatile uint8_t *DDR_READ_ST_CSR_BASE;
volatile uint8_t *DDR_READ_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR_WRITE_ST_CSR_BASE;
volatile uint8_t *DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR3_WINDOW_BASE;

// NPU Memory mapped offsets
uint32_t npu_ddr_base = 0x20000000;
uint32_t dma_ocm_base = 0x00040000;

// Offsets within DDR (Base: HPS_FPGA_RAM_BASE)
uint32_t inputs_offset = 0x000000;
uint32_t weights_l1_offset = 0x800000;
uint32_t weights_l2_offset = 0x840000;
uint32_t bias_l1_offset = 0x850000;
uint32_t bias_l2_offset = 0x850100;

// Offsets within OCM (Base: LWHPS2FPGA_BASE + NPU_OCM_OFFSET)
uint32_t scratch_offset = 0x000000;
uint32_t npu_out_offset = 0x001000;
uint32_t l2_final_offset = 0x002000;
uint32_t ocm_bias_l1_offset = 0x003000;
uint32_t ocm_bias_l2_offset = 0x003100;

int32_t bias_l1[64];
int32_t bias_l2[16];
int32_t labels[10000];

// ==========================================
// MSGDMA Helpers
// ==========================================
void msgdma_init(volatile uint8_t *csr_base) {
  // 1. Issue Reset Command (Bit 1 of Control Register)
  IOWR_32DIRECT(csr_base, 0x04, (1 << 1));

  // 2. Wait for Hardware to Finish Resetting (Poll Status Bit 6)
  while (IORD_32DIRECT(csr_base, 0x00) & (1 << 6)) {
    // Loop actively while resetting
  }

  // 3. Clear existing status bits (W1C bits)
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);

  // 4. Enable dispatcher globally
  // Disable Interrupts, Clear Stop Dispatcher, Clear Reset.
  IOWR_32DIRECT(csr_base, 0x04, 0x00000000);
}

void msgdma_read_stream_push(volatile uint8_t *descriptor_base,
                             uint32_t src_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000300); // GO | GEN_EOP | GEN_SOP
}

void msgdma_write_stream_push(volatile uint8_t *descriptor_base,
                              uint32_t dst_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C,
                0x80000000); // GO (No END_ON_EOP to prevent early termination)
}

void npu_parse_output(volatile uint8_t *src_addr, int32_t dst_matrix[8][8]) {
  uint32_t local_src[64];
  memcpy(local_src, (const void *)src_addr, 256);

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      int hw_c = c ^ 1;
      uint32_t raw = local_src[r * 8 + hw_c];
      dst_matrix[r][c] += (int32_t)__builtin_bswap32(raw);
    }
  }
}

void npu_load_weights(uint32_t weights_addr, int num_matrices) {
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000003);
  msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, weights_addr,
                          NPU_MAT_BYTES * num_matrices);
  while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0) {
  }
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
  while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0) {
  }
  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
  }
}

void npu_post_process(uint32_t src_addr, uint32_t dst_addr, uint32_t bias_addr,
                      uint32_t num_elements, uint32_t shift_val) {
  // PP base offset is 0x10 (16 in decimal)
  // 16+2=18 for src_addr, etc.
  IOWR(NPU_CTRL_BASE, 18, src_addr);
  IOWR(NPU_CTRL_BASE, 19, dst_addr);
  IOWR(NPU_CTRL_BASE, 20, bias_addr);
  IOWR(NPU_CTRL_BASE, 21, num_elements);
  IOWR(NPU_CTRL_BASE, 22, shift_val);

  // Start (Offset 0x10 = 16)
  IOWR(NPU_CTRL_BASE, 16, 1);
  IOWR(NPU_CTRL_BASE, 16, 0);

  // Wait for Done (Offset 0x11 = 17, bit 0 is pp_done)
  int timeout = 0;
  while ((IORD(NPU_CTRL_BASE, 17) & 0x01) == 0) {
    timeout++;
    if (timeout > 1000000) {
      printf("\n[TIMEOUT] PP is stuck! STATUS=0x%08X\n",
             IORD(NPU_CTRL_BASE, 17));
      break;
    }
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

double get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

uint32_t phys_ddr_base;
uint8_t *virt_ddr_base;

uint32_t phys_ocm_base;
uint8_t *virt_ocm_base;

// CPU MAC accumulation simulation
void cpu_mac_8x8(uint8_t *w_base, uint32_t w_offset, uint8_t *x_base,
                 uint32_t x_offset, int32_t Z[8][8]) {
  int8_t *W = (int8_t *)(w_base + w_offset);
  int8_t *X = (int8_t *)(x_base + x_offset);
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      int32_t sum = 0;
      for (int k = 0; k < 8; k++) {
        // Hardware expects: w_out[t, r] = w_8x8[r, c] where t = 7-c.
        // So to get w_8x8[k, c]:  we read W[(7-c)*8 + k]
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

  for (int b = 0; b < num_batches; b++) {
    // ---------------- Layer 1 ----------------
    for (int j = 0; j < 8; j++) {
      int32_t Z[8][8] = {0};

      for (int t = 0; t < 98; t++) {
        uint32_t curr_w1_offset = weights_l1_offset + (t * 8 + j) * 64;
        uint32_t curr_x_offset = inputs_offset + (b * 98 + t) * 64;

        if (use_npu) {
          printf("L1(b=%d,j=%d,t=%d): start MAC... ", b, j, t);
          fflush(stdout);
          npu_load_weights(npu_ddr_base + curr_w1_offset, 1);
          npu_get_matrix(dma_ocm_base + npu_out_offset, 1); // DMA writes to OCM
          npu_load_matrix(npu_ddr_base + curr_x_offset, 1);
          npu_wait_execution();
          printf("done. ");
          fflush(stdout);

          if (t == 97) {
            printf("PP start... ");
            fflush(stdout);
            // On the last iteration, trigger hardware post processor
            // FPGA view: OCM is at dma_ocm_base (19-bit mapped)
            npu_post_process(dma_ocm_base + npu_out_offset,
                             dma_ocm_base + scratch_offset + j * 64,
                             dma_ocm_base + ocm_bias_l1_offset +
                                 j * 32, // bias in OCM now
                             64, SHIFT1);
            printf("PP done.\n");
            fflush(stdout);
          } else {
            printf("\n");
            fflush(stdout);
          }
        } else {
          cpu_mac_8x8(virt_ddr_base, curr_w1_offset, virt_ddr_base,
                      curr_x_offset, Z);
        }
      }

      if (!use_npu) {
        // Post-process Layer 1 Output + Format for Layer 2 (CPU Only)
        volatile uint8_t *H_dest = virt_ocm_base + scratch_offset + j * 64;
        uint32_t local_H[16];
        for (int r = 0; r < 8; r++) {
          uint32_t low_32 = 0, high_32 = 0;
          for (int c = 0; c < 4; c++) {
            int32_t raw_l = Z[r][c] + bias_l1[j * 8 + c];
            int32_t raw_h = Z[r][c + 4] + bias_l1[j * 8 + c + 4];

            int h_val_l = raw_l / SHIFT1;
            int h_val_h = raw_h / SHIFT1;

            if (h_val_l < 0)
              h_val_l = 0;
            else if (h_val_l > 127)
              h_val_l = 127;
            if (h_val_h < 0)
              h_val_h = 0;
            else if (h_val_h > 127)
              h_val_h = 127;

            low_32 |= (((uint32_t)(uint8_t)h_val_l) << (c * 8));
            high_32 |= (((uint32_t)(uint8_t)h_val_h) << (c * 8));
          }
          local_H[r * 2 + 0] = low_32;
          local_H[r * 2 + 1] = high_32;
        }
        memcpy((void *)H_dest, local_H, 64);
      }
    }

    // ---------------- Layer 2 ----------------
    int32_t Y_final[8][16] = {0};
    for (int j = 0; j < 2; j++) {
      int32_t Z2[8][8] = {0};

      for (int t = 0; t < 8; t++) {
        uint32_t curr_w2_offset = weights_l2_offset + (t * 2 + j) * 64;
        uint32_t curr_h_offset = scratch_offset + t * 64;

        if (use_npu) {
          printf("L2(b=%d,j=%d,t=%d): start MAC... ", b, j, t);
          fflush(stdout);
          npu_load_weights(npu_ddr_base + curr_w2_offset, 1);
          npu_get_matrix(dma_ocm_base + npu_out_offset, 1);
          npu_load_matrix(dma_ocm_base + curr_h_offset, 1);
          npu_wait_execution();
          printf("done. ");
          fflush(stdout);

          if (t == 7) {
            printf("PP start... ");
            fflush(stdout);
            // On the last iteration, trigger hardware post processor
            // Layer 2 outputs 16 elements per row (2 chunks of 8).
            // We'll write the final 8-bit packed results to a new OCM offset.
            npu_post_process(dma_ocm_base + npu_out_offset,
                             dma_ocm_base + l2_final_offset + j * 64,
                             dma_ocm_base + ocm_bias_l2_offset +
                                 j * 32, // bias in OCM now
                             64, SHIFT2);
            printf("PP done.\n");
            fflush(stdout);
          } else {
            printf("\n");
            fflush(stdout);
          }
        } else {
          cpu_mac_8x8(virt_ddr_base, curr_w2_offset, virt_ocm_base,
                      curr_h_offset, Z2);
        }
      }

      if (!use_npu) {
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++)
            Y_final[r][j * 8 + c] = Z2[r][c] + bias_l2[j * 8 + c];
        }
      }
    }

    // ArgMax
    for (int r = 0; r < 8; r++) {
      int best_class = 0;

      if (use_npu) {
        uint32_t l2_final_offset = 0x002000;
        // Read the 8-bit packed results back from OCM
        int8_t *final_results = (int8_t *)(virt_ocm_base + l2_final_offset);

        int8_t max_val = final_results[r * 16]; // Layer 2 output is 8x16
        for (int c = 1; c < 10; c++) {
          int8_t val = final_results[r * 16 + c];
          if (val > max_val) {
            max_val = val;
            best_class = c;
          }
        }
      } else {
        int32_t max_val = Y_final[r][0];
        for (int c = 1; c < 10; c++) {
          if (Y_final[r][c] > max_val) {
            max_val = Y_final[r][c];
            best_class = c;
          }
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
  int num_test_batches = 1; // Default to 1 for debugging

  if (argc > 1) {
    num_test_batches = atoi(argv[1]);
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

  uint8_t *h2f_bridge_map =
      (uint8_t *)mmap(NULL, HPS2FPGA_AXI_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, HPS2FPGA_AXI_BASE);

  if (lw_bridge_map == MAP_FAILED || ddr_map == MAP_FAILED ||
      h2f_bridge_map == MAP_FAILED) {
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

  phys_ocm_base = HPS2FPGA_AXI_BASE + 0x00000000;
  virt_ocm_base = h2f_bridge_map + 0x00000000;

  printf("\nLoading Binaries...\n");
  load_binary_file("inputs.bin", virt_ddr_base + inputs_offset, 8000000);
  load_binary_file("weights_l1.bin", virt_ddr_base + weights_l1_offset, 200000);
  load_binary_file("weights_l2.bin", virt_ddr_base + weights_l2_offset, 200000);
  load_binary_file("bias_l1.bin", (uint8_t *)bias_l1, 256);
  load_binary_file("bias_l2.bin", (uint8_t *)bias_l2, 64);
  load_binary_file("labels.bin", (uint8_t *)labels, 40000);

  // Copy biases to OCM so the 19-bit Post Processor bridge can access them
  memcpy(virt_ocm_base + ocm_bias_l1_offset, bias_l1, 256);
  memcpy(virt_ocm_base + ocm_bias_l2_offset, bias_l2, 64);

  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 8);
  double cpu_time_ms = 0, cpu_acc = 0;
  double npu_time_ms = 0, npu_acc = 0;

  printf("\n[0] Diagnostic: First 8x8 Matrix Result...\n");
  int32_t Z_cpu[8][8] = {0};
  int32_t Z_npu[8][8] = {0};
  cpu_mac_8x8(virt_ddr_base, weights_l1_offset, virt_ddr_base, inputs_offset,
              Z_cpu);

  npu_load_weights(npu_ddr_base + weights_l1_offset, 1);
  npu_get_matrix(dma_ocm_base + npu_out_offset, 1);
  npu_load_matrix(npu_ddr_base + inputs_offset, 1);
  npu_wait_execution();
  npu_parse_output(virt_ocm_base + npu_out_offset, Z_npu);

  printf("CPU Matrix:\n");
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      printf("%6d ", Z_cpu[r][c]);
    }
    printf("\n");
  }

  printf("NPU Matrix:\n");
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      printf("%6d ", Z_npu[r][c]);
    }
    printf("\n");
  }

  // printf("\n[1] Running S/W (CPU) Inference...\n");
  // run_inference(num_test_batches, false, &cpu_time_ms, &cpu_acc);
  // printf("    CPU Accuracy : %.2f%%\n", cpu_acc);
  // printf("    CPU Time     : %.2f ms (%.3f ms/img)\n", cpu_time_ms,
  //        cpu_time_ms / (num_test_batches * 8));

  printf("\n[2] Running H/W (NPU) Inference...\n");
  run_inference(num_test_batches, true, &npu_time_ms, &npu_acc);
  printf("    NPU Accuracy : %.2f%%\n", npu_acc);
  printf("    NPU Time     : %.2f ms (%.3f ms/img)\n", npu_time_ms,
         npu_time_ms / (num_test_batches * 8));

  // printf("\n=== Speedup ===\n");
  // if (npu_time_ms > 0) {
  //   printf("    H/W Acceleration: %.2f x\n", cpu_time_ms / npu_time_ms);
  // }

  munmap((void *)h2f_bridge_map, HPS2FPGA_AXI_SPAN);
  munmap((void *)ddr_map, HPS_FPGA_RAM_SPAN);
  munmap((void *)lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);
  return 0;
}
