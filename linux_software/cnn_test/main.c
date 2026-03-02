#include "hw_addresses.h"
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

// ==========================================
// MSGDMA Helpers
// ==========================================
void msgdma_init(volatile uint8_t *csr_base) {
  // 1. Issue Reset Command (Bit 1 of Control Register)
  IOWR_32DIRECT(csr_base, 0x04, (1 << 1));

  // 2. Wait for Hardware to Finish Resetting (Poll Status Bit 6)
  volatile int timeout = 0;
  while (IORD_32DIRECT(csr_base, 0x00) & (1 << 6)) {
    // Loop actively while resetting
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] msgdma_init: Reset stuck!\n");
      break;
    }
  }

  // 3. Clear existing status bits (W1C bits)
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);

  // 4. Enable dispatcher globally
  // Disable Interrupts, Clear Stop Dispatcher, Clear Reset.
  IOWR_32DIRECT(csr_base, 0x04, 0x00000000);
}

void msgdma_read_stream_push(volatile uint8_t *csr_base,
                             volatile uint8_t *descriptor_base,
                             uint32_t src_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000300); // GO | GEN_EOP | GEN_SOP
}

void msgdma_write_stream_push(volatile uint8_t *csr_base,
                              volatile uint8_t *descriptor_base,
                              uint32_t dst_addr, uint32_t length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);
  IOWR_32DIRECT(descriptor_base, 0x0C,
                0x80000000); // GO (No END_ON_EOP to prevent early termination)
}

void npu_parse_output_row(volatile uint8_t *src_addr, int32_t *dst_row) {
  for (int c = 0; c < 8; c++) {
    int hw_c = c ^ 1; // NPU hardware output is swapped in adjacent word pairs
    uint32_t raw = IORD_32DIRECT(src_addr, hw_c * 4);
    dst_row[c] = (int32_t)__builtin_bswap32(raw);
  }
}

void npu_load_weights(uint32_t weights_addr, int num_bytes) {
  // Wait until Read FIFO has space
  volatile int timeout = 0;
  while (1) {
    int read_full = IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & (1 << 2);
    if (!read_full)
      break;
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] npu_load_weights: Read FIFO stuck full!\n");
      break;
    }
  }

  msgdma_read_stream_push(DDR_READ_ST_CSR_BASE,
                          DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, weights_addr,
                          num_bytes);
  IOWR(NPU_CTRL_BASE, REG_CTRL,
       0x00000001); // seq_mode=0 (Weight load), start=1

  // Wait for DMA flush
  timeout = 0;
  while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0) {
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] npu_load_weights: DMA flush stuck!\n");
      break;
    }
  }

  // Wait for sequencer idle before latch
  timeout = 0;
  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] npu_load_weights: NPU sequencer stuck!\n");
      break;
    }
  }

  // Latch Weight shift-register into internal PE bounds
  IOWR(NPU_CTRL_BASE, 7, 1);
  IOWR(NPU_CTRL_BASE, 7, 0);
}

void npu_run_inference(uint32_t inputs_addr, uint32_t dst_addr, int in_bytes,
                       int out_bytes, int rows) {
  // 1. Check if we have space in both FIFOs simultaneously
  volatile int timeout = 0;
  while (1) {
    int write_full = IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & (1 << 2);
    int read_full = IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & (1 << 2);
    if (!write_full && !read_full)
      break;
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] npu_run_inference: FIFOs stuck full!\n");
      printf("write_full: %d, read_full: %d\n", write_full != 0,
             read_full != 0);
      break;
    }
  }

  // 2. Setup Output Stream (Write DMA) FIRST so it's ready to catch data
  msgdma_write_stream_push(DDR_WRITE_ST_CSR_BASE,
                           DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE, dst_addr,
                           out_bytes);

  // 2. Setup NPU Rows to let sequencer know when to emit EOP
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, rows);

  // 3. Trigger NPU FSM (seq_mode=1 (Execution), seq_start=1)
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000003);

  // 4. Blast Input Stream (Read DMA) LAST to shove data into the waiting NPU
  // pipeline
  msgdma_read_stream_push(DDR_READ_ST_CSR_BASE,
                          DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, inputs_addr,
                          in_bytes);
}

void npu_wait_execution() {
  volatile int timeout = 0;
  while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0) {
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] NPU Execution stuck at: MSGDMA WRITE BUSY!\n");
      printf("Write CSR Status: 0x%08X\n",
             IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0));
      break;
    }
  }

  timeout = 0;
  while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0) {
    timeout++;
    if (timeout > 200000000) {
      printf("\n[TIMEOUT] NPU Execution stuck at: NPU REG_STATUS BUSY!\n");
      printf("NPU REG_STATUS: 0x%08X\n", IORD(NPU_CTRL_BASE, REG_STATUS));
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

// Memory Layout Constants
uint32_t phys_base;
uint8_t *virt_base;
uint32_t inputs_offset = 0x000000;
uint32_t conv_w_offset = 0x800000;
uint32_t fc_w_offset = 0x840000;
uint32_t scratch_offset = 0x900000;
uint32_t im2col_offset = 0xA00000;
uint32_t npu_out_offset = 0xB00000;

#define SHIFT1 128 // Fake scaling value for original data

// CNN Parameters
#define INPUT_H 28
#define INPUT_W 28
#define INPUT_C 1
#define CONV_K 3
#define CONV_F 8
#define CONV_STRIDE 2

// Output dimensions
#define OUT_H (((INPUT_H - CONV_K) / CONV_STRIDE) + 1) // 13
#define OUT_W (((INPUT_W - CONV_K) / CONV_STRIDE) + 1) // 13
#define NUM_PATCHES (OUT_H * OUT_W)                    // 169

int8_t conv_bias[8];
int8_t fc_bias[16];
uint8_t labels[10000];

void im2col_cpu(uint8_t *image, int8_t *col_matrix) {
  // image is [28x28x1]
  // col_matrix is [NUM_PATCHES, 9] mapped to [NUM_PATCHES, 16] for 8x8 NPU
  // padding NPU handles matrices in 8x8 blocks. K*K*C = 9 -> Needs 2 blocks
  // horizontally (16 columns). So the width of the hardware matrix is 16.

  int patch_idx = 0;
  for (int h = 0; h < OUT_H; h++) {
    for (int w = 0; w < OUT_W; w++) {
      int h_start = h * CONV_STRIDE;
      int w_start = w * CONV_STRIDE;

      // Extract Kh x Kw patch
      int col_idx = 0;
      for (int kh = 0; kh < CONV_K; kh++) {
        for (int kw = 0; kw < CONV_K; kw++) {
          int r = h_start + kh;
          int c = w_start + kw;
          // The original Python un-normalized data is integer 0-255.
          // Network was trained on (X - 128)/128.0
          // We must convert the raw byte to a signed int8 for Hardware MAC
          int val = (int)image[r * INPUT_W + c] - 128;
          // Cap bounds if needed
          if (val < -128)
            val = -128;
          if (val > 127)
            val = 127;

          col_matrix[patch_idx * 16 + col_idx] = (int8_t)val;
          col_idx++;
        }
      }
      // Pad the rest of the 16 elements with zeros
      for (; col_idx < 16; ++col_idx) {
        col_matrix[patch_idx * 16 + col_idx] = 0;
      }
      patch_idx++;
    }
  }
}

void run_cnn_inference(int num_batches, bool use_npu, double *time_ms,
                       double *accuracy) {
  int correct_predictions = 0;
  double start_time = get_time_us();

  // NPU requires seq row length to be 16 to match the patched weights (9 -> 16)
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 16);

  for (int b = 0; b < num_batches; b++) {
    uint8_t *img = virt_base + inputs_offset + (b * INPUT_H * INPUT_W);
    int8_t *X_col = (int8_t *)(virt_base + im2col_offset);

    // 1. Software im2col Generation
    im2col_cpu(img, X_col);

    // Output feature map size: [NUM_PATCHES, CONV_F] => [169, 8]
    int32_t Z_conv[NUM_PATCHES][8] = {0};

    // 2. Convolution via Pipelined MSGDMA Multi-row execution
    if (use_npu) {
      // Tell NPU sequencer 1 output takes K=16 iterations
      IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 16);

      // Process in chunks of 8 patches (NPU Max Batch)
      for (int patch_start = 0; patch_start < NUM_PATCHES; patch_start += 8) {
        int chunk_size = 8;
        if (patch_start + chunk_size > NUM_PATCHES) {
          chunk_size = NUM_PATCHES - patch_start;
        }

        // Load the 3x3x8 Filter kernel into the NPU PE latches
        // Weights must be reloaded for each independent DMA burst
        npu_load_weights(phys_base + conv_w_offset, 128);

        // Stream chunk of patches
        npu_run_inference(phys_base + im2col_offset + (patch_start * 16),
                          phys_base + npu_out_offset, chunk_size * 16,
                          chunk_size * 32, 16);

        npu_wait_execution();

        // Extract result matrix block linearly mapping rows
        for (int p = 0; p < chunk_size; p++) {
          int32_t raw_row[8];
          npu_parse_output_row(virt_base + npu_out_offset + (p * 8 * 4),
                               raw_row);
          for (int c = 0; c < 8; c++) {
            Z_conv[patch_start + p][c] = raw_row[c] + conv_bias[c];
          }
        }
      }
    }

    // 3. ReLU and Flatten for FC
    // Size: 169 * 8 = 1352 dimensions
    int8_t *A_fc_in = (int8_t *)(virt_base + scratch_offset);
    for (int p = 0; p < NUM_PATCHES; p++) {
      for (int f = 0; f < 8; f++) {
        int val = Z_conv[p][f];
        // Scale emulation - temporary simplistic right shift for Fake
        // Quantization
        val = val >> 9;
        if (val < 0)
          val = 0; // ReLU
        if (val > 127)
          val = 127;
        A_fc_in[p * 8 + f] = (int8_t)val;
      }
    }

    // 4. Fully Connected Layer (1 x 1352) * (1352 x 10)
    IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 1352);
    int32_t Z_fc[8][16] = {0}; // FC pad to 16

    for (int j = 0; j < 2; j++) {
      if (use_npu) {
        uint32_t curr_w_offset = fc_w_offset + (j * 1352 * 8);

        // Tell NPU sequencer 1 output takes K=1 iterations per chunk
        IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 1);

        for (int chunk = 0; chunk < 169; chunk++) {
          // Load 1 tiled 8x8 weight slice (64 bytes)
          npu_load_weights(phys_base + curr_w_offset + (chunk * 64), 64);

          // Stream 1 tile of FC inputs (8 elements * 1 byte = 8 bytes) -> 32
          // byte result
          npu_run_inference(phys_base + scratch_offset + (chunk * 8),
                            phys_base + npu_out_offset, 8, 32, 1);

          npu_wait_execution();

          int32_t raw_row[8];
          npu_parse_output_row(virt_base + npu_out_offset, raw_row);

          for (int c = 0; c < 8; c++) {
            Z_fc[0][j * 8 + c] += raw_row[c];
          }
        }

        // Add bias at the end of the sum
        for (int c = 0; c < 8; c++) {
          Z_fc[0][j * 8 + c] += fc_bias[j * 8 + c];
        }
      }
    }

    // 5. ArgMax
    int best_class = 0;
    int32_t max_val = Z_fc[0][0];
    for (int c = 1; c < 10; c++) {
      if (Z_fc[0][c] > max_val) {
        max_val = Z_fc[0][c];
        best_class = c;
      }
    }
    if (best_class == labels[b]) {
      correct_predictions++;
    }
  }

  double end_time = get_time_us();
  *time_ms = (end_time - start_time) / 1000.0;
  *accuracy = (double)correct_predictions / num_batches * 100.0;
}

int main(int argc, char **argv) {
  int num_test_batches = 10;
  if (argc > 1) {
    num_test_batches = atoi(argv[1]);
  }

  printf("=== CNN im2col Inference Benchmark ===\n");
  printf("Images:   %d\n", num_test_batches);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0)
    return EXIT_FAILURE;

  uint8_t *lw_bridge_map =
      (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, LWHPS2FPGA_BASE);
  uint8_t *ddr_map =
      (uint8_t *)mmap(NULL, HPS_FPGA_RAM_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, HPS_FPGA_RAM_BASE);

  NPU_CTRL_BASE = lw_bridge_map + NPU_CTRL_OFFSET;
  DDR_READ_ST_CSR_BASE = lw_bridge_map + DDR_READ_ST_CSR_OFFSET;
  DDR_READ_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_READ_ST_DESC_OFFSET;
  DDR_WRITE_ST_CSR_BASE = lw_bridge_map + DDR_WRITE_ST_CSR_OFFSET;
  DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_WRITE_ST_DESC_OFFSET;

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  // Only initializing one virt_base offset mapping
  // offset mapping in global space
  phys_base = HPS_FPGA_RAM_BASE;
  virt_base = ddr_map;

  printf("\nLoading CNN Binaries...\n");
  // Only load the required ones
  load_binary_file("../mnist_test/t10k-images-idx3-ubyte",
                   virt_base + inputs_offset + 16,
                   7840000); // offset 16 for idx bytes
  load_binary_file("conv_weights.bin", virt_base + conv_w_offset, 1000);
  load_binary_file("fc_weights.bin", virt_base + fc_w_offset, 25000);
  load_binary_file("conv_bias.bin", (uint8_t *)conv_bias, 8);
  load_binary_file("fc_bias.bin", (uint8_t *)fc_bias, 16);

  // Read labels properly bypassing 8 bytes header
  FILE *f_label = fopen("../mnist_test/t10k-labels-idx1-ubyte", "rb");
  if (f_label) {
    fseek(f_label, 8, SEEK_SET);
    if (fread(labels, 1, 10000, f_label) != 10000) {
      printf("Warning: Failed to read all 10000 labels\n");
    }
    fclose(f_label);
  }

  // Copy images downward by 16 bytes to ignore header
  for (int i = 0; i < num_test_batches * 784; i++) {
    *(virt_base + inputs_offset + i) = *(virt_base + inputs_offset + 16 + i);
  }

  double npu_time_ms = 0, npu_acc = 0;

  printf("\n[1] Running H/W (NPU CNN) Inference...\n");
  run_cnn_inference(num_test_batches, true, &npu_time_ms, &npu_acc);
  printf("    NPU Accuracy : %.2f%%\n", npu_acc);
  printf("    NPU Time     : %.2f ms (%.3f ms/img)\n", npu_time_ms,
         npu_time_ms / num_test_batches);

  munmap((void *)ddr_map, HPS_FPGA_RAM_SPAN);
  munmap((void *)lw_bridge_map, LWHPS2FPGA_SPAN);
  close(fd);
  return 0;
}
