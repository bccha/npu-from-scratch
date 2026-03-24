#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

int load_binary_file(const char *filename, uint8_t *dest, size_t max_size) {
  FILE *f = fopen(filename, "rb");
  if (!f) return 0;
  size_t bytes = fread(dest, 1, max_size, f);
  fclose(f);
  return bytes;
}

// Memory Sizes
#define IMG_BYTES 8000000
#define W1_BYTES  200000
#define W2_BYTES  200000

uint8_t *virt_ddr_base;
uint32_t inputs_offset = 0x000000;
uint32_t weights_l1_offset = 0x400000;
uint32_t weights_l2_offset = 0x600000;

int32_t bias_l1[8];
int32_t bias_l2[16];
int32_t labels[10000];

// CNN Layer 1
#define IN_H 28
#define IN_W 28
#define IN_C 1
#define K_H 3
#define K_W 3
#define STRIDE 2
#define OUT_H 13 
#define OUT_W 13
#define FILTERS 8
#define IM2COL_DEPTH 9

int8_t (*global_im2col)[169][16];
int8_t W_fc_linear_global[1352][16];
#define SHIFT1 256

void run_inference(int num_batches, double *time_ms, double *accuracy) {
  int correct_predictions = 0;
  if (!global_im2col) global_im2col = malloc(num_batches * sizeof(*global_im2col));

  int8_t *W_fc_raw = (int8_t *)(virt_ddr_base + weights_l2_offset);
  for (int i = 0; i < 169; i++) {
    for (int j = 0; j < 2; j++) {
      int8_t *block = W_fc_raw + (i * 2 + j) * 64;
      for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) W_fc_linear_global[i * 8 + r][j * 8 + (7 - c)] = block[c * 8 + r];
      }
    }
  }

  // pre-im2col (exclude from timed loop)
  for (int b = 0; b < num_batches; b++) {
    int8_t *img = (int8_t *)(virt_ddr_base + inputs_offset + b * IN_H * IN_W);
    int patch_idx = 0;
    for (int h = 0; h < OUT_H; h++) {
      for (int w = 0; w < OUT_W; w++) {
        int h_start = h * STRIDE;
        int w_start = w * STRIDE;
        for (int kh = 0; kh < K_H; kh++) {
          for (int kw = 0; kw < K_W; kw++) {
            global_im2col[b][patch_idx][kh * K_W + kw] = img[(h_start + kh) * IN_W + (w_start + kw)];
          }
        }
        for (int i = IM2COL_DEPTH; i < 16; i++) global_im2col[b][patch_idx][i] = 0;
        patch_idx++;
      }
    }
  }

  double start_time = get_time_us();

  // W_linear_ref setup
  int8_t *W_conv_ref = (int8_t *)(virt_ddr_base + weights_l1_offset);
  int8_t W_linear_ref[16][8];
  for (int k_chunk = 0; k_chunk < 2; k_chunk++) {
    int8_t *block = W_conv_ref + k_chunk * 64;
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) W_linear_ref[k_chunk * 8 + r][7 - c] = block[c * 8 + r];
    }
  }

  for (int b = 0; b < num_batches; b++) {
    int8_t (*im2col_buf)[16] = global_im2col[b];
    int8_t L1_out_local[1352];

    // L1: CPU Convolution
    for (int p = 0; p < 169; p++) {
      for (int f = 0; f < 8; f++) {
        int32_t sum = 0;
        for (int k = 0; k < 16; k++) sum += im2col_buf[p][k] * W_linear_ref[k][f];
        int z_val = (sum + bias_l1[f]) / SHIFT1;
        if (z_val < 0) z_val = 0;     // ReLU
        if (z_val > 127) z_val = 127; // Max Cap
        L1_out_local[p * 8 + f] = (int8_t)z_val;
      }
    }

    // L2: CPU FC Logits
    int32_t Z_fc[16];
    memset(Z_fc, 0, sizeof(Z_fc));
    for (int i = 0; i < 1352; i++) {
        for (int f = 0; f < 10; f++) {
            Z_fc[f] += L1_out_local[i] * W_fc_linear_global[i][f];
        }
    }

    // ArgMax
    int best_class = 0;
    int32_t max_val = Z_fc[0] + bias_l2[0];
    for (int c = 1; c < 10; c++) {
      int32_t val = Z_fc[c] + bias_l2[c];
      if (val > max_val) {
        max_val = val;
        best_class = c;
      }
    }
    if (best_class == labels[b]) correct_predictions++;
  }

  double end_time = get_time_us();
  *time_ms = (end_time - start_time) / 1000.0;
  *accuracy = (double)correct_predictions / num_batches * 100.0;
}

int main(int argc, char **argv) {
  int num_test_batches = 1000;
  if (argc > 1) num_test_batches = atoi(argv[1]);

  printf("=== Pure CPU CNN Inference Benchmark ===\n");
  printf("Images:   %d\n", num_test_batches);

  virt_ddr_base = malloc(10000000); // 10MB virtual RAM instead of FPGA /dev/mem

  printf("\nLoading Binaries...\n");
  if(load_binary_file("inputs.bin", virt_ddr_base + inputs_offset, 8000000) == 0) {
      printf("[ERROR] Failed to load inputs.bin\n"); return -1;
  }
  load_binary_file("weights_l1.bin", virt_ddr_base + weights_l1_offset, 200000);
  load_binary_file("weights_l2.bin", virt_ddr_base + weights_l2_offset, 200000);
  load_binary_file("bias_l1.bin", (uint8_t *)bias_l1, 256);
  load_binary_file("bias_l2.bin", (uint8_t *)bias_l2, 64);
  load_binary_file("labels.bin", (uint8_t *)labels, 40000);

  double cpu_time_ms = 0, cpu_acc = 0;
  
  printf("\n[1] Running S/W (CPU) Inference...\n");
  run_inference(num_test_batches, &cpu_time_ms, &cpu_acc);
  
  printf("    CPU Accuracy : %.2f%%\n", cpu_acc);
  printf("    CPU Time     : %.2f ms (%.3f ms/img)\n", cpu_time_ms, cpu_time_ms / num_test_batches);

  free(virt_ddr_base);
  return 0;
}
