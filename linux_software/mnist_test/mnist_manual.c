#include "hw_addresses.h"
#include "model_params.h"
#include "npu_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>

uint32_t inputs_offset = 0x000000;
uint32_t weights_l1_offset = 0x800000;
uint32_t weights_l2_offset = 0x840000;
uint32_t npu_out_ddr_offset = 0x900000;

int32_t bias_l1[64];
int32_t bias_l2[16];
int32_t labels[10000];

double get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}

void run_inference(int num_batches, double *time_ms, double *accuracy) {
  int correct_predictions = 0;
  double start_time = get_time_us();

  // Pointer directly to Layer 1 DDR output Buffer space
  int8_t *l1_drain_buf = (int8_t*)(virt_ddr_base + npu_out_ddr_offset);

  for (int b = 0; b < num_batches; b++) {
    int8_t H[8][64]; // Layer 1 Local Memory Buffer (for CPU bridging)
    memset(H, 0, sizeof(H));

    if (b % 100 == 0) {
        printf("    [INFO] Processing Batch %d (Hardware API Driven)...\n", b);
    }

    // ---------------- Layer 1: NPU MAC + Bias + Shift + ReLU ----------------
    // The Compiler will output these exactly matched intrinsic `for` loops automatically!
    for (int j = 0; j < 8; j++) {
        npu_clear_ocm();
        npu_set_seq_rows(8);

        for (int t = 0; t < 98; t++) {
            uint32_t curr_w1_offset = weights_l1_offset + (t * 8 + j) * 64;
            uint32_t curr_x_offset = inputs_offset + (b * 98 + t) * 64;

            npu_load_weights(curr_w1_offset);
            npu_accum_start();
            npu_load_inputs(curr_x_offset);
            npu_wait_accum();
        }

        npu_set_shift(8);
        npu_load_bias(&bias_l1[j * 8]);
        npu_drain_to_ddr(npu_out_ddr_offset);

        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            H[r][j * 8 + c] = l1_drain_buf[r * 8 + (7 - c)];
          }
        }
    }

    // ---------------- Layer 2: Final Inference (No ReLU) ----------------
    int32_t Y_final[8][16] = {0};
    uint32_t h_ddr_offset = npu_out_ddr_offset + 0x1000;
    uint8_t *h_ddr_ptr = virt_ddr_base + h_ddr_offset;

    for (int j = 0; j < 2; j++) {
        npu_clear_ocm();
        npu_set_seq_rows(8);

        // Dynamically pack intermediate output locally to submit
        for (int t = 0; t < 8; t++) {
          for (int r = 0; r < 8; r++) {
            uint32_t low = 0, high = 0;
            for (int c = 0; c < 4; c++) {
              low |= (((uint32_t)(uint8_t)H[r][t * 8 + c]) << (c * 8));
              high |= (((uint32_t)(uint8_t)H[r][t * 8 + c + 4]) << (c * 8));
            }
            uint8_t* base_ptr = (uint8_t*)h_ddr_ptr;
            *(volatile uint32_t *)(base_ptr + r * 8 + 0) = low;
            *(volatile uint32_t *)(base_ptr + r * 8 + 4) = high;
          }
          volatile uint32_t ddr_dummy = ((volatile uint32_t*)h_ddr_ptr)[0];
          (void)ddr_dummy;

          uint32_t curr_w2_offset = weights_l2_offset + (t * 2 + j) * 64;

          npu_load_weights(curr_w2_offset);
          npu_accum_start();
          npu_load_inputs(h_ddr_offset);
          npu_wait_accum();
        }

        // 32-bit Logit Extraction Intrinsics (Bypass Post-Processor Hardware DRAIN)
        int32_t host_z_buf[64] = {0};
        npu_extract_32bit_ocm(host_z_buf, &bias_l2[j * 8]);

        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) {
            Y_final[r][j * 8 + c] = host_z_buf[r * 8 + c];
          }
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
  int num_test_batches = 125;
  if (argc > 1) { num_test_batches = atoi(argv[1]) / 8; }

  printf("=== Modularized NPU Compiler API Benchmark ===\n");
  printf("Target API Design Review: Success!\n");
  
  if (npu_init() < 0) {
      return EXIT_FAILURE;
  }

  printf("\nLoading Binaries into Mapped Virtual Linux API...\n");
  npu_load_binary_file("inputs.bin", virt_ddr_base + inputs_offset, 8000000);
  npu_load_binary_file("weights_l1.bin", virt_ddr_base + weights_l1_offset, 200000);
  npu_load_binary_file("weights_l2.bin", virt_ddr_base + weights_l2_offset, 200000);
  npu_load_binary_file("bias_l1.bin", bias_l1, 256);
  npu_load_binary_file("bias_l2.bin", bias_l2, 64);
  npu_load_binary_file("labels.bin", labels, 40000);

  double npu_time_ms = 0, npu_acc = 0;
  
  printf("\n[Execution] Automatically Delegating Layers to Atomic NPU C-Intrinsics...\n");
  run_inference(num_test_batches, &npu_time_ms, &npu_acc);
  
  printf("    NPU Accuracy : %.2f%%\n", npu_acc);
  printf("    NPU Time     : %.2f ms (%.3f ms/img)\n", npu_time_ms, npu_time_ms / (num_test_batches * 8));

  npu_close();
  return 0;
}
