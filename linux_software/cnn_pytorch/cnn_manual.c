#include "npu_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CNN Model Geometry
#define IN_C 1
#define IN_H 28
#define IN_W 28
#define K_SIZE 3
#define OUT_C 8
#define OUT_H 26
#define OUT_W 26
#define PATCHES (OUT_H * OUT_W) // 676
#define POOL_SIZE 2
#define POOL_H (OUT_H / POOL_SIZE) // 13
#define POOL_W (OUT_W / POOL_SIZE) // 13
#define FLAT_FEATURES (OUT_C * POOL_H * POOL_W) // 1352

// Memory Map
#define IMG_BASE         0x000000  // Base Inputs [10000, 1, 28, 28] -> 8MB
#define W1_BASE          0x800000  // Layer 1 Conv Weights
#define W2_BASE          0x801000  // Layer 2 FC Weights (Example offset)
#define W3_BASE          0x805000  // Layer 3 FC Weights
#define NPU_SCRATCH_IN   0x900000  // NPU Inputs Ping
#define Y_OUT_BASE       0x820000  // Y Conv Output Temporary DRAIN
#define H_FLAT_BASE      0x920000  // 1MB Global Flattened Tensor

int32_t bias_l1[8];
int32_t bias_l2[128]; // Max 16 tiles * 8
int32_t bias_l3[16];

int32_t labels[10000];

void debug_tensor_stats(const char* name, int8_t* ptr, int size) {
    int8_t min_v = 127;
    int8_t max_v = -128;
    int non_zero = 0;
    for(int i=0; i<size; i++) {
        if(ptr[i] < min_v) min_v = ptr[i];
        if(ptr[i] > max_v) max_v = ptr[i];
        if(ptr[i] != 0) non_zero++;
    }
    printf("DEBUG [%s] Range: [%d ~ %d], Non-Zeros: %d / %d\n", name, min_v, max_v, non_zero, size);
    if(non_zero > 0) {
        printf("      First 10 non-zero: ");
        int count = 0;
        for(int i=0; i<size && count<10; i++) {
            if(ptr[i] != 0) { printf("%4d ", ptr[i]); count++; }
        }
        printf("\n");
    }
}

int main() {
    printf("=== Bare-Metal CNN Manual Sandbox ===\n");
    if (npu_init() < 0) return -1;
    
    printf("\n[1] Mapping Binaries to DDR...\n");
    npu_load_binary_file("inputs.bin", virt_ddr_base + IMG_BASE, 8000000);
    int w1_sz = npu_load_binary_file("weights_l1.bin", virt_ddr_base + W1_BASE, 200000);
    int b1_sz = npu_load_binary_file("bias_l1.bin", bias_l1, 256);
    npu_load_binary_file("labels.bin", labels, 40000);
    
    printf("W1 loaded: %d bytes, B1 loaded: %d bytes.\n", w1_sz, b1_sz);

    // Grab EXACTLY the first picture (Batch 0)
    int b = 0;
    uint8_t* img_src = (uint8_t*)(virt_ddr_base + IMG_BASE + b * 28 * 28);
    printf("Label for Image 0: %d\n\n", labels[0]);

    // -------------------------------------------------------------
    // STEP 1: Im2Col Software Unrolling (CPU)
    // -------------------------------------------------------------
    printf("[2] Executing CPU Im2Col Unrolling...\n");
    int8_t* X_hw = (int8_t*)(virt_ddr_base + NPU_SCRATCH_IN);
    int in_tiles_l1 = 2; // 1 c_in * 3 * 3 = 9 features -> 2 tiles
    memset(X_hw, 0, ((PATCHES + 7)/8) * in_tiles_l1 * 64);
    
    int patch_idx = 0;
    for(int py = 0; py < OUT_H; py++) {
        for(int px = 0; px < OUT_W; px++) {
            int feature_idx = 0;
            for(int c_in = 0; c_in < IN_C; c_in++) {
                for(int ky = 0; ky < K_SIZE; ky++) {
                    for(int kx = 0; kx < K_SIZE; kx++) {
                        int orig_y = (py * 1) - 0 + ky;
                        int orig_x = (px * 1) - 0 + kx;
                        int8_t pval = 0;
                        if (orig_y >= 0 && orig_y < IN_H && orig_x >=0 && orig_x < IN_W) {
                            pval = (int8_t)img_src[c_in * (IN_H*IN_W) + orig_y * IN_W + orig_x];
                        }
                        
                        int p_tile = patch_idx / 8; int p_rem = patch_idx % 8;
                        int f_tile = feature_idx / 8; int f_rem = feature_idx % 8;
                        int mem_offset = (p_tile * in_tiles_l1 + f_tile) * 64 + (p_rem * 8 + f_rem);
                        X_hw[mem_offset] = pval;
                        feature_idx++;
                    }
                }
            }
            patch_idx++;
        }
    }
    
    debug_tensor_stats("Im2Col X_hw", X_hw, ((PATCHES + 7)/8) * in_tiles_l1 * 64);
    
    // -------------------------------------------------------------
    // STEP 2: NPU Convolution Execution
    // -------------------------------------------------------------
    printf("[3] Launching NPU Convolution...\n");
    int out_patches_tiles = (PATCHES + 7) / 8;
    int out_tiles_l1 = 1; // 8 channels -> 1 tile
    int8_t* Y_img_out = (int8_t*)(virt_ddr_base + Y_OUT_BASE);
    memset(Y_img_out, 0, OUT_C * PATCHES);
    
    for (int p_t = 0; p_t < out_patches_tiles; p_t++) {
        for (int j = 0; j < out_tiles_l1; j++) {
            npu_clear_ocm(); 
            npu_set_seq_rows(8);
            
            // MANUAL OVERRIDE: TRYING SHIFT = 8
            npu_set_shift(8); 
            
            for (int t = 0; t < in_tiles_l1; t++) {
                npu_load_weights(W1_BASE + (t * out_tiles_l1 + j) * 64);
                npu_accum_start();
                npu_load_inputs(NPU_SCRATCH_IN + (p_t * in_tiles_l1 + t) * 64);
                npu_wait_accum();
            }
            npu_load_bias(&bias_l1[j * 8]);
            
            // Drain output
            npu_drain_to_ddr(0x910000);
            
            for(int r=0; r<8; r++) {
                int absolute_patch = p_t * 8 + r; if (absolute_patch >= PATCHES) continue;
                for(int c=0; c<8; c++) {
                    int absolute_channel = j * 8 + c; if (absolute_channel >= OUT_C) continue;
                    // Natively extracted spatial map
                    Y_img_out[absolute_channel * PATCHES + absolute_patch] = ((int8_t*)(virt_ddr_base + 0x910000))[r * 8 + (7 - c)];
                }
            }
        }
    }
    
    debug_tensor_stats("Conv1 Out", Y_img_out, OUT_C * PATCHES);

    // -------------------------------------------------------------
    // STEP 3: MaxPool2D execution
    // -------------------------------------------------------------
    printf("[4] Executing CPU MaxPool2D...\n");
    int8_t* H_flat = (int8_t*)(virt_ddr_base + H_FLAT_BASE);
    
    for(int c_out=0; c_out<OUT_C; c_out++) {
        for(int py=0; py<POOL_H; py++) {
            for(int px=0; px<POOL_W; px++) {
                int8_t max_v = -128;
                for(int ky=0; ky<POOL_SIZE; ky++) {
                    for(int kx=0; kx<POOL_SIZE; kx++) {
                        int orig_y = py*POOL_SIZE + ky; int orig_x = px*POOL_SIZE + kx;
                        int8_t v = Y_img_out[c_out*(OUT_H * OUT_W) + orig_y*OUT_W + orig_x];
                        if (v > max_v) max_v = v;
                    }
                }
                H_flat[c_out*(POOL_H * POOL_W) + py*POOL_W + px] = max_v;
            }
        }
    }
    
    debug_tensor_stats("Pool1 Out", H_flat, 1352);
    // -------------------------------------------------------------
    // STEP 4: FC1 Linear execution
    // -------------------------------------------------------------
    printf("[5] Launching NPU Dense FC1...\n");
    int in_tiles_l2 = 1352 / 8; // 169
    int out_tiles_l2 = 128 / 8; // 16
    int8_t* H_flat_next = (int8_t*)(virt_ddr_base + 0x930000);
    
    for (int j = 0; j < out_tiles_l2; j++) {
        npu_clear_ocm(); npu_set_seq_rows(8);
        for (int t = 0; t < in_tiles_l2; t++) {
            for(int r=0; r<8; r++) {
                for(int c_i=0; c_i<8; c_i++) {
                    int mem_offset = (t * 64) + (r * 8) + c_i;
                    if (r == 0 && (t*8+c_i) < FLAT_FEATURES) {
                        int8_t v0 = H_flat[t*8+c_i];
                        int8_t v1 = H_flat[t*8+c_i+4];
                        if (c_i < 4) {
                            X_hw[mem_offset] = v0; X_hw[mem_offset+4] = v1;
                        }
                    } else {
                        X_hw[mem_offset] = 0;
                    }
                }
            }
            npu_load_weights(W2_BASE + (t * out_tiles_l2 + j) * 64);
            npu_accum_start();
            npu_load_inputs(NPU_SCRATCH_IN + t * 64);
            npu_wait_accum();
        }
        npu_set_shift(8);
        npu_load_bias(&bias_l2[j * 8]); npu_drain_to_ddr(0x900000);
        for(int r=0; r<8; r++) { for(int c_i=0; c_i<8; c_i++) {
            if (r==0) H_flat_next[j*8+c_i] = ((int8_t*)(virt_ddr_base + 0x900000))[r*8+(7-c_i)];
        } }
    }
    
    memcpy(H_flat, H_flat_next, 128);
    debug_tensor_stats("FC1 Out", H_flat, 128);

    // -------------------------------------------------------------
    // STEP 5: FC2 Linear Logits
    // -------------------------------------------------------------
    printf("[6] Launching NPU Dense Logits (FC2)...\n");
    int in_tiles_l3 = 128 / 8; // 16
    int out_tiles_l3 = 2; // 10 / 8 + 1 = 2 tiles
    int32_t* Y_final = (int32_t*)(virt_ddr_base + 0x910000);
    
    for (int j = 0; j < out_tiles_l3; j++) {
        npu_clear_ocm(); npu_set_seq_rows(8);
        for (int t = 0; t < in_tiles_l3; t++) {
            for(int r=0; r<8; r++) {
                for(int c_i=0; c_i<8; c_i++) {
                    int mem_offset = (t * 64) + (r * 8) + c_i;
                    if (r == 0 && (t*8+c_i) < 128) {
                        int8_t v0 = H_flat[t*8+c_i];
                        int8_t v1 = (t*8+c_i+4 < 128) ? H_flat[t*8+c_i+4] : 0;
                        if (c_i < 4) {
                            X_hw[mem_offset] = v0; X_hw[mem_offset+4] = v1;
                        }
                    } else {
                        X_hw[mem_offset] = 0;
                    }
                }
            }
            npu_load_weights(W3_BASE + (t * out_tiles_l3 + j) * 64);
            npu_accum_start();
            npu_load_inputs(NPU_SCRATCH_IN + t * 64);
            npu_wait_accum();
        }
        
        int32_t z_b[64] = {0};
        npu_extract_32bit_ocm(z_b, &bias_l3[j * 8]);
        for(int r=0; r<8; r++) { 
            for(int c_i=0; c_i<8; c_i++) { 
                Y_final[r*16 + (j*8+c_i)] = z_b[r*8+c_i]; 
            } 
        }
    }
    
    printf("DEBUG [FC2 Out Logits]: ");
    int best_class = 0; int32_t max_val = Y_final[0];
    for (int k = 0; k < 10; k++) { 
        printf("%d ", Y_final[k]);
        if (Y_final[k] > max_val) { max_val = Y_final[k]; best_class = k; } 
    }
    printf("\n");
    
    printf(">> Prediction: %d (Ground Truth: %d)\n", best_class, labels[0]);

    printf("\nManual Test End. Executed Layers: End-to-End CNN Pipeline.\n");
    
    npu_close();
    return 0;
}
