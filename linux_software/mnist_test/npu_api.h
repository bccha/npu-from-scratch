#ifndef NPU_API_H
#define NPU_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Exposed DDR mapping pointers if needed by the caller
extern uint32_t npu_ddr_base;
extern uint8_t* virt_ddr_base;

// 1. Initialization and Teardown
int npu_init();
void npu_close();

// 2. Data Loading Utility
int npu_load_binary_file(const char *filename, void *dest, size_t max_size);

// 3. Execution Intrinsics (These are exactly the loops emitted by the Compiler!)
void npu_set_seq_rows(int rows);
void npu_set_shift(int shift_val);
void npu_clear_ocm();

void npu_load_weights(uint32_t w_ddr_offset);
void npu_load_inputs(uint32_t x_ddr_offset);
void npu_accum_start();
void npu_wait_accum();

// 4. Post-Processor Intrinsics
void npu_load_bias(int32_t* bias_array_8_elements);
void npu_drain_to_ddr(uint32_t dest_ddr_offset);

// 5. Raw Extraction (Fallback for CPU post-processing / Final logits)
void npu_extract_32bit_ocm(int32_t* host_buffer, int32_t* bias_array_8_elements);

#endif
