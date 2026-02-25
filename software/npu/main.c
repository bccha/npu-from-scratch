#include "common.h"
#include <alt_types.h>
#include <io.h>
#include <stdio.h>
#include <system.h>

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
#define NPU_MAT_BYTES (NPU_MAT_SIZE * 8) // 64 bytes per input/weight matrix
#define NPU_OUT_BYTES                                                          \
  (NPU_MAT_SIZE * 32) // 256 bytes per output matrix (8 rows * 256-bit)

// ==========================================
// Data Formatting API
// ==========================================

// 0. C 배열 -> NPU 하드웨어 형식 변환 (입력 데이터: 행 단위 순차 전송)
void npu_format_inputs(alt_u32 dst_addr, signed char src_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    alt_u32 low_32 = 0;
    alt_u32 high_32 = 0;

    for (int c = 0; c < 4; c++) {
      low_32 |= (((alt_u32)(unsigned char)src_matrix[r][c]) << (c * 8));
      high_32 |= (((alt_u32)(unsigned char)src_matrix[r][c + 4]) << (c * 8));
    }

    IOWR_32DIRECT(dst_addr, r * 8 + 0, low_32);
    IOWR_32DIRECT(dst_addr, r * 8 + 4, high_32);
  }
}

// 0.5. C 배열 -> NPU 가중치 형식 변환 (가중치: 열 단위 역순 전송, Col 7 -> Col
// 0)
void npu_format_weights(alt_u32 dst_addr, signed char src_matrix[8][8]) {
  for (int t = 0; t < 8; t++) {
    int c = 7 - t; // Col 7 down to 0
    alt_u32 low_32 = 0;
    alt_u32 high_32 = 0;

    for (int r = 0; r < 4; r++) {
      low_32 |= (((alt_u32)(unsigned char)src_matrix[r][c]) << (r * 8));
      high_32 |= (((alt_u32)(unsigned char)src_matrix[r + 4][c]) << (r * 8));
    }

    // 버스트 순서: t=0일 때 Col 7 데이터 경록, t=1일 때 Col 6...
    IOWR_32DIRECT(dst_addr, t * 8 + 0, low_32);
    IOWR_32DIRECT(dst_addr, t * 8 + 4, high_32);
  }
}

// 0.8. NPU 하드웨어 출력 형식 -> C 배열 변환
void npu_parse_output(alt_u32 src_addr, alt_u32 dst_matrix[8][8]) {
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      // MSGDMA Avalon-ST to Avalon-MM 64-bit bridge causes Big-Endian byte
      // swap. E.g., {Col 1, Col 0} arrives as Addr 0: bswap(Col 1), Addr 4:
      // bswap(Col 0).
      int hw_c = c ^ 1;
      alt_u32 raw = IORD_32DIRECT(src_addr, (r * 8 + hw_c) * 4);
      dst_matrix[r][c] = __builtin_bswap32(raw);
    }
  }
}

// ==========================================
// NPU Control API
// ==========================================

// 1. 가중치 로드 (Weight Load) - 다중 행렬 지원
void npu_load_weights(alt_u32 weights_addr, int num_matrices) {
  // NPU를 Load Weight 모드(1)로 설정 -> CSR 값 0x3 (Bit1: seq_mode=1, Bit0:
  // seq_start=1)
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000003);
  // MSGDMA Read 큐잉 (메모리 -> st_sink)
  msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, weights_addr,
                          NPU_MAT_BYTES * num_matrices);
  // 완료 대기
  while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0)
    ;

  // Wait for the slow 50MHz FPGA internal systolic processing to complete
  // before latching the data! (seq_busy bit is bit 0 in NPU_CTRL offset 1).
  while ((IORD(NPU_CTRL_BASE, 1) & 0x01) != 0)
    ;

  // Latch the active weights
  IOWR(NPU_CTRL_BASE, 7, 1);
  IOWR(NPU_CTRL_BASE, 7, 0);
}

// 2. 결과 가져오기 (Matrix Get) - 여러 개 저장 대기 가능
void npu_get_matrix(alt_u32 dst_addr, int num_matrices) {
  // MSGDMA Write 큐잉 (st_source -> 메모리)
  msgdma_write_stream_push(DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE, dst_addr,
                           NPU_OUT_BYTES * num_matrices);
}

// 3. 행렬 연산 데이터 피딩 (Matrix Load) - 실행 트리거
void npu_load_matrix(alt_u32 inputs_addr, int num_matrices) {
  // NPU를 Execute 모드(1)로 설정
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000001);
  // MSGDMA Read 큐잉 (메모리 -> st_sink)
  msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, inputs_addr,
                          NPU_MAT_BYTES * num_matrices);
}

// 4. 연산 완료 대기
void npu_wait_execution() {
  // 결과 수집(Write MSGDMA)이 완벽히 끝날 때까지 대기
  while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0)
    ;

  // Wait for NPU calculation to be totally idle on the FPGA fabric
  while ((IORD(NPU_CTRL_BASE, 1) & 0x01) != 0)
    ;
}

// ==========================================
// System Validation
// ==========================================

void verify_full_system() {
  printf("\nStarting Full System Matrix Validation (Fixed 8x8 HW with 4x4 "
         "submatrix)...\n");

  // Initialize both MSGDMA Dispatchers by clearing the Stop bit and performing
  // Soft Resets
  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  // Force NPU Sequencer to IDLE state (Double write to flush)
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0);
  IOWR(NPU_CTRL_BASE, REG_CTRL, 0);
  while ((IORD(NPU_CTRL_BASE, 1) & 0x01) != 0)
    ;

  IOWR_32DIRECT(ADDRESS_SPAN_EXTENDER_0_CNTL_BASE, 0, 0x20000000);
  alt_u32 physical_base = 0x20000000;
  alt_u32 weights_addr = DDR3_WINDOW_BASE;
  alt_u32 inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  alt_u32 dst_addr = DDR3_WINDOW_BASE + 0x2000;

  printf("Clearing Memories...\n");
  for (int i = 0; i < 64; i++) { // Clear 256 bytes per region
    IOWR_32DIRECT(weights_addr, i * 4, 0);
    IOWR_32DIRECT(inputs_addr, i * 4, 0);
    IOWR_32DIRECT(dst_addr, i * 4, 0);
  }

  printf("Preparing 8x8 Identity Weight Matrix...\n");

  signed char test_weights[8][8] = {0};
  signed char test_inputs[8][8] = {0};

  // Weights = 8x8 Identity
  for (int i = 0; i < 8; i++) {
    test_weights[i][i] = 1;
  }

  // Inputs = 1 to 64
  int val = 1;
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      test_inputs[r][c] = val++;
    }
  }

  // C 배열을 64-byte 크기의 DMA 친화적인 포맷으로 변환 후 DDR에 기록
  npu_format_weights(weights_addr, test_weights);
  npu_format_inputs(inputs_addr, test_inputs);

  // Flush the Nios II Data Cache so that MSGDMA (Hardware) sees the new data
  alt_dcache_flush_all();

  printf("Phase 1: Loading Weights via MSGDMA API...\n");
  npu_load_weights(physical_base, 1); // 1 matrix
  printf("Weights Loaded!\n");

  printf("Phase 2: Execution via MSGDMA API...\n");
  npu_get_matrix(physical_base + 0x2000, 1);
  npu_load_matrix(physical_base + 0x1000, 1);

  npu_wait_execution();
  printf("Execution Finished!\n\n");

  int errors = 0;

  printf("Verifying Output (Expecting Y=X for 8x8 matrix)...\n");

  alt_u32 hw_matrix[8][8];
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
      alt_u32 hw_val = hw_matrix[r][c];
      alt_u32 np_val = r * 8 + c + 1; // 1 to 64

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
  printf("\nStarting Streaming Batch Test (10 Matrices)...\n");

  msgdma_init(DDR_READ_ST_CSR_BASE);
  msgdma_init(DDR_WRITE_ST_CSR_BASE);

  IOWR_32DIRECT(ADDRESS_SPAN_EXTENDER_0_CNTL_BASE, 0, 0x20000000);
  alt_u32 physical_base = 0x20000000;
  alt_u32 weights_addr = DDR3_WINDOW_BASE;
  alt_u32 inputs_addr = DDR3_WINDOW_BASE + 0x1000;
  alt_u32 outputs_addr = DDR3_WINDOW_BASE + 0x8000;

  // 1. Prepare 1 Weight Matrix (Identity)
  signed char weight_matrix[8][8];
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      weight_matrix[r][c] = (r == c) ? 1 : 0;
    }
  }
  npu_format_weights(weights_addr, weight_matrix);

  // 2. Prepare 10 Input Matrices
  for (int i = 0; i < 10; i++) {
    signed char in_mat[8][8];
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        // -128 to 127
        in_mat[r][c] = (signed char)(((i * 10 + r * 8 + c) % 256) - 128);
      }
    }
    npu_format_inputs(inputs_addr + i * NPU_MAT_BYTES, in_mat);
  }

  printf("Clearing Memories...\n");
  for (int i = 0; i < (10 * NPU_OUT_BYTES) / 4; i++) {
    IOWR_32DIRECT(outputs_addr, i * 4, 0);
  }

  // Flush the Nios II Data Cache so that MSGDMA (Hardware) sees the new data
  alt_dcache_flush_all();

  printf("Loading Weights...\n");
  npu_load_weights(physical_base, 1);

  printf("Firing 10-Batch Streaming Pipeline...\n");

  // Set total sequence rows for EOP generation (10 batches * 8 rows = 80 rows)
  IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 10 * 8);

  // Queue 10 output reads
  npu_get_matrix(physical_base + 0x8000, 10);
  // Queue 10 input writes
  npu_load_matrix(physical_base + 0x1000, 10);

  // Wait for all execution to stream through hardware
  npu_wait_execution();

  // Validate results
  int total_errors = 0;
  for (int i = 0; i < 10; i++) {
    alt_u32 hw_matrix[8][8];
    npu_parse_output(outputs_addr + i * NPU_OUT_BYTES, hw_matrix);

    int errors = 0;
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        alt_u32 hw_val = hw_matrix[r][c];

        // Expected is exactly what went in since weight is Identity matrix.
        // Cast to signed char to mimic 8-bit, then cast to int to see 32-bit
        // sign extension.
        int expected_signed =
            (int)((signed char)(((i * 10 + r * 8 + c) % 256) - 128));
        alt_u32 expected = (alt_u32)expected_signed;

        if (hw_val != expected) {
          if (errors < 5) { // Print only first 5 errors per batch
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
  // NEW RTL needs valid_in_x=1 AND weight_shift_in=1 to load weight.
  // In `mac_pe_ctrl.v`: valid_in=bit1, load_weight=bit0 -> 11(binary) = 3
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 3);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);

  // The PE is double-buffered. Trigger `weight_latch_en` (NPU_CTRL offset 7).
  IOWR(NPU_CTRL_BASE, 7, 1);
  IOWR(NPU_CTRL_BASE, 7, 0);

  IOWR(NPU_CTRL_BASE, REG_PE_X_IN, 3);
  IOWR(NPU_CTRL_BASE, REG_PE_Y_IN, 10);
  // Execute MAC: valid_in_x=1, valid_in_y=1, weight_shift_in=0
  // valid_in=bit1, load_weight=bit0 -> 10(binary) = 2
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 2);
  IOWR(NPU_CTRL_BASE, REG_PE_CTRL, 0);

  alt_u32 result = IORD(NPU_CTRL_BASE, REG_PE_Y_OUT);
  printf("Result: %d (Expected: 31)\n", (int)result);

  if (result == 31)
    printf("MAC PE Test: PASS\n");
  else
    printf("MAC PE Test: FAIL\n");
}

int main() {
  while (1) {
    printf("\nNPU System Verification (Full Framework)\n");
    printf("----------------------------------------------\n");
    printf("1. Verify MAC PE\n");
    printf("2. Verify Full System Data path\n");
    printf("3. Verify 10-Batch Streaming Pipeline\n");
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
    } else if (c == 'q') {
      printf("Exiting...\n");
      break;
    }
  }

  return 0;
}
