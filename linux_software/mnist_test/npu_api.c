#include "npu_api.h"
#include "hw_addresses.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define IOWR_32DIRECT(base, offset, data) (*(volatile uint32_t *)((uint8_t *)(base) + (offset)) = (data))
#define IORD_32DIRECT(base, offset)       (*(volatile uint32_t *)((uint8_t *)(base) + (offset)))
#define IOWR(base, reg, data)             (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)) = (data))
#define IORD(base, reg)                   (*(volatile uint32_t *)((uint8_t *)(base) + ((reg) * 4)))

#define REG_CTRL 0
#define REG_STATUS 1
#define REG_SEQ_ROWS 6
#define NPU_MAT_BYTES 64

volatile uint8_t *NPU_CTRL_BASE;
volatile uint8_t *DDR_READ_ST_CSR_BASE;
volatile uint8_t *DDR_READ_ST_DESCRIPTOR_SLAVE_BASE;
volatile uint8_t *DDR_WRITE_ST_CSR_BASE;
volatile uint8_t *DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE;

uint32_t phys_ddr_base;
uint8_t *virt_ddr_base;
volatile uint8_t *virt_ocm_base;

uint32_t npu_ddr_base = 0x20000000;
uint32_t dma_ocm_base = 0x00040000;
static int mem_fd = -1;

void msgdma_init(volatile uint8_t *csr_base) {
    IOWR_32DIRECT(csr_base, 0x04, (1 << 1)); // stop
    while (IORD_32DIRECT(csr_base, 0x00) & (1 << 6)); // wait stop
    IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF); // clear status
    IOWR_32DIRECT(csr_base, 0x04, 0x00000000); // clear stop
}

void msgdma_read_stream_push(volatile uint8_t *descriptor_base, uint32_t src_addr, uint32_t length) {
    IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
    IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
    IOWR_32DIRECT(descriptor_base, 0x08, length);
    IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000300);
}

void msgdma_write_stream_push(volatile uint8_t *descriptor_base, uint32_t dst_addr, uint32_t length) {
    IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
    IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
    IOWR_32DIRECT(descriptor_base, 0x08, length);
    IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000000);
}

int npu_init() {
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) return -1;
    uint8_t *lw_bridge_map = (uint8_t *)mmap(NULL, LWHPS2FPGA_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, LWHPS2FPGA_BASE);
    uint8_t *ddr_map = (uint8_t *)mmap(NULL, HPS_FPGA_RAM_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, HPS_FPGA_RAM_BASE);
    if (lw_bridge_map == MAP_FAILED || ddr_map == MAP_FAILED) return -1;

    NPU_CTRL_BASE = lw_bridge_map + NPU_CTRL_OFFSET;
    DDR_READ_ST_CSR_BASE = lw_bridge_map + DDR_READ_ST_CSR_OFFSET;
    DDR_READ_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_READ_ST_DESC_OFFSET;
    DDR_WRITE_ST_CSR_BASE = lw_bridge_map + DDR_WRITE_ST_CSR_OFFSET;
    DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE = lw_bridge_map + DDR_WRITE_ST_DESC_OFFSET;

    msgdma_init(DDR_READ_ST_CSR_BASE);
    msgdma_init(DDR_WRITE_ST_CSR_BASE);

    phys_ddr_base = HPS_FPGA_RAM_BASE;
    virt_ddr_base = ddr_map;
    virt_ocm_base = lw_bridge_map;
    IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, 8);
    return 0;
}

void npu_close() {
    if (virt_ddr_base) munmap((void *)virt_ddr_base, HPS_FPGA_RAM_SPAN);
    if (virt_ocm_base) munmap((void *)virt_ocm_base, LWHPS2FPGA_SPAN);
    if (mem_fd >= 0) close(mem_fd);
}

int npu_load_binary_file(const char *filename, void *dest, size_t max_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    size_t bytes = fread(dest, 1, max_size, f);
    fclose(f);
    return bytes;
}

void npu_set_seq_rows(int rows) {
    IOWR(NPU_CTRL_BASE, REG_SEQ_ROWS, rows);
}

void npu_set_shift(int shift_val) {
    IOWR(NPU_CTRL_BASE, 2, shift_val);
}

void npu_clear_ocm() {
    volatile uint8_t *ocm_dst_addr = virt_ocm_base + dma_ocm_base + 0x8000;
    for (int i = 0; i < 256 / 4; i++) IOWR_32DIRECT(ocm_dst_addr, i * 4, 0);
    volatile uint32_t dummy = IORD_32DIRECT(ocm_dst_addr, 0); (void)dummy;
}

void npu_load_weights(uint32_t w_ddr_offset) {
    IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000003); // seq_mode=3
    msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, npu_ddr_base + w_ddr_offset, NPU_MAT_BYTES);
    
    while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0);
    while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0);
    
    IOWR(NPU_CTRL_BASE, 7, 1);
    IOWR(NPU_CTRL_BASE, 7, 0);
}

void npu_load_inputs(uint32_t x_ddr_offset) {
    IOWR(NPU_CTRL_BASE, REG_CTRL, 0x00000001); // seq_mode=1
    msgdma_read_stream_push(DDR_READ_ST_DESCRIPTOR_SLAVE_BASE, npu_ddr_base + x_ddr_offset, NPU_MAT_BYTES);
}

void npu_accum_start() {
    IOWR(NPU_CTRL_BASE, 0x23, dma_ocm_base + 0x8000);
    IOWR(NPU_CTRL_BASE, 0x25, 64);
    IOWR(NPU_CTRL_BASE, 0x20, 1); // accum_start pulse
}

void npu_wait_accum() {
    while ((IORD_32DIRECT(DDR_READ_ST_CSR_BASE, 0) & 0x01) != 0);
    while ((IORD(NPU_CTRL_BASE, REG_STATUS) & 0x01) != 0); 
    while ((IORD(NPU_CTRL_BASE, 0x21) & 1) == 0); 
}

void npu_load_bias(int32_t* bias_array_8_elements) {
    for (int f = 0; f < 8; f++) IOWR(NPU_CTRL_BASE, 0x200 + f, bias_array_8_elements[f]);
}

void npu_drain_to_ddr(uint32_t dest_ddr_offset) {
    IOWR(NPU_CTRL_BASE, REG_CTRL, 4); // seq_mode=2 (DRAIN)
    IOWR(NPU_CTRL_BASE, 0x23, dma_ocm_base + 0x8000);
    IOWR(NPU_CTRL_BASE, 0x25, 64);
    IOWR(NPU_CTRL_BASE, 0x20, 1); // accum_start pulse
    
    msgdma_write_stream_push(DDR_WRITE_ST_DESCRIPTOR_SLAVE_BASE, npu_ddr_base + dest_ddr_offset, 64);
    while ((IORD_32DIRECT(DDR_WRITE_ST_CSR_BASE, 0) & 0x01) != 0);
    while ((IORD(NPU_CTRL_BASE, 0x21) & 1) == 0);
    
    uint8_t* ptr = virt_ddr_base + dest_ddr_offset;
    volatile uint8_t dummy = ptr[0]; (void)dummy;
}

void npu_extract_32bit_ocm(int32_t* host_buffer, int32_t* bias_array_8_elements) {
    volatile uint8_t *ocm_dst_addr = virt_ocm_base + dma_ocm_base + 0x8000;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            uint32_t raw_val = IORD_32DIRECT(ocm_dst_addr, (r * 8 + c) * 4);
            host_buffer[r * 8 + c] = (int32_t)raw_val + bias_array_8_elements[c];
        }
    }
}
