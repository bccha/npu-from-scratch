/*
 * Auto-generated Linux Hardware Address Header
 * Generated on: 2026-02-24 16:18:29
 * This file maps Qsys/BSP components to Linux userspace /dev/mem offsets.
 */

#ifndef _HW_ADDRESSES_H_
#define _HW_ADDRESSES_H_

// ==========================================
// Bridge Offsets (Fixed Architecture)
// ==========================================
#define LWHPS2FPGA_BASE 0xFF200000
#define LWHPS2FPGA_SPAN 0x00200000

#define HPS_FPGA_RAM_BASE 0x20000000
#define HPS_FPGA_RAM_SPAN 0x01000000 // 16MB Window

// ==========================================
// Component Offsets (Extracted from Qsys)
// ==========================================
#define ADDRESS_SPAN_EXTENDER_0_CNTL_OFFSET 0x00000080
#define ADDRESS_SPAN_EXTENDER_0_WINDOWED_SLAVE_OFFSET 0x08000000
#define DDR_READ_ST_CSR_OFFSET 0x31000
#define DDR_READ_ST_DESC_OFFSET 0x31040
#define DDR_WRITE_ST_CSR_OFFSET 0x31020
#define DDR_WRITE_ST_DESC_OFFSET 0x31050
#define NPU_CTRL_OFFSET 0x00030000

#endif /* _HW_ADDRESSES_H_ */
