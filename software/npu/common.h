#ifndef COMMON_H_
#define COMMON_H_

#include "altera_avalon_jtag_uart_regs.h"
#include "altera_avalon_timer_regs.h"
#include "io.h"
#include "sys/alt_alarm.h"
#include "sys/alt_cache.h"
#include <stdio.h>
#include <system.h>

// ============================================================================
// Global Configuration
// ============================================================================

// Nios II Data Cache Bypass Mask (Bit 31)
#define CACHE_BYPASS_MASK 0x80000000

// DDR3 Window Base Address
#define DDR3_WINDOW_BASE                                                       \
  (ADDRESS_SPAN_EXTENDER_0_WINDOWED_SLAVE_BASE | CACHE_BYPASS_MASK)

// ============================================================================
// Helper Functions
// ============================================================================

// Blocking version: Waits until a character is received
char get_char_polled();

// Non-blocking (Async) version: Returns char if available, else returns 0
char get_char_async();

// Returns current physical cycles (50MHz) since boot
unsigned long long get_total_cycles();

// MSGDMA Helpers
void msgdma_init(alt_u32 csr_base);
void msgdma_read_stream_push(alt_u32 descriptor_base, alt_u32 src_addr,
                             alt_u32 length);
void msgdma_write_stream_push(alt_u32 descriptor_base, alt_u32 dst_addr,
                              alt_u32 length);

#endif /* COMMON_H_ */
