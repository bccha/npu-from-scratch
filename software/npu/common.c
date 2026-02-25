#include "common.h"
#include <alt_types.h>
#include <io.h>
char get_char_polled() {
  unsigned int data;
  while (1) {
    data = IORD_ALTERA_AVALON_JTAG_UART_DATA(JTAG_UART_BASE);
    if (data & ALTERA_AVALON_JTAG_UART_DATA_RVALID_MSK) {
      return (char)(data & ALTERA_AVALON_JTAG_UART_DATA_DATA_MSK);
    }
  }
}

char get_char_async() {
  unsigned int data = IORD_ALTERA_AVALON_JTAG_UART_DATA(JTAG_UART_BASE);
  if (data & ALTERA_AVALON_JTAG_UART_DATA_RVALID_MSK) {
    return (char)(data & ALTERA_AVALON_JTAG_UART_DATA_DATA_MSK);
  }
  return 0;
}

unsigned long long get_total_cycles() {
  unsigned int t1, t2, snap;
  do {
    t1 = alt_nticks();
    IOWR_ALTERA_AVALON_TIMER_SNAPL(TIMER_0_BASE, 0);
    unsigned int low = IORD_ALTERA_AVALON_TIMER_SNAPL(TIMER_0_BASE);
    unsigned int high = IORD_ALTERA_AVALON_TIMER_SNAPH(TIMER_0_BASE);
    snap = (high << 16) | low;
    t2 = alt_nticks();
  } while (t1 != t2);

  unsigned long long cycles = (unsigned long long)t1 * 50000;
  cycles += (49999 - snap);
  return cycles;
}

void msgdma_init(alt_u32 csr_base) {
  // Clear status register (W1C bits)
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);
  // Enable dispatcher globally (Bit 4 = Global Interrupt Enable, Bit 5 = Stop
  // Dispatcher) To start, we just need to ensure bits 1 (Reset) and 5 (Stop)
  // are 0, and bit 4 (Interrupt) is 0 since we poll. Actually, for generic
  // Altera MSGDMA, writing 0 to the Control register clears the Stop bit.
  IOWR_32DIRECT(csr_base, 0x04,
                0x00000000); // Clear Stop Dispatcher, Disable Interrupts.
}

void msgdma_read_stream_push(alt_u32 descriptor_base, alt_u32 src_addr,
                             alt_u32 length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);

  // Standard Descriptor Control:
  // Bit 31: GO
  // Bit 27: Generate EOP
  // Bit 26: Generate SOP
  // 0x8C000000 sets GO, Gen-EOP, Gen-SOP.
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x8C000000);
}

void msgdma_write_stream_push(alt_u32 descriptor_base, alt_u32 dst_addr,
                              alt_u32 length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);

  // Bit 31: GO
  // Bit 23: End on Length (1)
  // Bit 22: End on EOP (1)
  // 0x80000000 | 0x00800000 | 0x00400000 = 0x80C00000
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80C00000);
}
