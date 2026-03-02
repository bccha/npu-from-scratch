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
  // 1. Issue Reset Command (Bit 1 of Control Register)
  IOWR_32DIRECT(csr_base, 0x04, (1 << 1));

  // 2. Wait for Hardware to Finish Resetting (Poll Status Bit 6)
  while (IORD_32DIRECT(csr_base, 0x00) & (1 << 6)) {
    // Loop actively while resetting
  }

  // 3. Clear existing status bits (W1C bits)
  IOWR_32DIRECT(csr_base, 0x00, 0xFFFFFFFF);

  // 4. Enable dispatcher globally
  // Disable Interrupts, Clear Stop Dispatcher, Clear Reset.
  IOWR_32DIRECT(csr_base, 0x04, 0x00000000);
}

void msgdma_read_stream_push(alt_u32 descriptor_base, alt_u32 src_addr,
                             alt_u32 length) {
  IOWR_32DIRECT(descriptor_base, 0x00, src_addr);
  IOWR_32DIRECT(descriptor_base, 0x04, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x08, length);

  // Standard Descriptor Control (Altera MSGDMA):
  // Bit 31: GO           (0x80000000)
  // Bit 9:  Generate EOP (0x00000200)
  // Bit 8:  Generate SOP (0x00000100)
  // 0x80000000 | 0x00000200 | 0x00000100 = 0x80000300
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000300);
}

void msgdma_write_stream_push(alt_u32 descriptor_base, alt_u32 dst_addr,
                              alt_u32 length) {
  IOWR_32DIRECT(descriptor_base, 0x00, 0x00000000);
  IOWR_32DIRECT(descriptor_base, 0x04, dst_addr);
  IOWR_32DIRECT(descriptor_base, 0x08, length);

  // Standard Descriptor Control (Altera MSGDMA):
  // Bit 31: GO           (0x80000000)
  // We use exactly 'length' bytes, avoiding END_ON_EOP to ensure robustness
  // against spurious EOPs from the Avalon-ST sink which cause early
  // termination.
  IOWR_32DIRECT(descriptor_base, 0x0C, 0x80000000);
}
