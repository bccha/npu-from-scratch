`timescale 1ns / 1ps

module npu_stream_ctrl (
    input clk,
    input rst_n,

    // Avalon-ST Sink (from Memory/MSGDMA Read Master)
    input  [63:0] st_sink_data,
    input         st_sink_valid,
    output        st_sink_ready,
    input         st_sink_startofpacket,
    input         st_sink_endofpacket,
    input  [2:0]  st_sink_empty,

    // Avalon-ST Source (to Memory/MSGDMA Write Master)
    output [63:0] st_source_data,
    output        st_source_valid,
    input         st_source_ready,
    output        st_source_startofpacket,
    output        st_source_endofpacket,
    output [2:0]  st_source_empty,

    // NPU Global Configuration
    input  [31:0] seq_total_rows,

    // Interface to NPU PE Array (Upstream)
    output [63:0] pe_din,
    output        pe_valid_in,
    input         pe_ready_in, // Array is ready to accept

    // Interface from Post-Processor (Downstream)
    input  [7:0]  pp_data,
    input         pp_valid,
    output        pp_ready 
);

    // =========================================================================
    // Sink Control (Memory -> NPU)
    // =========================================================================
    assign pe_din = st_sink_data;
    assign pe_valid_in = st_sink_valid;
    assign st_sink_ready = pe_ready_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset logic
        end else if (st_sink_valid && st_sink_ready) begin
            if (st_sink_startofpacket) begin
                // Start of a new matrix
            end
            if (st_sink_endofpacket) begin
                // End of matrix
            end
        end
    end

    // =========================================================================
    // Source Control (Post-Processor -> Memory) : 8-bit to 64-bit Packer
    // =========================================================================
    // The Post-Processor outputs 8 bits per cycle.
    // We need to pack 8 outputs into a single 64-bit flit to send to MSGDMA.
    
    reg [63:0] pack_reg;
    reg [2:0]  pack_cnt;
    reg        pack_full;
    reg [31:0] tx_row_count;

    // We can accept data if the pack_reg is not full OR if st_source_ready allows us to flush it
    wire can_accept = !pack_full || st_source_ready;
    assign pp_ready = can_accept;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pack_cnt     <= 3'd0;
            pack_full    <= 1'b0;
            tx_row_count <= 32'd0;
            pack_reg     <= 64'd0;
        end else begin
            // Handshake with DMA
            if (pack_full && st_source_ready) begin
                pack_full <= 1'b0; 
                // Update Sequence Row Tracker
                if ((seq_total_rows > 0) && (tx_row_count == seq_total_rows - 1)) begin
                    tx_row_count <= 32'd0;
                end else begin
                    tx_row_count <= tx_row_count + 1'b1;
                end
            end

            // Accept from Post-Processor
            if (pp_valid && can_accept) begin
                pack_reg <= {pp_data, pack_reg[63:8]}; // Shift down (Little-Endian packing)
                if (pack_cnt == 3'd7) begin
                    pack_cnt  <= 3'd0;
                    pack_full <= 1'b1; // Trigger DMA write
                end else begin
                    pack_cnt  <= pack_cnt + 1'b1;
                end
            end
        end
    end

    // Drive Avalon-ST Source Signals
    assign st_source_data  = pack_reg;
    // We assert valid only when we have a full 64-bit word packed
    assign st_source_valid = pack_full;
    
    assign st_source_startofpacket = (pack_full && tx_row_count == 32'd0);
    assign st_source_endofpacket   = (seq_total_rows > 0) && (pack_full && tx_row_count == (seq_total_rows - 1));
    assign st_source_empty         = 3'b000;

endmodule
