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

    // Interface to NPU PE Array (Bufferless)
    // TODO: Connect these to MAC and accumulator
    output [63:0] pe_din,
    output        pe_valid_in,
    input         pe_ready_in, // (e.g., pipeline is ready)

    input  [255:0] pe_dout,
    input         pe_valid_out,
    output        pe_ready_out 
);

    // =========================================================================
    // Sink Control (Memory -> NPU)
    // =========================================================================
    // Pass data directly to PE if valid. 
    // Backpressure: If PE is not ready, we drop st_sink_ready to 0.
    assign pe_din = st_sink_data;
    assign pe_valid_in = st_sink_valid;
    assign st_sink_ready = pe_ready_in;

    // TODO: Handling SOP/EOP to reset MAC accumulators or define matrix boundaries
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset logic
        end else if (st_sink_valid && st_sink_ready) begin
            if (st_sink_startofpacket) begin
                // Start of a new 8x8 matrix or batch
            end
            if (st_sink_endofpacket) begin
                // End of the 8x8 matrix
            end
        end
    end

    // =========================================================================
    // Source Control (NPU -> Memory) : 256-bit to 64-bit Serializer
    // =========================================================================
    // The PE array (8x8) outputs 256 bits (32 bits x 8 elements) at once.
    // We need to serialize this over 4 cycles (64 bits per cycle) to the MSGDMA.
    // BUT the systolic array outputs a new 256-bit row every cycle once it starts!
    // So we need an output FIFO to hold the rows while they are serialized.

    // 8-depth 256-bit FIFO
    reg [255:0] out_fifo [0:7];
    reg [2:0]   fifo_wr_ptr;
    reg [2:0]   fifo_rd_ptr;
    reg [3:0]   fifo_count;

    wire fifo_full  = (fifo_count >= 4'd8);
    wire fifo_empty = (fifo_count == 4'd0);

    // FIFO Write Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wr_ptr <= 3'd0;
        end else if (pe_valid_out && !fifo_full) begin
            out_fifo[fifo_wr_ptr] <= pe_dout;
            fifo_wr_ptr <= fifo_wr_ptr + 1'b1;
        end
    end

    reg [255:0] tx_shift_reg;
    reg [2:0]   tx_count;    // 0 to 4
    reg         tx_active;   // 1 when actively transmitting 4 flits
    reg         tx_sending_flit; // High when valid is true
    reg [31:0]  tx_row_count;    // Number of 256-bit rows transmitted in current sequence

    // FIFO Read & Serializer Logic
    wire fifo_push = pe_valid_out && !fifo_full;
    wire fifo_pop  = (!tx_active && !fifo_empty) || 
                     (tx_active && st_source_ready && tx_sending_flit && (tx_count == 3'd3) && !fifo_empty);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd_ptr  <= 3'd0;
            fifo_count   <= 4'd0;
            tx_shift_reg <= 256'd0;
            tx_count     <= 3'd0;
            tx_active    <= 1'b0;
            tx_sending_flit <= 1'b0;
            tx_row_count <= 32'd0;
        end else begin
            // Track FIFO Count
            if (fifo_push && !fifo_pop) begin
                fifo_count <= fifo_count + 1'b1;
            end else if (!fifo_push && fifo_pop) begin
                fifo_count <= fifo_count - 1'b1;
            end // else both or neither -> count stays same

            if (!tx_active) begin
                // Idle state: Wait for FIFO to have data
                if (!fifo_empty) begin
                    tx_shift_reg <= out_fifo[fifo_rd_ptr];
                    fifo_rd_ptr  <= fifo_rd_ptr + 1'b1;
                    tx_count     <= 3'd0;
                    tx_active    <= 1'b1;
                    tx_sending_flit <= 1'b1; // Start sending first flit immediately if ready
                end
            end else begin
                // Active Transmitting state
                if (st_source_ready && tx_sending_flit) begin
                    if (tx_count == 3'd3) begin
                        // Transmitted all 4 flits (4 x 64 = 256 bits)
                        
                        // Update Sequence Row Tracker
                        if ((seq_total_rows > 0) && (tx_row_count == seq_total_rows - 1)) begin
                            tx_row_count <= 32'd0; // Sequence complete, reset tracker
                        end else begin
                            tx_row_count <= tx_row_count + 1'b1;
                        end

                        // If FIFO still has data, pop next immediately
                        if (!fifo_empty) begin
                            tx_shift_reg <= out_fifo[fifo_rd_ptr];
                            fifo_rd_ptr  <= fifo_rd_ptr + 1'b1;
                            tx_count     <= 3'd0;
                            // fifo_count is handled by the overall tracker above
                        end else begin
                            tx_active <= 1'b0;
                            tx_sending_flit <= 1'b0;
                        end
                    end else begin
                        // Shift data and increment counter
                        tx_shift_reg <= {64'd0, tx_shift_reg[255:64]};
                        tx_count <= tx_count + 1'b1;
                    end
                end
            end
        end
    end

    // The PE array doesn't have a stall input in the original design.
    // If the FIFO gets full, we have to backpressure before the PE starts, or PE drops data.
    assign pe_ready_out = !fifo_full;

    // Drive Avalon-ST Source Signals
    assign st_source_data  = tx_shift_reg[63:0]; // Always output the bottom 64 bits
    assign st_source_valid = tx_sending_flit;
    
    // Packet boundaries: Useful for the DMA to know when a full 256-bit row is done.
    // SOP is asserted on the VERY first 64-bit flit of the entire sequence.
    assign st_source_startofpacket = (tx_sending_flit && tx_count == 3'd0 && tx_row_count == 32'd0);
    // EOP is asserted on the VERY last 64-bit flit of the entire sequence (or if seq_total_rows is 0, we don't send EOP)
    assign st_source_endofpacket   = (seq_total_rows > 0) && (tx_sending_flit && tx_count == 3'd3 && tx_row_count == (seq_total_rows - 1));
    assign st_source_empty         = 3'b000; // All 8 bytes in the 64-bit flit are active.

endmodule
