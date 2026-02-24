`timescale 1ns / 1ps

module npu_dma #(
    parameter AXI_WIDTH = 64
)(
    input  wire        clk,
    input  wire        rst_n,

    // Control Interface (from npu_ctrl)
    input  wire [31:0] rd_addr,
    input  wire [31:0] rd_len,
    input  wire        rd_start_pulse,
    input  wire [31:0] wr_addr,
    input  wire [31:0] wr_len,
    input  wire        wr_start_pulse,
    
    // Status Interface (to npu_ctrl)
    output reg         rd_busy,
    output reg         rd_done,
    output reg         wr_busy,
    output reg         wr_done,

    // Avalon-MM Read Master Interface
    input  wire        rd_m_waitrequest,
    input  wire [AXI_WIDTH-1:0] rd_m_readdata,
    input  wire        rd_m_readdatavalid,
    output reg  [4:0]  rd_m_burstcount,
    output reg  [31:0] rd_m_address,
    output reg         rd_m_read,

    // Avalon-MM Write Master Interface
    input  wire        wr_m_waitrequest,
    output reg  [4:0]  wr_m_burstcount,
    output reg  [31:0] wr_m_address,
    output reg         wr_m_write,
    output wire [AXI_WIDTH-1:0] wr_m_writedata,

// Internal Streaming Interface
    output wire [AXI_WIDTH-1:0] data_to_npu,
    output wire        data_to_npu_valid,
    input  wire        data_to_npu_ready, // 👈 ADDED
    input  wire [AXI_WIDTH-1:0] data_from_npu,
    input  wire        data_from_npu_valid,
    output wire        data_from_npu_ready
);

    // ----------------------------------------------------
    // Read FIFO (Memory -> in_fifo -> NPU)
    // ----------------------------------------------------
    localparam FIFO_DEPTH = 32;
    localparam ADDR_WIDTH = 5;
    reg [AXI_WIDTH-1:0] in_fifo [0:FIFO_DEPTH-1];
    reg [ADDR_WIDTH-1:0] in_fifo_wr_ptr;
    reg [ADDR_WIDTH-1:0] in_fifo_rd_ptr;
    reg [ADDR_WIDTH:0]   in_fifo_count;
    
    wire in_fifo_full  = (in_fifo_count == FIFO_DEPTH);
    wire in_fifo_empty = (in_fifo_count == 0);

    // ----------------------------------------------------
    // Write FIFO (NPU -> out_fifo -> Memory)
    // ----------------------------------------------------
    reg [AXI_WIDTH-1:0] out_fifo [0:FIFO_DEPTH-1];
    reg [ADDR_WIDTH-1:0] out_fifo_wr_ptr;
    reg [ADDR_WIDTH-1:0] out_fifo_rd_ptr;
    reg [ADDR_WIDTH:0]   out_fifo_count;

    wire out_fifo_full  = (out_fifo_count == FIFO_DEPTH);
    wire out_fifo_empty = (out_fifo_count == 0);


    // --- Read Master Variables ---
    reg [31:0] rd_pending_beats;
    wire [ADDR_WIDTH:0] in_fifo_free_space = FIFO_DEPTH[ADDR_WIDTH:0] - in_fifo_count - rd_pending_beats[ADDR_WIDTH:0];
    
    localparam RD_IDLE   = 2'd0;
    localparam RD_BURST  = 2'd1;
    localparam RD_WAIT   = 2'd2;
    reg [1:0] rd_state;
    reg [31:0] rd_rem_len;
    reg [4:0]  current_rd_burst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state        <= RD_IDLE;
            rd_m_read       <= 1'b0;
            rd_m_address    <= 32'd0;
            rd_m_burstcount <= 5'd0;
            rd_busy         <= 1'b0;
            rd_done         <= 1'b0;
            rd_rem_len      <= 32'd0;
            rd_pending_beats <= 32'd0;
            current_rd_burst <= 5'd0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    if (rd_start_pulse) begin
                        rd_busy    <= 1'b1;
                        rd_done    <= 1'b0;
                        rd_rem_len <= rd_len;
                        rd_m_address <= rd_addr;
                        rd_pending_beats <= 32'd0;
                        rd_state   <= RD_BURST;
                    end
                end

                RD_BURST: begin
                    if (rd_rem_len == 0) begin
                        if (rd_pending_beats == 0) begin
                            rd_busy <= 1'b0;
                            rd_done <= 1'b1;
                            rd_state <= RD_IDLE;
                        end
                    end else if (rd_rem_len > 0) begin
                        // Decide burst size (max 16)
                        if (in_fifo_free_space >= 16 || (rd_rem_len < 16 && in_fifo_free_space >= rd_rem_len[ADDR_WIDTH:0])) begin
                            current_rd_burst <= (rd_rem_len >= 16) ? 5'd16 : rd_rem_len[4:0];
                            rd_m_read        <= 1'b1;
                            rd_m_burstcount  <= (rd_rem_len >= 16) ? 5'd16 : rd_rem_len[4:0];
                            rd_state         <= RD_WAIT;
                        end
                    end
                end

                RD_WAIT: begin
                    if (rd_m_waitrequest == 1'b0) begin
                        rd_m_read       <= 1'b0;
                        rd_rem_len      <= rd_rem_len - {22'd0, rd_m_burstcount}; // Note: rd_rem_len is now in 64-bit words logic if user provides length in 64-bit unit, or addr calculation needs *8
                        rd_m_address    <= rd_m_address + ({22'd0, rd_m_burstcount} * (AXI_WIDTH/8)); // AXI_WIDTH/8 bytes per burst word
                        rd_state        <= RD_BURST;
                    end
                end
            endcase

            // Robust rd_pending_beats logic: Use case to avoid X-propagation
            case ({ (rd_state == RD_WAIT && rd_m_waitrequest == 1'b0), (rd_m_readdatavalid == 1'b1) })
                2'b10: rd_pending_beats <= rd_pending_beats + {22'd0, rd_m_burstcount};
                2'b01: rd_pending_beats <= rd_pending_beats - 32'd1;
                2'b11: rd_pending_beats <= rd_pending_beats + {22'd0, rd_m_burstcount} - 32'd1;
                default: ;
            endcase
        end
    end

    // ----------------------------------------------------
    // IN_FIFO Routing & Control
    // ----------------------------------------------------
    wire in_fifo_push = rd_m_readdatavalid;
    assign data_to_npu_valid = !in_fifo_empty;
    wire in_fifo_pop = data_to_npu_valid && data_to_npu_ready;
    assign data_to_npu = in_fifo[in_fifo_rd_ptr];

    always @(posedge clk) begin
        if (in_fifo_push) begin
            in_fifo[in_fifo_wr_ptr] <= rd_m_readdata;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_fifo_wr_ptr <= 0; in_fifo_rd_ptr <= 0; in_fifo_count  <= 0;
        end else if (rd_start_pulse) begin
            in_fifo_wr_ptr <= 0; in_fifo_rd_ptr <= 0; in_fifo_count  <= 0;
        end else begin
            case ({in_fifo_push, in_fifo_pop})
                2'b10: begin in_fifo_wr_ptr <= in_fifo_wr_ptr + 1'b1; in_fifo_count <= in_fifo_count + 1'b1; end
                2'b01: begin in_fifo_rd_ptr <= in_fifo_rd_ptr + 1'b1; in_fifo_count <= in_fifo_count - 1'b1; end
                2'b11: begin in_fifo_wr_ptr <= in_fifo_wr_ptr + 1'b1; in_fifo_rd_ptr <= in_fifo_rd_ptr + 1'b1; end
                default: ;
            endcase
        end
    end

    // ----------------------------------------------------
    // OUT_FIFO Routing & Control
    // ----------------------------------------------------
    assign data_from_npu_ready = !out_fifo_full;
    wire out_fifo_push = data_from_npu_valid && data_from_npu_ready;
    wire out_fifo_pop  = (wr_m_write && !wr_m_waitrequest);
    assign wr_m_writedata = out_fifo[out_fifo_rd_ptr];
    
    always @(posedge clk) begin
        if (out_fifo_push) begin
            out_fifo[out_fifo_wr_ptr] <= data_from_npu;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_fifo_wr_ptr <= 0; out_fifo_rd_ptr <= 0; out_fifo_count <= 0;
        end else if (wr_start_pulse) begin
            out_fifo_wr_ptr <= 0; out_fifo_rd_ptr <= 0; out_fifo_count <= 0;
        end else begin
            case ({out_fifo_push, out_fifo_pop})
                2'b10: begin out_fifo_wr_ptr <= out_fifo_wr_ptr + 1'b1; out_fifo_count <= out_fifo_count + 1'b1; end
                2'b01: begin out_fifo_rd_ptr <= out_fifo_rd_ptr + 1'b1; out_fifo_count <= out_fifo_count - 1'b1; end
                2'b11: begin out_fifo_wr_ptr <= out_fifo_wr_ptr + 1'b1; out_fifo_rd_ptr <= out_fifo_rd_ptr + 1'b1; end
                default: ;
            endcase
        end
    end



    // Write Master FSM
    localparam WR_IDLE   = 2'd0;
    localparam WR_BURST  = 2'd1;
    localparam WR_DATA   = 2'd2;
    reg [1:0] wr_state;
    reg [31:0] wr_rem_len;
    reg [4:0]  wr_burst_rem;
    reg [4:0]  wr_current_burst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state        <= WR_IDLE;
            wr_m_write      <= 1'b0;
            wr_m_address    <= 32'd0;
            wr_m_burstcount <= 5'd0;
            wr_busy         <= 1'b0;
            wr_done         <= 1'b1; // Default to idle-done
            wr_rem_len      <= 32'd0;
            wr_burst_rem    <= 5'd0;
            wr_current_burst <= 5'd0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    if (wr_start_pulse) begin
                        wr_busy    <= 1'b1;
                        wr_done    <= 1'b0;
                        wr_rem_len <= wr_len;
                        wr_m_address <= wr_addr;
                        wr_state   <= WR_BURST;
                    end
                end

                WR_BURST: begin
                    if (wr_rem_len == 0) begin
                        wr_busy  <= 1'b0;
                        wr_done  <= 1'b1;
                        wr_state <= WR_IDLE;
                    end else begin
                        // Decide burst size (max 16)
                        if (out_fifo_count > 0 && (out_fifo_count >= 16 || (wr_rem_len < 16 && out_fifo_count >= wr_rem_len[ADDR_WIDTH:0]))) begin
                            wr_current_burst = (wr_rem_len >= 16 && out_fifo_count >= 16) ? 5'd16 : wr_rem_len[4:0];
                            wr_m_write      <= 1'b1;
                            wr_m_burstcount <= wr_current_burst;
                            wr_burst_rem    <= wr_current_burst;
                            wr_state        <= WR_DATA;
                        end
                    end
                end

                WR_DATA: begin
                    if (wr_m_waitrequest == 1'b0) begin
                        if (wr_burst_rem == 1) begin
                            wr_m_write   <= 1'b0;
                            wr_rem_len    <= wr_rem_len - {22'd0, wr_m_burstcount};
                            wr_m_address  <= wr_m_address + ({22'd0, wr_m_burstcount} * (AXI_WIDTH/8)); // AXI_WIDTH/8 bytes
                            wr_state      <= WR_BURST;
                        end else begin
                            wr_burst_rem <= wr_burst_rem - 1'b1;
                            // wr_m_writedata is updated automatically in the FIFO control block
                        end
                    end
                end
            endcase
        end
    end

endmodule
