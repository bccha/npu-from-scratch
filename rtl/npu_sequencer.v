`timescale 1ns / 1ps

// --- Clean Sync Result FIFO (FWFT) ---
module npu_res_fifo #(
    parameter WIDTH = 128,
    parameter DEPTH = 32,
    parameter ADDR_W = 5
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [WIDTH-1:0] din,
    input  wire wr_en,
    input  wire rd_en,
    output wire [WIDTH-1:0] dout,
    output wire full,
    output wire empty,
    output reg  [ADDR_W:0] count
);
    reg [WIDTH-1:0] ram [0:DEPTH-1];
    reg [ADDR_W-1:0] wr_ptr, rd_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);
    assign dout  = ram[rd_ptr]; 

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) ram[i] = 0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0; rd_ptr <= 0; count <= 0;
        end else begin
            if (wr_en && !full) begin
                ram[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ; 
            endcase
        end
    end
endmodule

module npu_in_fifo #(
    parameter WIDTH = 64
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [WIDTH-1:0] din,
    input  wire din_valid,
    output wire din_ready,
    output wire [WIDTH-1:0] dout,
    input  wire rd_en,
    output wire empty
);
    reg [WIDTH-1:0] ram [0:7];
    reg [2:0] wr_ptr, rd_ptr;
    reg [3:0] count;
    assign din_ready = (count < 8);
    assign empty = (count == 0);
    assign dout = ram[rd_ptr];

    integer i;
    initial begin
        for (i = 0; i < 8; i = i + 1) ram[i] = 0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wr_ptr <= 0; rd_ptr <= 0; count <= 0; end
        else begin
            if (din_valid && din_ready) begin
                ram[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
            case ({din_valid && din_ready, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ;
            endcase
        end
    end
endmodule

// --- Main NPU Sequencer (Sync-First) ---
module npu_sequencer #(
    parameter N = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter AXI_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Control/Status
    input  wire         start,           
    input  wire [1:0]   mode,            // 0: Weight Load, 1: Execution
    input  wire [31:0]  total_rows,      
    output reg          busy,
    output reg          done,

    // DMA Streaming Interface
    input  wire [AXI_WIDTH-1:0]  dma_data_in,
    input  wire         dma_data_in_valid,
    output wire         dma_data_in_ready,

    output reg  [AXI_WIDTH-1:0]  dma_data_out,
    output reg         dma_data_out_valid,
    input  wire        dma_data_out_ready,

    // Systolic Core Interface
    output reg                      core_load_weight,
    output reg  [N-1:0]             core_valid_in,
    output reg  [N*DATA_WIDTH-1:0]  core_x_in,
    output wire [N*ACC_WIDTH-1:0]   core_y_in,
    input  wire [N*ACC_WIDTH-1:0]   core_y_out,
    input  wire [N-1:0]             core_valid_out
);

    assign core_y_in = {N*ACC_WIDTH{1'b0}};

    // --- 1. Internal Buffers ---
    wire [AXI_WIDTH-1:0] in_fifo_dout;
    wire in_fifo_empty;
    wire in_fifo_rd = (f_state == F_BEAT) && !in_fifo_empty && (res_fifo_count < 26);
    // AXI_WIDTH wide in_fifo
    npu_in_fifo #(.WIDTH(AXI_WIDTH)) inf (
        .clk(clk), .rst_n(rst_n),
        .din(dma_data_in), .din_valid(dma_data_in_valid), .din_ready(dma_data_in_ready),
        .dout(in_fifo_dout), .rd_en(in_fifo_rd), .empty(in_fifo_empty)
    );

    wire [255:0] res_fifo_dout; // N*ACC_WIDTH = 8*32 = 256
    wire res_fifo_empty;
    wire [5:0] res_fifo_count;
    reg res_fifo_rd;
    npu_res_fifo #(
        .WIDTH(N*ACC_WIDTH),
        .DEPTH(32),
        .ADDR_W(5)
    ) outf (
        .clk(clk), .rst_n(rst_n),
        .din(core_y_out), .wr_en(&core_valid_out), .rd_en(res_fifo_rd),
        .dout(res_fifo_dout), .empty(res_fifo_empty), .count(res_fifo_count)
    );

    // --- 2. Feeder (1 Beat -> 1 Core Row for N=8) ---
    // With 64-bit data_in, 1 cycle provides all 8 elements.
    reg [1:0] f_state;
    localparam F_IDLE=0, F_BEAT=1, F_WAIT=2;
    reg [31:0] rows_fed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_state <= F_IDLE; busy <= 0; rows_fed <= 0; core_valid_in <= 0;
            core_load_weight <= 0; core_x_in <= 0;
        end else begin
            case (f_state)
                F_IDLE: if (start) begin
                    busy <= 1; rows_fed <= 0;
                    f_state <= F_BEAT; 
                end
                
                F_BEAT: begin
                    core_valid_in <= 0;
                    core_load_weight <= (mode == 2'd0);
                    if (!in_fifo_empty && (res_fifo_count < 26)) begin 
                        core_x_in <= in_fifo_dout; // Directly assign 64-bit to core_x_in
                        core_valid_in <= {N{1'b1}};
                        rows_fed <= rows_fed + 1;
                        if (rows_fed == total_rows - 1) f_state <= F_WAIT;
                    end
                end
                
                F_WAIT: begin 
                    core_valid_in <= 0; 
                    core_load_weight <= 0; // IMPORTANT: Clear load_weight when done feeding
                    if (done) begin busy <= 0; f_state <= F_IDLE; end 
                end
            endcase
        end
    end

    // --- 3. Drainer (1 Row -> N Beats) ---
    // With generic mapping, we fetch N*ACC_WIDTH bits from res_fifo_dout.
    // N*ACC_WIDTH / AXI_WIDTH words out per Row.
    reg [2:0] d_state;
    localparam D_IDLE=0, D_RUN=1, D_UNPACK=2, D_WAIT_FIFO=3; 
    reg [31:0] rows_drained;
    reg [3:0]  sub_cnt; // 0 to 3 for 4 words per row
    reg [7:0]  done_cnt;

    // Calculate number of outputs needed for N row
    // If N=8, ACC=32, total bits=256 bits. 256/64 = 4 words
    localparam OUT_BEATS_PER_ROW = (N*ACC_WIDTH) / AXI_WIDTH; 

    // Padding to avoid Synthesizer index out of bounds error for N=8, AXI=64
    // where sub_cnt runs up to 3 but the case statement covers up to 7
    wire [1023:0] res_fifo_dout_padded = { {(1024 - N*ACC_WIDTH){1'b0}}, res_fifo_dout };

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d_state <= D_IDLE; done <= 0; rows_drained <= 0; 
            dma_data_out_valid <= 0; dma_data_out <= 0; sub_cnt <= 0;
            res_fifo_rd <= 0; done_cnt <= 0;
        end else begin
            res_fifo_rd <= 0; // 기본값
            case (d_state)
                D_IDLE: begin 
                    done <= 0; 
                    if (start) begin rows_drained <= 0; d_state <= D_RUN; end 
                end
                
                D_RUN: begin
                    if (!res_fifo_empty) begin
                        sub_cnt <= 0;
                        dma_data_out_valid <= 1;
                        dma_data_out <= res_fifo_dout_padded[0*AXI_WIDTH +: AXI_WIDTH];
                        d_state <= D_UNPACK;
                    end else if (f_state == F_WAIT && (mode == 2'd0 || rows_drained == total_rows)) begin
                        if (done_cnt == 8'd100) done <= 1;
                        else done_cnt <= done_cnt + 1;
                    end
                end
                
                D_UNPACK: begin
                    // Data (dma_data_out) is ALREADY presented for sub_cnt
                    if (dma_data_out_ready && dma_data_out_valid) begin
                        if (sub_cnt == OUT_BEATS_PER_ROW - 1) begin
                            // Last beat of the row accepted
                            dma_data_out_valid <= 0;
                            rows_drained <= rows_drained + 1;
                            res_fifo_rd <= 1;
                            d_state <= D_WAIT_FIFO;
                        end else begin
                            // Move to next beat
                            sub_cnt <= sub_cnt + 1;
                            case (sub_cnt + 4'd1)
                                4'd0: dma_data_out <= res_fifo_dout_padded[0*AXI_WIDTH +: AXI_WIDTH];
                                4'd1: dma_data_out <= res_fifo_dout_padded[1*AXI_WIDTH +: AXI_WIDTH];
                                4'd2: dma_data_out <= res_fifo_dout_padded[2*AXI_WIDTH +: AXI_WIDTH];
                                4'd3: dma_data_out <= res_fifo_dout_padded[3*AXI_WIDTH +: AXI_WIDTH];
                                4'd4: dma_data_out <= res_fifo_dout_padded[4*AXI_WIDTH +: AXI_WIDTH];
                                4'd5: dma_data_out <= res_fifo_dout_padded[5*AXI_WIDTH +: AXI_WIDTH];
                                4'd6: dma_data_out <= res_fifo_dout_padded[6*AXI_WIDTH +: AXI_WIDTH];
                                4'd7: dma_data_out <= res_fifo_dout_padded[7*AXI_WIDTH +: AXI_WIDTH];
                                default: dma_data_out <= 0;
                            endcase
                            dma_data_out_valid <= 1;
                        end
                    end
                end
                
                D_WAIT_FIFO: begin
                    dma_data_out_valid <= 0;
                    d_state <= D_RUN; // Bubble for FIFO dout update
                end
            endcase
            if (done) d_state <= D_IDLE;
        end
    end

    // --- 4. Synchronous Trace Log ---
    always @(posedge clk) begin
        if (wr_en_internal) $display("[%0d] RTL_WR: row_data=%h", $time, core_y_out[N*ACC_WIDTH-1:0]);
        // To avoid unresolvable reference in some simulators, logic uses raw references
    end
    wire wr_en_internal = &core_valid_out;

endmodule