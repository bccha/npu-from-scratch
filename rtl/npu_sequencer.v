`timescale 1ns / 1ps

// --- Clean Sync Result FIFO (FWFT) ---
module npu_res_fifo #(
    parameter WIDTH = 256,
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

// --- Small 32-bit DMA Input FIFO ---
module npu_in_fifo (
    input  wire clk,
    input  wire rst_n,
    input  wire [31:0] din,
    input  wire din_valid,
    output wire din_ready,
    output wire [31:0] dout,
    input  wire rd_en,
    output wire empty
);
    reg [31:0] ram [0:7];
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
    parameter N = 8,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
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
    input  wire [31:0]  dma_data_in,
    input  wire         dma_data_in_valid,
    output wire         dma_data_in_ready,

    output reg  [31:0]  dma_data_out,
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
    wire [31:0] in_fifo_dout;
    wire in_fifo_empty;
    reg in_fifo_rd;
    npu_in_fifo inf (
        .clk(clk), .rst_n(rst_n),
        .din(dma_data_in), .din_valid(dma_data_in_valid), .din_ready(dma_data_in_ready),
        .dout(in_fifo_dout), .rd_en(in_fifo_rd), .empty(in_fifo_empty)
    );

    wire [N*ACC_WIDTH-1:0] res_fifo_dout;
    wire res_fifo_empty;
    wire [5:0] res_fifo_count;
    reg res_fifo_rd;
    npu_res_fifo outf (
        .clk(clk), .rst_n(rst_n),
        .din(core_y_out), .wr_en(&core_valid_out), .rd_en(res_fifo_rd),
        .dout(res_fifo_dout), .empty(res_fifo_empty), .count(res_fifo_count)
    );

    // --- 2. Feeder (2 Beats -> 1 Core Row) ---    
    reg [2:0] f_state;
    localparam F_IDLE=0, F_BEAT0=1, F_BEAT1_WAIT=2, F_BEAT1=3, F_BEAT0_WAIT=4, F_WAIT=5;
    reg [31:0] rows_fed;
    reg [31:0] saved_low;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_state <= F_IDLE; busy <= 0; rows_fed <= 0; core_valid_in <= 0; in_fifo_rd <= 0;
            core_load_weight <= 0; core_x_in <= 0;
        end else begin
            in_fifo_rd <= 0;
            case (f_state)
                F_IDLE: if (start) begin
                    busy <= 1; rows_fed <= 0;
                    f_state <= (mode == 2'd1) ? F_BEAT0 : F_IDLE; 
                end
                
                F_BEAT0: begin
                    core_valid_in <= 0;
                    // res_fifo_count < (32 - 6) to allow for in-flight rows in systolic pipe
                    if (!in_fifo_empty && (res_fifo_count < 26)) begin 
                        saved_low <= in_fifo_dout;
                        in_fifo_rd <= 1; // Beat 0 빼기
                        f_state <= F_BEAT1_WAIT;
                    end
                end
                
                F_BEAT1_WAIT: begin
                    f_state <= F_BEAT1; // 1클럭 버블 (dout 업데이트 대기)
                end
                
                F_BEAT1: begin
                    if (!in_fifo_empty) begin
                        core_x_in <= {in_fifo_dout, saved_low}; // 이제 진짜 Beat 1 도착
                        core_valid_in <= 8'hFF;
                        in_fifo_rd <= 1; // Beat 1 빼기
                        rows_fed <= rows_fed + 1;
                        if (rows_fed == total_rows - 1) f_state <= F_WAIT;
                        else f_state <= F_BEAT0_WAIT; // 👈 다음 행으로 가기 전에 기다림!
                    end
                end
                
                F_BEAT0_WAIT: begin
                    core_valid_in <= 0;
                    f_state <= F_BEAT0; // 1클럭 버블
                end
                
                F_WAIT: begin 
                    core_valid_in <= 0; 
                    if (done) begin busy <= 0; f_state <= F_IDLE; end 
                end
            endcase
        end
    end

    // --- 3. Drainer (1 Row -> 8 Beats) ---
    reg [2:0] d_state; // 상태가 4개가 되므로 3비트로 확장
    localparam D_IDLE=0, D_RUN=1, D_UNPACK=2, D_WAIT_FIFO=3; // 👈 WAIT 상태 추가!
    reg [31:0] rows_drained;
    reg [3:0]  sub_cnt;
    reg [7:0]  done_cnt;

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
                        dma_data_out_valid <= 1;
                        dma_data_out <= res_fifo_dout[0*32 +: 32]; // 첫 번째 비트
                        sub_cnt <= 1;
                        d_state <= D_UNPACK;
                    end else if (f_state == F_WAIT && rows_drained == total_rows) begin
                        if (done_cnt == 8'd100) done <= 1;
                        else done_cnt <= done_cnt + 1;
                    end
                end
                
                D_UNPACK: begin
                    // Ready가 들어왔을 때만 다음 데이터로 넘어감 (Avalon-ST 방식)
                    if (dma_data_out_ready && dma_data_out_valid) begin
                        if (sub_cnt == 8) begin
                            dma_data_out_valid <= 0; 
                            rows_drained <= rows_drained + 1;
                            res_fifo_rd <= 1; // FIFO에서 이전 행 버림
                            d_state <= D_WAIT_FIFO;
                        end else begin
                            dma_data_out_valid <= 1; 
                            dma_data_out <= res_fifo_dout[sub_cnt*32 +: 32];
                            sub_cnt <= sub_cnt + 1;
                        end
                    end
                end
                
                D_WAIT_FIFO: begin
                    d_state <= D_RUN; // 1클럭 버블 (FIFO dout 갱신 대기)
                end
            endcase
            if (done) d_state <= D_IDLE;
        end
    end

    // --- 4. Synchronous Trace Log ---
    always @(posedge clk) begin
        if (wr_en_internal) $display("[%0d] RTL_WR: row_data=%h", $time, core_y_out[31:0]);
        if (inf_rd_internal) $display("[%0d] RTL_FED: row=%0d data=%h", $time, rows_fed, core_x_in);
    end
    wire wr_en_internal = &core_valid_out;
    wire inf_rd_internal = (f_state == F_BEAT1 && !in_fifo_empty);

endmodule