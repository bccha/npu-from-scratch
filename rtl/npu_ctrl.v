`timescale 1ns / 1ps

module npu_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // Unified Register Interface
    input  wire [3:0]  address,
    input  wire        write,
    input  wire [31:0] writedata,
    input  wire        read,
    output reg  [31:0] readdata,
    output reg         readdatavalid,

    // NPU Global Control (Sequencer)
    output reg         seq_start,
    output reg  [1:0]  seq_mode,       // 0: Weight Load, 1: Execution
    output reg  [31:0] seq_total_rows,
    input  wire        seq_busy,
    input  wire        seq_done,
    output reg         weight_latch_en,

    // DMA Control
    output reg  [31:0] dma_rd_addr,
    output reg  [31:0] dma_rd_len,
    output reg         dma_rd_start,
    output reg  [31:0] dma_wr_addr,
    output reg  [31:0] dma_wr_len,
    output reg         dma_wr_start,
    input  wire        dma_rd_busy,
    input  wire        dma_rd_done,
    input  wire        dma_wr_busy,
    input  wire        dma_wr_done,

    // Legacy MAC PE Interface
    output wire         pe_load_weight,
    output wire         pe_valid_in,
    output wire  signed [7:0]  pe_x_in,
    output wire  signed [31:0] pe_y_in,
    input  wire signed [7:0]  pe_x_out,
    input  wire signed [31:0] pe_y_out,
    input  wire               pe_valid_out
);

    wire select_pe = (address[3] == 1'b1);
    wire select_sys = (address[3] == 1'b0);

    // Legacy MAC PE Controller
    wire [31:0] pe_readdata;
    wire        pe_readdatavalid;
    mac_pe_ctrl u_mac_pe_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .reg_addr       (address[1:0]),
        .reg_read       (read & select_pe),
        .reg_write      (write & select_pe),
        .reg_writedata  (writedata),
        .reg_readdata   (pe_readdata),
        .reg_readdatavalid(pe_readdatavalid),
        .load_weight    (pe_load_weight),
        .valid_in       (pe_valid_in),
        .x_in           (pe_x_in),
        .y_in           (pe_y_in),
        .x_out          (pe_x_out),
        .y_out          (pe_y_out),
        .valid_out      (pe_valid_out)
    );

    // System Registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq_start <= 1'b0;
            seq_mode  <= 2'd0;
            seq_total_rows <= 32'd0;
            weight_latch_en <= 1'b0;
            dma_rd_addr <= 32'd0;
            dma_rd_len  <= 32'd0;
            dma_rd_start <= 1'b0;
            dma_wr_addr <= 32'd0;
            dma_wr_len  <= 32'd0;
            dma_wr_start <= 1'b0;
        end else begin
            seq_start <= 1'b0;
            dma_rd_start <= 1'b0;
            dma_wr_start <= 1'b0;
            weight_latch_en <= 1'b0;

            if (write && select_sys) begin
                case (address[2:0])
                    3'd0: begin
                        seq_mode  <= writedata[2:1];
                        seq_start <= writedata[0];
                    end
                    3'd2: dma_rd_addr <= writedata;
                    3'd3: dma_rd_len  <= writedata;
                    3'd4: dma_wr_addr <= writedata;
                    3'd5: begin
                        dma_wr_len   <= {16'd0, writedata[15:0]};
                        dma_rd_start <= writedata[16];
                        dma_wr_start <= writedata[17];
                    end
                    3'd6: seq_total_rows <= writedata;
                    3'd7: weight_latch_en <= writedata[0];
                    default: ;
                endcase
            end
        end
    end

    // Read Multiplexer
    reg [31:0] sys_readdata;
    reg        sys_readdatavalid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_readdata <= 32'd0;
            sys_readdatavalid <= 1'b0;
        end else begin
            sys_readdatavalid <= (read && select_sys);
            if (read && select_sys) begin
                case (address[2:0])
                    3'd0: sys_readdata <= {29'd0, seq_mode, 1'b0};
                    3'd1: sys_readdata <= {30'd0, seq_done, seq_busy};
                    3'd2: sys_readdata <= dma_rd_addr;
                    3'd3: sys_readdata <= dma_rd_len;
                    3'd4: sys_readdata <= dma_wr_addr;
                    3'd5: sys_readdata <= {14'd0, dma_wr_done, dma_rd_done, 14'd0, dma_wr_busy, dma_rd_busy}; // DMA status mapping matching legacy
                    3'd6: sys_readdata <= seq_total_rows;
                    3'd7: sys_readdata <= {28'd0, dma_wr_done, dma_rd_done, dma_wr_busy, dma_rd_busy};
                    default: sys_readdata <= 32'd0;
                endcase
            end
        end
    end

    always @(*) begin
        if (sys_readdatavalid) begin
            readdata = sys_readdata;
            readdatavalid = sys_readdatavalid;
        end else if (pe_readdatavalid) begin
            readdata = pe_readdata;
            readdatavalid = pe_readdatavalid;
        end else begin
            readdata = 32'd0;
            readdatavalid = 1'b0;
        end
    end

endmodule
