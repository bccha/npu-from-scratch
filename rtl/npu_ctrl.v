`timescale 1ns / 1ps

module npu_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // Unified Register Interface
    input  wire [7:0]  address,
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

    // Legacy MAC PE Interface
    output wire         pe_load_weight,
    output wire         pe_valid_in,
    output wire  signed [7:0]  pe_x_in,
    output wire  signed [31:0] pe_y_in,
    input  wire signed [7:0]  pe_x_out,
    input  wire signed [31:0] pe_y_out,
    input  wire               pe_valid_out,

    // Post-Processor Interface
    output reg         pp_start,
    input  wire        pp_done,
    output reg  [31:0] pp_src_addr,
    output reg  [31:0] pp_dst_addr,
    output reg  [31:0] pp_bias_addr,
    output reg  [31:0] pp_num_elements,
    output reg  [31:0] pp_shift_val
);

    wire select_sys = (address[7:4] == 4'h0) && (address[3] == 1'b0);
    wire select_pe  = (address[7:4] == 4'h0) && (address[3] == 1'b1);
    wire select_pp  = (address[7:4] == 4'h1); // 0x10 to 0x1F

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
    reg pp_done_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq_start <= 1'b0;
            seq_mode  <= 2'd0;
            seq_total_rows <= 32'd0;
            weight_latch_en <= 1'b0;
            
            pp_start <= 1'b0;
            pp_src_addr <= 32'd0;
            pp_dst_addr <= 32'd0;
            pp_bias_addr <= 32'd0;
            pp_num_elements <= 32'd0;
            pp_shift_val <= 32'd0;
        end else begin
            seq_start <= 1'b0;
            weight_latch_en <= 1'b0;
            pp_start <= 1'b0;

            // PP Done Latch Logic
            if (pp_start) begin
                pp_done_latched <= 1'b0;
            end else if (pp_done) begin
                pp_done_latched <= 1'b1;
            end

            if (write && select_sys) begin
                case (address[2:0])
                    3'd0: begin
                        seq_mode  <= writedata[2:1];
                        seq_start <= writedata[0];
                    end
                    3'd6: seq_total_rows <= writedata;
                    3'd7: weight_latch_en <= writedata[0];
                    default: ;
                endcase
            end
            
            if (write && select_pp) begin
                case (address[3:0])
                    4'h0: pp_start <= writedata[0];
                    4'h2: pp_src_addr <= writedata;
                    4'h3: pp_dst_addr <= writedata;
                    4'h4: pp_bias_addr <= writedata;
                    4'h5: pp_num_elements <= writedata;
                    4'h6: pp_shift_val <= writedata;
                    default: ;
                endcase
            end
        end
    end

    // Read Multiplexer
    reg [31:0] sys_readdata;
    reg        sys_readdatavalid;
    reg [31:0] pp_readdata;
    reg        pp_readdatavalid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_readdata <= 32'd0;
            sys_readdatavalid <= 1'b0;
            pp_readdata <= 32'd0;
            pp_readdatavalid <= 1'b0;
        end else begin
            sys_readdatavalid <= (read && select_sys);
            if (read && select_sys) begin
                case (address[2:0])
                    3'd0: sys_readdata <= {29'd0, seq_mode, 1'b0};
                    3'd1: sys_readdata <= {30'd0, seq_done, seq_busy};
                    3'd6: sys_readdata <= seq_total_rows;
                    3'd7: sys_readdata <= {31'd0, weight_latch_en};
                    default: sys_readdata <= 32'd0;
                endcase
            end
            
            pp_readdatavalid <= (read && select_pp);
            if (read && select_pp) begin
                case (address[3:0])
                    4'h0: pp_readdata <= {31'd0, 1'b0}; // Start bit is self-clearing, always read 0
                    4'h1: pp_readdata <= {31'd0, pp_done_latched}; // Latch the Status
                    4'h2: pp_readdata <= pp_src_addr;
                    4'h3: pp_readdata <= pp_dst_addr;
                    4'h4: pp_readdata <= pp_bias_addr;
                    4'h5: pp_readdata <= pp_num_elements;
                    4'h6: pp_readdata <= pp_shift_val;
                    default: pp_readdata <= 32'd0;
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
        end else if (pp_readdatavalid) begin
            readdata = pp_readdata;
            readdatavalid = pp_readdatavalid;
        end else begin
            readdata = 32'd0;
            readdatavalid = 1'b0;
        end
    end

endmodule
