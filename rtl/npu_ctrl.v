`timescale 1ns / 1ps

module npu_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // Unified Register Interface
    input  wire [9:0]  address,
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
    output reg  [4:0]  shift_val,

    // Legacy MAC PE Interface
    output wire         pe_load_weight,
    output wire         pe_valid_in,
    output wire  signed [7:0]  pe_x_in,
    output wire  signed [31:0] pe_y_in,
    input  wire signed [7:0]  pe_x_out,
    input  wire signed [31:0] pe_y_out,
    input  wire               pe_valid_out,

    // Standalone OCM Accumulator Interface
    output reg         accum_start,
    input  wire        accum_done,
    output reg  [31:0] accum_src_addr,
    output reg  [31:0] accum_dst_addr,
    output reg  [31:0] accum_num_elements,

    // Post-Processor Bias RAM Interface (Port B)
    input  wire [8:0]  pp_bias_addr,
    output reg  [31:0] pp_bias_rdata
);

    wire select_sys  = (address[9:8] == 2'b00) && (address[7:4] == 4'h0) && (address[3] == 1'b0);
    wire select_pe   = (address[9:8] == 2'b00) && (address[7:4] == 4'h0) && (address[3] == 1'b1);
    wire select_acc  = (address[9:8] == 2'b00) && (address[7:4] == 4'h2); // 0x020 to 0x02F
    wire select_bias = (address[9] == 1'b1);                              // 0x200 to 0x3FF (512 words)

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
    reg accum_done_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq_start <= 1'b0;
            seq_mode  <= 2'd0;
            seq_total_rows <= 32'd0;
            weight_latch_en <= 1'b0;
            shift_val <= 5'd8; // Default 8

            accum_start <= 1'b0;
            accum_src_addr <= 32'd0;
            accum_dst_addr <= 32'd0;
            accum_num_elements <= 32'd0;
        end else begin
            seq_start <= 1'b0;
            weight_latch_en <= 1'b0;
            accum_start <= 1'b0;

            // Accum Done Latch Logic
            if (accum_start) begin
                accum_done_latched <= 1'b0;
            end else if (accum_done) begin
                accum_done_latched <= 1'b1;
            end

            if (write && select_sys) begin
                case (address[2:0])
                    3'd0: begin
                        seq_mode  <= writedata[2:1];
                        seq_start <= writedata[0];
                    end
                    3'd2: shift_val <= writedata[4:0];
                    3'd6: seq_total_rows <= writedata;
                    3'd7: weight_latch_en <= writedata[0];
                    default: ;
                endcase
            end
            
            if (write && select_acc) begin
                case (address[3:0])
                    4'h0: accum_start <= writedata[0];
                    4'h1: ; // Read-only status
                    4'h2: accum_src_addr <= writedata;
                    4'h3: accum_dst_addr <= writedata;
                    4'h5: accum_num_elements <= writedata;
                    default: ;
                endcase
            end
        end
    end

    // Read Multiplexer
    reg [31:0] sys_readdata;
    reg        sys_readdatavalid;
    reg [31:0] acc_readdata;
    reg        acc_readdatavalid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_readdata <= 32'd0;
            sys_readdatavalid <= 1'b0;
            acc_readdata <= 32'd0;
            acc_readdatavalid <= 1'b0;
        end else begin
            sys_readdatavalid <= (read && select_sys);
            if (read && select_sys) begin
                case (address[2:0])
                    3'd0: sys_readdata <= {29'd0, seq_mode, 1'b0};
                    3'd1: sys_readdata <= {30'd0, seq_done, seq_busy};
                    3'd2: sys_readdata <= {27'd0, shift_val};
                    3'd6: sys_readdata <= seq_total_rows;
                    3'd7: sys_readdata <= {31'd0, weight_latch_en};
                    default: sys_readdata <= 32'd0;
                endcase
            end
            
            acc_readdatavalid <= (read && select_acc);
            if (read && select_acc) begin
                case (address[3:0])
                    4'h0: acc_readdata <= {31'd0, 1'b0}; // Start is self-clearing
                    4'h1: acc_readdata <= {31'd0, accum_done_latched};
                    4'h2: acc_readdata <= accum_src_addr;
                    4'h3: acc_readdata <= accum_dst_addr;
                    4'h5: acc_readdata <= accum_num_elements;
                    default: acc_readdata <= 32'd0;
                endcase
            end
        end
    end

    // Bias SRAM Instantiation (512 x 32-bit = 2KB)
    reg [31:0] bias_ram [0:511];
    reg [31:0] bias_readdata;
    reg        bias_readdatavalid;

    // Port A: Avalon-MM (HPS side)
    always @(posedge clk) begin
        if (write && select_bias) begin
            bias_ram[address[8:0]] <= writedata;
        end
        if (read && select_bias) begin
            bias_readdata <= bias_ram[address[8:0]];
        end
    end

    // Delay valid for Bias RAM to match 1 cycle read latency
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_readdatavalid <= 1'b0;
        end else begin
            bias_readdatavalid <= (read && select_bias);
        end
    end

    // Port B: Post-Processor side (Dedicated Read)
    always @(posedge clk) begin
        pp_bias_rdata <= bias_ram[pp_bias_addr];
    end

    always @(*) begin
        if (sys_readdatavalid) begin
            readdata = sys_readdata;
            readdatavalid = sys_readdatavalid;
        end else if (pe_readdatavalid) begin
            readdata = pe_readdata;
            readdatavalid = pe_readdatavalid;
        end else if (acc_readdatavalid) begin
            readdata = acc_readdata;
            readdatavalid = acc_readdatavalid;
        end else if (bias_readdatavalid) begin
            readdata = bias_readdata;
            readdatavalid = bias_readdatavalid;
        end else begin
            readdata = 32'd0;
            readdatavalid = 1'b0;
        end
    end

endmodule
