`timescale 1ns / 1ps

module mac_pe_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // Register Interface (from npu_ctrl)
    input  wire [1:0]  reg_addr,
    input  wire        reg_read,
    input  wire        reg_write,
    input  wire [31:0] reg_writedata,
    output reg  [31:0] reg_readdata,
    output reg         reg_readdatavalid,

    // Interface to PE
    output reg         load_weight,
    output reg         valid_in,
    output reg  signed [7:0]  x_in,
    output reg  signed [31:0] y_in,
    input  wire signed [7:0]  x_out,
    input  wire signed [31:0] y_out,
    input  wire               valid_out
);

    // Register Bank (Write) & Registered Read Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_weight <= 1'b0;
            valid_in    <= 1'b0;
            x_in        <= 8'd0;
            y_in        <= 32'd0;
            reg_readdata      <= 32'd0;
            reg_readdatavalid <= 1'b0;
        end else begin
            // Write Logic
            if (reg_write) begin
                case (reg_addr)
                    2'd0: begin // REG_CTRL
                        load_weight <= reg_writedata[0];
                        valid_in    <= reg_writedata[1];
                    end
                    2'd1: x_in <= reg_writedata[7:0];
                    2'd2: y_in <= reg_writedata[31:0];
                    default: ;
                endcase
            end

            // Registered Read Logic
            reg_readdatavalid <= reg_read;
            if (reg_read) begin
                case (reg_addr)
                    2'd0: reg_readdata <= {30'd0, valid_in, load_weight};
                    2'd1: reg_readdata <= {24'd0, x_in};
                    2'd2: reg_readdata <= y_in;
                    2'd3: reg_readdata <= y_out;
                    default: reg_readdata <= 32'd0;
                endcase
            end
        end
    end

endmodule
