`timescale 1ns / 1ps

module npu_ocm_accumulator (
    input  wire        clk,
    input  wire        rst_n,

    // Control Interface (from npu_ctrl)
    input  wire        start,
    output reg         done,
    input  wire [31:0] src_addr,
    input  wire [31:0] dst_addr,       // Also the Accumulator target
    input  wire [31:0] num_elements,

    // Avalon-MM Master (Read)
    output reg  [31:0] avm_read_address,
    output reg         avm_read_read,
    input  wire [31:0] avm_read_readdata,
    input  wire        avm_read_waitrequest,
    input  wire        avm_read_readdatavalid,
    output wire [7:0]  avm_read_burstcount,

    // Avalon-MM Master (Write)
    output reg  [31:0] avm_write_address,
    output reg         avm_write_write,
    output reg  [31:0] avm_write_writedata,
    output wire [3:0]  avm_write_byteenable,
    input  wire        avm_write_waitrequest,
    output wire [7:0]  avm_write_burstcount
);

    assign avm_read_burstcount = 8'd1;
    assign avm_write_burstcount = 8'd1;
    assign avm_write_byteenable = 4'b1111; // Always write 32-bit aligned words

    // State Machine Codes
    localparam STATE_IDLE           = 3'd0;
    localparam STATE_READ_SRC_CMD   = 3'd1;
    localparam STATE_WAIT_SRC_DATA  = 3'd2;
    localparam STATE_READ_ACCUM_CMD = 3'd3;
    localparam STATE_WAIT_ACCUM_DATA= 3'd4;
    localparam STATE_WRITE_CMD      = 3'd5;
    localparam STATE_DONE           = 3'd6;

    reg [2:0] state;
    reg [31:0] elements_processed;

    reg signed [31:0] current_src;
    reg signed [31:0] current_accum;

    // ----- Endianness Hardware Swapper for MSGDMA compatibility -----
    // The NPU output deposited by MSGDMA ST-to-MM bridge is Big-Endian byte-swapped per word.
    // We must unswap before addition so carries propagate across bytes correctly, then re-swap for memory.
    wire [31:0] unswapped_src = {current_src[7:0], current_src[15:8], current_src[23:16], current_src[31:24]};
    wire [31:0] unswapped_acc = {avm_read_readdata[7:0], avm_read_readdata[15:8], avm_read_readdata[23:16], avm_read_readdata[31:24]};
    wire [31:0] sum_unswapped = unswapped_src + unswapped_acc;
    wire [31:0] swapped_sum_result = {sum_unswapped[7:0], sum_unswapped[15:8], sum_unswapped[23:16], sum_unswapped[31:24]};
    // ----------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            elements_processed <= 32'd0;
            
            avm_read_address <= 32'd0;
            avm_read_read <= 1'b0;
            
            avm_write_address <= 32'd0;
            avm_write_write <= 1'b0;
            avm_write_writedata <= 32'd0;
            
            current_src <= 32'd0;
            current_accum <= 32'd0;
            done <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= STATE_READ_SRC_CMD;
                        elements_processed <= 32'd0;
                        
                        avm_read_address <= src_addr;
                        avm_read_read <= 1'b1;
                        avm_write_address <= dst_addr;
                    end
                end

                STATE_READ_SRC_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0; 
                        state <= STATE_WAIT_SRC_DATA;
                    end
                end

                STATE_WAIT_SRC_DATA: begin
                    if (avm_read_readdatavalid) begin
                        current_src <= avm_read_readdata;
                        
                        // Proceed to read Accumulator value (dst_addr)
                        state <= STATE_READ_ACCUM_CMD;
                        avm_read_address <= dst_addr + (elements_processed << 2);
                        avm_read_read <= 1'b1;
                    end
                end

                STATE_READ_ACCUM_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0; 
                        state <= STATE_WAIT_ACCUM_DATA;
                    end
                end

                STATE_WAIT_ACCUM_DATA: begin
                    if (avm_read_readdatavalid) begin
                        current_accum <= avm_read_readdata;
                        
                        // Addition process inside write command to save a cycle
                        state <= STATE_WRITE_CMD;
                        avm_write_write <= 1'b1;
                        // Accumulate using the Endianness Swapping network!
                        avm_write_writedata <= swapped_sum_result;
                    end
                end

                STATE_WRITE_CMD: begin
                    if (!avm_write_waitrequest) begin
                        avm_write_write <= 1'b0;
                        avm_write_address <= avm_write_address + 32'd4;
                        elements_processed <= elements_processed + 1;
                        
                        if ((elements_processed + 1) >= num_elements) begin
                            state <= STATE_DONE;
                        end else begin
                            state <= STATE_READ_SRC_CMD;
                            avm_read_address <= src_addr + ((elements_processed + 1) << 2);
                            avm_read_read <= 1'b1;
                        end
                    end
                end

                STATE_DONE: begin
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
