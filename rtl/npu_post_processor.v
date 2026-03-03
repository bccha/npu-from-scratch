`timescale 1ns / 1ps

module npu_post_processor (
    input  wire        clk,
    input  wire        rst_n,

    // Control Interface (from npu_ctrl)
    input  wire        start,
    output reg         done,
    input  wire [31:0] src_addr,
    input  wire [31:0] dst_addr,
    input  wire [31:0] bias_addr,
    input  wire [31:0] num_elements,
    input  wire [31:0] shift_val,

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
    localparam STATE_IDLE           = 4'd0;
    localparam STATE_READ_BIAS_CMD  = 4'd1;
    localparam STATE_WAIT_BIAS_DATA = 4'd2;
    localparam STATE_READ_Z_CMD     = 4'd3;
    localparam STATE_WAIT_Z_DATA    = 4'd4;
    localparam STATE_PROCESS        = 4'd5;
    localparam STATE_WRITE_CMD      = 4'd6;
    localparam STATE_DONE           = 4'd7;

    reg [3:0] state;

    reg [31:0] elements_processed;
    reg [2:0]  pack_counter; // 0 to 3 to pack 4 bytes into 1 word
    reg [31:0] packed_word;

    reg signed [31:0] current_bias;
    reg signed [31:0] current_z;

    // Registers for pipeline
    reg signed [31:0] h_raw;
    reg signed [31:0] h_shift;
    reg signed [31:0] h_final; // clamped 0..127

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            elements_processed <= 32'd0;
            pack_counter <= 3'd0;
            packed_word <= 32'd0;
            
            avm_read_address <= 32'd0;
            avm_read_read <= 1'b0;
            
            avm_write_address <= 32'd0;
            avm_write_write <= 1'b0;
            avm_write_writedata <= 32'd0;
            
            current_bias <= 32'd0;
            current_z <= 32'd0;
            done <= 1'b0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= STATE_READ_BIAS_CMD;
                        elements_processed <= 32'd0;
                        pack_counter <= 3'd0;
                        packed_word <= 32'd0;
                        
                        avm_read_address <= bias_addr; // the very first bias read at elements_processed=0
                        avm_read_read <= 1'b1;
                        avm_write_address <= dst_addr;
                    end
                end

                STATE_READ_BIAS_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0; 
                        state <= STATE_WAIT_BIAS_DATA;
                    end
                end

                STATE_WAIT_BIAS_DATA: begin
                    if (avm_read_readdatavalid) begin
                        current_bias <= avm_read_readdata;
                        
                        // Proceed to read Z
                        state <= STATE_READ_Z_CMD;
                        avm_read_address <= src_addr + (elements_processed << 2);
                        avm_read_read <= 1'b1;
                    end
                end

                STATE_READ_Z_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0; 
                        state <= STATE_WAIT_Z_DATA;
                    end
                end

                STATE_WAIT_Z_DATA: begin
                    if (avm_read_readdatavalid) begin
                        current_z <= avm_read_readdata;
                        state <= STATE_PROCESS;
                    end
                end

                STATE_PROCESS: begin
                    // $Z + Bias -> Shift -> ReLU/Clamp
                    // Hardware Z values from NPU output are byte-swapped, matching __builtin_bswap32 in C code
                    h_raw = {current_z[7:0], current_z[15:8], current_z[23:16], current_z[31:24]} + current_bias;
                    h_shift = h_raw >>> shift_val;
                    
                    if (h_shift < 0) begin
                        h_final = 0;
                    end else if (h_shift > 127) begin
                        h_final = 127;
                    end else begin
                        h_final = h_shift;
                    end

                    // Pack into the 32-bit register.
                    // To match the `hw_c = c ^ 1` logic in C, we must swap pairs of columns.
                    // Since pack_counter (0,1,2,3) corresponds to 4 columns, we XOR it with 1 
                    // to write the data into the correct byte position of `packed_word`.
                    case (pack_counter ^ 3'd1)
                        3'd0: packed_word[7:0]   <= h_final[7:0];
                        3'd1: packed_word[15:8]  <= h_final[7:0];
                        3'd2: packed_word[23:16] <= h_final[7:0];
                        3'd3: packed_word[31:24] <= h_final[7:0];
                        default:;
                    endcase
                    
                    elements_processed <= elements_processed + 1;
                    
                    if (pack_counter == 3'd3 || (elements_processed + 1) == num_elements) begin
                        state <= STATE_WRITE_CMD;
                        avm_write_write <= 1'b1;
                        
                        // Guarantee the latest byte is combo-assigned to write data matching the XOR layout
                        avm_write_writedata[31:24] <= ((pack_counter ^ 3'd1) == 3'd3) ? h_final[7:0] : packed_word[31:24];
                        avm_write_writedata[23:16] <= ((pack_counter ^ 3'd1) == 3'd2) ? h_final[7:0] : packed_word[23:16];
                        avm_write_writedata[15:8]  <= ((pack_counter ^ 3'd1) == 3'd1) ? h_final[7:0] : packed_word[15:8];
                        avm_write_writedata[7:0]   <= ((pack_counter ^ 3'd1) == 3'd0) ? h_final[7:0] : packed_word[7:0];
                        
                        pack_counter <= 3'd0;
                    end else begin
                        pack_counter <= pack_counter + 3'd1;
                        state <= STATE_READ_BIAS_CMD;
                        // Wrap bias reads every 8 elements for MAC columns using bitwise AND.
                        avm_read_address <= bias_addr + (( (elements_processed + 1) & 32'd7 ) << 2);
                        avm_read_read <= 1'b1;
                    end
                end

                STATE_WRITE_CMD: begin
                    if (!avm_write_waitrequest) begin
                        avm_write_write <= 1'b0;
                        avm_write_address <= avm_write_address + 32'd4;
                        packed_word <= 32'd0;
                        
                        if (elements_processed >= num_elements) begin
                            state <= STATE_DONE;
                        end else begin
                            state <= STATE_READ_BIAS_CMD;
                            // Wrap bias reads every 8 elements using bitwise AND
                            avm_read_address <= bias_addr + (( elements_processed & 32'd7 ) << 2);
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
