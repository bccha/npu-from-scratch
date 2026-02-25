`timescale 1ns / 1ps

module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // --- X-Axis: Activation & Weight Shift (Left to Right) ---
    input  wire                         valid_in_x,
    output wire                         ready_out_x, 
    input  wire                         weight_shift_in,
    input  wire signed [DATA_WIDTH-1:0] x_in,

    output reg                          valid_out_x,
    input  wire                         ready_in_x,
    output reg                          weight_shift_out,
    output reg  signed [DATA_WIDTH-1:0] x_out,

    // --- Y-Axis: Partial Sum (Top to Bottom) ---
    input  wire                         valid_in_y,
    output wire                         ready_out_y,
    input  wire signed [ACC_WIDTH-1:0]  y_in,

    output reg                          valid_out_y,
    input  wire                         ready_in_y,
    output reg  signed [ACC_WIDTH-1:0]  y_out,

    // --- Global Controls ---
    input  wire                         weight_latch_en
);

    // 1. 가중치 이중 버퍼링 (Double Buffering)
    reg signed [DATA_WIDTH-1:0] shadow_weight_reg;
    reg signed [DATA_WIDTH-1:0] active_weight_reg;

    // 2. 명시적 부호 확장 (Explicit Sign-Extension)
    wire signed [DATA_WIDTH*2-1:0] mult_out;
    wire signed [ACC_WIDTH-1:0]    add_out;
    wire signed [ACC_WIDTH-1:0]    mult_ext;

    assign mult_out = x_in * active_weight_reg;
    assign mult_ext = { {(ACC_WIDTH - DATA_WIDTH*2){mult_out[DATA_WIDTH*2-1]}}, mult_out };
    assign add_out  = y_in + mult_ext;

    // --- 3. 2D Elastic Handshake Logic (Fork & Join) ---
    wire stall_x = valid_out_x && !ready_in_x;
    wire stall_y = valid_out_y && !ready_in_y;

    wire can_fire_x = !stall_x;
    wire can_fire_y = !stall_y;

    // Weight Phase: Only X needs to be valid. Y is ignored.
    wire fire_load = valid_in_x && weight_shift_in && can_fire_x;
    // MAC Phase: Both X and Y must be valid.
    wire fire_calc = valid_in_x && valid_in_y && !weight_shift_in && can_fire_x && can_fire_y;

    wire fire = fire_load || fire_calc;

    assign ready_out_x = fire;
    assign ready_out_y = fire_calc; // Y is consumed ONLY during calculation

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shadow_weight_reg <= {DATA_WIDTH{1'b0}};
            active_weight_reg <= {DATA_WIDTH{1'b0}};
            x_out            <= {DATA_WIDTH{1'b0}};
            y_out            <= {ACC_WIDTH{1'b0}};
            valid_out_x      <= 1'b0;
            valid_out_y      <= 1'b0;
            weight_shift_out <= 1'b0;
        end else begin
            
            // Latch 펄스
            if (weight_latch_en) begin
                active_weight_reg <= shadow_weight_reg;
            end

            if (fire_load) begin
                shadow_weight_reg <= x_in;           
                x_out             <= shadow_weight_reg; 
                valid_out_x       <= 1'b1; // MUST propagate valid so downstream knows a token arrived           
                valid_out_y       <= 1'b0; 
                weight_shift_out  <= 1'b1; // Pass the shift command downstream          
            end else if (fire_calc) begin
                x_out            <= x_in;     
                y_out            <= add_out;  
                valid_out_x      <= 1'b1;
                valid_out_y      <= 1'b1;
                weight_shift_out <= 1'b0;
            end else begin
                if (valid_out_x && ready_in_x) valid_out_x <= 1'b0;
                if (valid_out_y && ready_in_y) valid_out_y <= 1'b0;
            end
        end
    end

endmodule
