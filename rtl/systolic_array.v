module systolic_array #(
    parameter N = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Upstream Inputs (from Left and Top)
    input  wire [N-1:0]            load_weight_in,
    input  wire [N-1:0]            valid_in,       // Row valid
    output wire [N-1:0]            ready_out,      // Row ready
    input  wire [N*DATA_WIDTH-1:0] x_in,           // Row activations
    input  wire [N*ACC_WIDTH-1:0]  y_in,           // Col initial partial sums

    // Global Controls
    input  wire                    weight_latch_en,

    // Downstream Outputs (to Right and Bottom)
    output wire [N-1:0]            valid_out_x,    // Row valid out (flushed past right edge)
    output wire [N-1:0]            valid_out_y,    // Col valid out (from bottom)
    input  wire [N-1:0]            ready_in,       // Col ready in (from bottom)
    output wire [N*DATA_WIDTH-1:0] x_out,          // Row activations out
    output wire [N*ACC_WIDTH-1:0]  y_out           // Col final sums
);

    // 2D Wires for PE Connections
    wire [DATA_WIDTH-1:0] x_wire   [0:N-1][0:N];
    wire [ACC_WIDTH-1:0]  y_wire   [0:N][0:N-1];
    
    // Handshake Wires
    wire                  v_wire_x [0:N-1][0:N];   // Valid propagates left-to-right (horizontal)
    wire                  v_wire_y [0:N][0:N-1];   // Valid propagates top-to-bottom (vertical)
    wire                  r_wire_x [0:N-1][0:N];   // Ready propagates right-to-left (horizontal)
    wire                  r_wire_y [0:N][0:N-1];   // Ready propagates bottom-to-top (vertical)
    
    wire                  lw_wire  [0:N-1][0:N];   // Load weight propagates left-to-right

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : row
            // Horizontal Mapping (Left-to-Right)
            assign x_wire[i][0]   = x_in[i*DATA_WIDTH +: DATA_WIDTH];
            assign v_wire_x[i][0] = valid_in[i];
            assign lw_wire[i][0]  = load_weight_in[i];
            assign ready_out[i]   = r_wire_x[i][0];
            
            // To Right Edge (Flush)
            assign x_out[i*DATA_WIDTH +: DATA_WIDTH] = x_wire[i][N];
            assign valid_out_x[i] = v_wire_x[i][N];
            assign r_wire_x[i][N] = 1'b1; // The right boundary is always ready to consume waste

            for (j = 0; j < N; j = j + 1) begin : col
                // Vertical Mapping (Top-to-Bottom)
                if (i == 0) begin
                    assign y_wire[0][j]   = y_in[j*ACC_WIDTH +: ACC_WIDTH];
                    assign v_wire_y[0][j] = 1'b1; // Top edge y_in (bias) is always valid
                end
                
                if (i == N-1) begin
                    assign y_out[j*ACC_WIDTH +: ACC_WIDTH] = y_wire[N][j];
                    assign valid_out_y[j] = v_wire_y[N][j];
                    assign r_wire_y[N][j] = ready_in[j]; // Bottom boundary stalls based on downstream
                end
                
                // Split the PE's ready_out back into horizontal and vertical dependencies
                wire pe_ready_out_x, pe_ready_out_y;
                
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    
                    // Upstream
                    .valid_in_x(v_wire_x[i][j]),
                    .valid_in_y(v_wire_y[i][j]),
                    .ready_out_x(pe_ready_out_x),
                    .ready_out_y(pe_ready_out_y),
                    .weight_shift_in(lw_wire[i][j]),
                    .x_in(x_wire[i][j]),
                    .y_in(y_wire[i][j]),

                    // Downstream
                    .valid_out_x(v_wire_x[i][j+1]),
                    .valid_out_y(v_wire_y[i+1][j]),
                    .ready_in_x(r_wire_x[i][j+1]),
                    .ready_in_y(r_wire_y[i+1][j]),
                    .weight_shift_out(lw_wire[i][j+1]),
                    .x_out(x_wire[i][j+1]),     
                    .y_out(y_wire[i+1][j]),
                    
                    // Global
                    .weight_latch_en(weight_latch_en)
                );

                assign r_wire_x[i][j] = pe_ready_out_x;
                assign r_wire_y[i][j] = pe_ready_out_y;
            end
        end
    endgenerate

endmodule