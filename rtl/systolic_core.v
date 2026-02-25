module systolic_core #(
    parameter N = 8,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Upstream (AXI-Stream like)
    input  wire [N-1:0]            load_weight_in,
    input  wire                    valid_in, // Global valid for the set of inputs
    output wire                    ready_out,// Global ready to upstream
    input  wire [N*DATA_WIDTH-1:0] x_in,     // Activations
    input  wire [N*ACC_WIDTH-1:0]  y_in,     // Initial sums

    // Global Controls
    input  wire                    weight_latch_en,

    // Downstream (AXI-Stream like)
    output wire [N*ACC_WIDTH-1:0]  y_out,    // Final sums
    output wire                    valid_out,// Global valid out
    input  wire                    ready_in  // Global ready from downstream
);

    // ---------------------------------------------------------
    // 1. Input Skew Pipelines
    // ---------------------------------------------------------
    wire [N*DATA_WIDTH-1:0] x_skewed;
    wire [N-1:0]            v_skewed;
    wire [N-1:0]            lw_skewed;
    wire [N-1:0]            r_skewed_in; // Ready signals going BACK from array to the skew buffers

    // We must generate a "ready" signal to the upstream. We are ready if ALL row inputs are ready.
    wire [N-1:0] row_ready_out;
    assign ready_out = &row_ready_out;

    genvar i, j, k;
    generate
        for (i = 0; i < N; i = i + 1) begin : input_skew
            // Local valid/ready for this row's skew pipeline
            wire [i:0] valid_pipe;
            wire [i:0] ready_pipe;
            wire [DATA_WIDTH-1:0] x_pipe [0:i];
            wire [i:0] lw_pipe;

            // Stage 0 is the input to the skew logic
            assign x_pipe[0]      = x_in[i*DATA_WIDTH +: DATA_WIDTH];
            assign valid_pipe[0]  = valid_in;
            assign lw_pipe[0]     = load_weight_in[i];
            assign row_ready_out[i] = ready_pipe[0];

            for (j = 0; j < i; j = j + 1) begin : skew_stage
                // AXI-Stream Pipeline Register Stage
                reg [DATA_WIDTH-1:0] x_reg;
                reg                  v_reg;
                reg                  lw_reg;
                
                assign ready_pipe[j] = ready_pipe[j+1] || !v_reg;

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        x_reg  <= {DATA_WIDTH{1'b0}};
                        v_reg  <= 1'b0;
                        lw_reg <= 1'b0;
                    end else begin
                        if (ready_pipe[j] && valid_pipe[j]) begin
                            x_reg  <= x_pipe[j];
                            v_reg  <= 1'b1;
                            lw_reg <= lw_pipe[j];
                        end else if (ready_pipe[j+1]) begin
                            v_reg  <= 1'b0;
                        end
                    end
                end

                assign x_pipe[j+1]     = x_reg;
                assign valid_pipe[j+1] = v_reg;
                assign lw_pipe[j+1]    = lw_reg;
            end

            // The last stage connects to the systolic array
            assign x_skewed[i*DATA_WIDTH +: DATA_WIDTH] = x_pipe[i];
            assign v_skewed[i]   = valid_pipe[i];
            assign lw_skewed[i]  = lw_pipe[i];
            assign ready_pipe[i] = r_skewed_in[i];
        end
    endgenerate

    // ---------------------------------------------------------
    // 2. Systolic Array
    // ---------------------------------------------------------
    wire [N*ACC_WIDTH-1:0] y_notskewed;
    wire [N-1:0]           v_notskewed;
    wire [N-1:0]           r_notskewed_in; // Ready signals from deskew logic to array
    systolic_array #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight_in(lw_skewed),
        .valid_in(v_skewed),
        .ready_out(r_skewed_in),
        .x_in(x_skewed),
        .y_in(y_in),
        .weight_latch_en(weight_latch_en), // Added missing connection
        .valid_out_x(v_out_dummy),
        .valid_out_y(v_notskewed),
        .ready_in(r_notskewed_in), // From output deskewers
        .x_out(),
        .y_out(y_notskewed)
    );

    // ---------------------------------------------------------
    // 3. Output De-skew Pipelines
    // ---------------------------------------------------------
    // The valids out of the array are per-column. But array columns finish staggered.
    // We deskew them so the entire N-wide result pops out at once.
    wire [N-1:0] col_valid_out;

    generate
        for (j = 0; j < N; j = j + 1) begin : output_deskew
            localparam DELAY = (N - 1) - j;
            
            wire [DELAY:0] valid_pipe;
            wire [DELAY:0] ready_pipe;
            wire [ACC_WIDTH-1:0] y_pipe [0:DELAY];

            // Stage 0 is from the systolic array
            assign y_pipe[0]      = y_notskewed[j*ACC_WIDTH +: ACC_WIDTH];
            assign valid_pipe[0]  = v_notskewed[j];
            
            // Fix: The array's ready_in for this column is simply the ready state of the very first register stage in the deskew buffer
            assign r_notskewed_in[j] = ready_pipe[0];

            for (k = 0; k < DELAY; k = k + 1) begin : deskew_stage
                reg [ACC_WIDTH-1:0] y_reg;
                reg                 v_reg;
                
                assign ready_pipe[k] = ready_pipe[k+1] || !v_reg;

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        y_reg <= {ACC_WIDTH{1'b0}};
                        v_reg <= 1'b0;
                    end else begin
                        if (ready_pipe[k] && valid_pipe[k]) begin
                            y_reg <= y_pipe[k];
                            v_reg <= 1'b1;
                        end else if (ready_pipe[k+1]) begin
                            v_reg <= 1'b0;
                        end
                    end
                end

                assign y_pipe[k+1]     = y_reg;
                assign valid_pipe[k+1] = v_reg;
            end

            // Final output stage for this column
            assign y_out[j*ACC_WIDTH +: ACC_WIDTH] = y_pipe[DELAY];
            assign col_valid_out[j] = valid_pipe[DELAY];
            
            // The last stage is ready to pop either if we're popping the whole row, or if it doesn't have valid data yet
            assign ready_pipe[DELAY] = (ready_in && valid_out) || !valid_pipe[DELAY]; 
        end
    endgenerate

    // Global valid_out is high only when ALL columns have valid data sitting at the edge of the deskew buffers.
    assign valid_out = &col_valid_out;

endmodule