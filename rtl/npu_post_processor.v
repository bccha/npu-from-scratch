`timescale 1ns / 1ps

module npu_post_processor (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [4:0]  shift_val,

    // Interface from OCM Accumulator (Drain Mode)
    input  wire [31:0] in_data,
    input  wire        in_valid,
    input  wire [3:0]  in_channel_idx,  // 0 to 7 to indicate which channel
    input  wire [8:0]  in_bias_base,    // Base address of the currently processed 8 channels in Bias RAM
    output wire        in_ready,

    // Interface to Bias RAM (npu_ctrl Port B)
    output wire [8:0]  pp_bias_addr,
    input  wire [31:0] pp_bias_rdata,

    // Interface to MSGDMA Stream Controller (Packer)
    output reg  [7:0]  out_data,
    output reg         out_valid,
    input  wire        out_ready
);

    // ==========================================
    // Fully Pipelined Post-Processor (4 Stages)
    // ==========================================
    
    // Stall logic
    wire pipeline_stall = out_valid && !out_ready;
    assign in_ready = !pipeline_stall;

    // --- Stage 1: Address Generation & Input Latch ---
    reg [31:0] stg1_data;
    reg        stg1_valid;
    
    // We can directly drive the bias RAM address asynchronously from the incoming valid signals
    // Assuming out_ready is mostly high, we fetch bias for the current input.
    // If stalled, we shouldn't fetch new bias. But BRAM is just read-only here, so safe.
    assign pp_bias_addr = in_bias_base + {5'd0, in_channel_idx};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_data  <= 32'd0;
            stg1_valid <= 1'b0;
        end else if (!pipeline_stall) begin
            stg1_data  <= in_data;
            stg1_valid <= in_valid;
        end
    end

    // --- Stage 2: Memory Read Wait & Alignment ---
    reg [31:0] stg2_data;
    reg [31:0] stg2_bias;
    reg        stg2_valid;

    // Bias RAM has 1 cycle latency. It arrives exactly when stg1_data is ready!
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg2_data  <= 32'd0;
            stg2_bias  <= 32'd0;
            stg2_valid <= 1'b0;
        end else if (!pipeline_stall) begin
            stg2_data  <= stg1_data;
            stg2_bias  <= pp_bias_rdata; 
            stg2_valid <= stg1_valid;
        end
    end

    // --- Stage 3: Addition (Fused Bias) ---
    // Make sure we carry the sign bit correctly.
    wire signed [31:0] signed_data = $signed(stg2_data);
    wire signed [31:0] signed_bias = $signed(stg2_bias);
    wire signed [31:0] sum         = signed_data + signed_bias;

    reg signed [31:0] stg3_sum;
    reg               stg3_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg3_sum   <= 32'd0;
            stg3_valid <= 1'b0;
        end else if (!pipeline_stall) begin
            stg3_sum   <= sum;
            stg3_valid <= stg2_valid;
        end
    end

    // --- Stage 4: Shift and ReLU ---
    // Shift arithmetic right
    wire signed [31:0] shifted_sum = stg3_sum >>> shift_val;
    
    wire [7:0] relu_out;
    assign relu_out = (shifted_sum < 0) ? 8'd0 : 
                      (shifted_sum > 127) ? 8'd127 : 
                      shifted_sum[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data  <= 8'd0;
            out_valid <= 1'b0;
        end else if (!pipeline_stall) begin
            out_data  <= relu_out;
            out_valid <= stg3_valid;
        end else if (out_ready) begin
            out_valid <= 1'b0; // Clear valid if it was consumed and no new pipeline data
        end
    end

endmodule
