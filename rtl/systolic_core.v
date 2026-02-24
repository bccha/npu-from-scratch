module systolic_core #(
    parameter N = 8,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire load_weight,
    input  wire [N-1:0] valid_in,
    input  wire [N*DATA_WIDTH-1:0] x_in,   // 평면적인 입력 데이터
    input  wire [N*ACC_WIDTH-1:0]  y_in,   // 초기 Partial Sum (보통 0)
    output wire [N*ACC_WIDTH-1:0]  y_out,  // 정렬된 최종 결과값
    output wire [N-1:0] valid_out          // 정렬된 출력 유효 신호
);

    wire [N*DATA_WIDTH-1:0] x_skewed;
    wire [N-1:0]            v_skewed;
    wire [N-1:0]            lw_skewed;
    wire [N*ACC_WIDTH-1:0]  y_notskewed; // 어레이에서 막 나온 삐뚤빼뚤한 결과
    wire [N-1:0]            v_notskewed;

    // 1. Input Skew Buffers (입력을 계단식으로 지연)
    // Row i는 i만큼의 클럭 지연을 가짐
    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : input_skew
            if (i == 0) begin
                assign x_skewed[i*DATA_WIDTH +: DATA_WIDTH] = x_in[i*DATA_WIDTH +: DATA_WIDTH];
                assign v_skewed[i] = valid_in[i];
                assign lw_skewed[i] = load_weight; 
            end else begin
                reg [DATA_WIDTH-1:0] x_delay [0:i-1];
                reg [i-1:0]          v_delay;
                reg [i-1:0]          lw_delay;
                integer k;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        v_delay <= 0;
                        lw_delay <= 0;
                        for (k = 0; k < i; k = k + 1) begin
                            x_delay[k] <= {DATA_WIDTH{1'b0}};
                        end
                    end else begin
                        x_delay[0] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        v_delay[0] <= valid_in[i];
                        lw_delay[0] <= load_weight;
                        for (k = 1; k < i; k = k + 1) begin
                            x_delay[k] <= x_delay[k-1];
                            v_delay[k] <= v_delay[k-1];
                            lw_delay[k] <= lw_delay[k-1];
                        end
                    end
                end
                assign x_skewed[i*DATA_WIDTH +: DATA_WIDTH] = x_delay[i-1];
                assign v_skewed[i] = v_delay[i-1];
                assign lw_skewed[i] = lw_delay[i-1];
            end
        end
    endgenerate

    // 2. 4x4 Systolic Array Instance
    systolic_array #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(lw_skewed),
        .valid_in(v_skewed),
        .x_in(x_skewed),
        .y_in(y_in),
        .y_out(y_notskewed),
        .valid_out(v_notskewed)
    );

    // 3. Output De-skew Buffers (결과를 다시 정렬)
    // Col j는 (N-1-j)만큼 추가 지연을 주어 수평 정렬
    generate
        for (j = 0; j < N; j = j + 1) begin : output_deskew
            localparam DELAY = (N - 1) - j;
            if (DELAY == 0) begin
                assign y_out[j*ACC_WIDTH +: ACC_WIDTH] = y_notskewed[j*ACC_WIDTH +: ACC_WIDTH];
                assign valid_out[j] = v_notskewed[j];
            end else begin
                reg [ACC_WIDTH-1:0] y_delay [0:DELAY-1];
                reg [DELAY-1:0] v_out_delay;
                integer k;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        v_out_delay <= 0;
                        for (k = 0; k < DELAY; k = k + 1) begin
                            y_delay[k] <= {ACC_WIDTH{1'b0}};
                        end
                    end else begin
                        y_delay[0] <= y_notskewed[j*ACC_WIDTH +: ACC_WIDTH];
                        v_out_delay[0] <= v_notskewed[j];
                        for (k = 1; k < DELAY; k = k + 1) begin
                            y_delay[k] <= y_delay[k-1];
                            v_out_delay[k] <= v_out_delay[k-1];
                        end
                    end
                end
                assign y_out[j*ACC_WIDTH +: ACC_WIDTH] = y_delay[DELAY-1];
                assign valid_out[j] = v_out_delay[DELAY-1];
            end
        end
    endgenerate

endmodule