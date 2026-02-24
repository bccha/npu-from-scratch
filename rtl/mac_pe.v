`timescale 1ns / 1ps

module mac_pe #(
    parameter DATA_WIDTH = 8,  // INT8 입력 (Activation & Weight)
    parameter ACC_WIDTH  = 32  // 32-bit 누산기 (Overflow 방지)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Control Signals
    input  wire                         load_weight, // 1: 가중치 로드 모드, 0: 연산 모드
    input  wire                         valid_in,    // 1: 입력 데이터 유효함

    // Data Inputs
    input  wire signed [DATA_WIDTH-1:0] x_in,        // 왼쪽에서 들어오는 입력 (Activation)
    input  wire signed [ACC_WIDTH-1:0]  y_in,        // 위에서 내려오는 부분합 (Partial Sum)

    // Data Outputs
    output reg  signed [DATA_WIDTH-1:0] x_out,       // 오른쪽 다음 PE로 넘겨줄 입력
    output reg  signed [ACC_WIDTH-1:0]  y_out,       // 아래쪽 다음 PE로 내려줄 부분합
    output reg                          valid_out    // 출력 데이터 유효함
);

    // 1. 내부 가중치 레지스터 (Weight Stationary의 핵심)
    reg signed [DATA_WIDTH-1:0] weight_reg;

    // 2. 연산용 와이어 선언
    wire signed [DATA_WIDTH*2-1:0] mult_out; // 8bit x 8bit = 16bit
    wire signed [ACC_WIDTH-1:0]    add_out;

    // 3. 조합 논리 (Combinational Logic) - 합성 시 DSP 블록에 자동 매핑됨
    assign mult_out = x_in * weight_reg;      // 곱셈
    assign add_out  = y_in + mult_out;        // 누산 (위에서 온 값 + 현재 곱셈 결과)

    // 4. 순차 논리 (Sequential Logic)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= {DATA_WIDTH{1'b0}};
            x_out      <= {DATA_WIDTH{1'b0}};
            y_out      <= {ACC_WIDTH{1'b0}};
            valid_out  <= 1'b0;
        end else begin
            if (load_weight) begin
                // [가중치 로드 모드] 
                // Weight Stationary 구조에서 각 행의 PE들은 시프트 레지스터처럼 동작하며 가중치를 채움
                weight_reg <= x_in;
                x_out      <= x_in; // 1클럭만에 바로 오른쪽으로 넘김 (정상 시프트)
                valid_out  <= 1'b0;
            end else begin
                // [연산 모드]
                x_out     <= x_in;     
                y_out     <= add_out;  
                valid_out <= valid_in;
            end
        end
    end

endmodule
