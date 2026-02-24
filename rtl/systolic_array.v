module systolic_array #(
    parameter N = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [N-1:0] load_weight,
    input  wire [N-1:0] valid_in,               // 각 행(Row)별 유효 신호
    input  wire [N*DATA_WIDTH-1:0] x_in,        // 왼쪽에서 들어오는 8개의 INT8 입력
    input  wire [N*ACC_WIDTH-1:0]  y_in,        // 위에서 내려오는 8개의 32-bit 초기 부분합
    output wire [N*DATA_WIDTH-1:0] x_out,       // 오른쪽으로 나가는 데이터 (디버깅/확장용)
    output wire [N*ACC_WIDTH-1:0]  y_out,       // 아래로 나오는 최종 누산 결과 8개
    output wire [N-1:0] valid_out               // 최종 출력 유효 신호
);

    // 2차원 와이어 배열 선언 (PE 간 연결용)
    wire [DATA_WIDTH-1:0] x_wire [0:N-1][0:N];
    wire [ACC_WIDTH-1:0]  y_wire [0:N][0:N-1];
    wire                  v_wire [0:N-1][0:N];

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : row
            // 각 행의 왼쪽 첫 번째 입력 연결
            assign x_wire[i][0] = x_in[i*DATA_WIDTH +: DATA_WIDTH];
            assign v_wire[i][0] = valid_in[i];
            assign x_out[i*DATA_WIDTH +: DATA_WIDTH] = x_wire[i][N];
            assign valid_out[i] = v_wire[i][N];

            for (j = 0; j < N; j = j + 1) begin : col
                // 각 열의 위쪽 첫 번째 입력 및 아래쪽 최종 출력 연결
                if (i == 0) assign y_wire[0][j] = y_in[j*ACC_WIDTH +: ACC_WIDTH];
                if (i == N-1) assign y_out[j*ACC_WIDTH +: ACC_WIDTH] = y_wire[N][j];

                // 8x8 PE 인스턴스화
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_weight[i]),
                    .valid_in(v_wire[i][j]),
                    .x_in(x_wire[i][j]),
                    .y_in(y_wire[i][j]),
                    .x_out(x_wire[i][j+1]),     // 오른쪽 PE로 전달
                    .y_out(y_wire[i+1][j]),     // 아래쪽 PE로 전달
                    .valid_out(v_wire[i][j+1])
                );
            end
        end
    endgenerate

endmodule