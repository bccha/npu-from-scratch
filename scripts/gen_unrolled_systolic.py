import os

N = 8

def gen_systolic_array():
    lines = []
    lines.append("module systolic_array #(")
    lines.append("    parameter N = 8,")
    lines.append("    parameter DATA_WIDTH = 8,")
    lines.append("    parameter ACC_WIDTH = 32")
    lines.append(")(")
    lines.append("    input  wire clk,")
    lines.append("    input  wire rst_n,")
    lines.append("    input  wire [N-1:0] load_weight,")
    lines.append("    input  wire [N-1:0] valid_in,")
    lines.append("    input  wire [N*DATA_WIDTH-1:0] x_in,")
    lines.append("    input  wire [N*ACC_WIDTH-1:0]  y_in,")
    lines.append("    output wire [N*DATA_WIDTH-1:0] x_out,")
    lines.append("    output wire [N*ACC_WIDTH-1:0]  y_out,")
    lines.append("    output wire [N-1:0] valid_out")
    lines.append(");")
    lines.append("")
    lines.append("    wire [DATA_WIDTH-1:0] x_wire [0:N-1][0:N];")
    lines.append("    wire [ACC_WIDTH-1:0]  y_wire [0:N][0:N-1];")
    lines.append("    wire                  v_wire [0:N-1][0:N];")
    lines.append("")

    for i in range(N):
        lines.append(f"    assign x_wire[{i}][0] = x_in[{i}*DATA_WIDTH +: DATA_WIDTH];")
        lines.append(f"    assign v_wire[{i}][0] = valid_in[{i}];")
        lines.append(f"    assign x_out[{i}*DATA_WIDTH +: DATA_WIDTH] = x_wire[{i}][N];")
        lines.append(f"    assign valid_out[{i}] = v_wire[{i}][N];")
    lines.append("")

    for j in range(N):
        lines.append(f"    assign y_wire[0][{j}] = y_in[{j}*ACC_WIDTH +: ACC_WIDTH];")
        lines.append(f"    assign y_out[{j}*ACC_WIDTH +: ACC_WIDTH] = y_wire[N][{j}];")
    lines.append("")

    for i in range(N):
        for j in range(N):
            lines.append(f"    mac_pe #(")
            lines.append(f"        .DATA_WIDTH(DATA_WIDTH),")
            lines.append(f"        .ACC_WIDTH(ACC_WIDTH)")
            lines.append(f"    ) u_pe_{i}_{j} (")
            lines.append(f"        .clk(clk),")
            lines.append(f"        .rst_n(rst_n),")
            lines.append(f"        .load_weight(load_weight[{i}]),")
            lines.append(f"        .valid_in(v_wire[{i}][{j}]),")
            lines.append(f"        .x_in(x_wire[{i}][{j}]),")
            lines.append(f"        .y_in(y_wire[{i}][{j}]),")
            lines.append(f"        .x_out(x_wire[{i}][{j+1}]),")
            lines.append(f"        .y_out(y_wire[{i+1}][{j}]),")
            lines.append(f"        .valid_out(v_wire[{i}][{j+1}])")
            lines.append(f"    );")
            lines.append("")

    lines.append("endmodule")
    return "\n".join(lines)


def gen_systolic_core():
    lines = []
    lines.append("module systolic_core #(")
    lines.append("    parameter N = 8,")
    lines.append("    parameter DATA_WIDTH = 8,")
    lines.append("    parameter ACC_WIDTH = 32")
    lines.append(")(")
    lines.append("    input  wire clk,")
    lines.append("    input  wire rst_n,")
    lines.append("    input  wire [N-1:0] load_weight,")
    lines.append("    input  wire [N-1:0] valid_in,")
    lines.append("    input  wire [N*DATA_WIDTH-1:0] x_in,")
    lines.append("    input  wire [N*ACC_WIDTH-1:0]  y_in,")
    lines.append("    output wire [N*ACC_WIDTH-1:0]  y_out,")
    lines.append("    output wire [N-1:0] valid_out")
    lines.append(");")
    lines.append("")
    lines.append("    wire [N*DATA_WIDTH-1:0] x_skewed;")
    lines.append("    wire [N-1:0]            v_skewed;")
    lines.append("    wire [N-1:0]            lw_skewed;")
    lines.append("    wire [N*ACC_WIDTH-1:0]  y_notskewed;")
    lines.append("    wire [N-1:0]            v_notskewed;")
    lines.append("")
    
    # Input skew
    lines.append("    // 1. Input Skew Buffers")
    lines.append("    assign x_skewed[0*DATA_WIDTH +: DATA_WIDTH] = x_in[0*DATA_WIDTH +: DATA_WIDTH];")
    lines.append("    assign v_skewed[0] = valid_in[0];")
    lines.append("    assign lw_skewed[0] = load_weight[0];")
    lines.append("")
    
    for i in range(1, N):
        lines.append(f"    reg [DATA_WIDTH-1:0] x_delay_row{i} [0:{i-1}];")
        lines.append(f"    reg [{i-1}:0] v_delay_row{i};")
        lines.append(f"    reg [{i-1}:0] lw_delay_row{i};")
        
        lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        lines.append(f"        if (!rst_n) begin")
        lines.append(f"            v_delay_row{i} <= 0;")
        lines.append(f"            lw_delay_row{i} <= 0;")
        for k in range(i):
            lines.append(f"            x_delay_row{i}[{k}] <= {{DATA_WIDTH{{1'b0}}}};")
        lines.append(f"        end else begin")
        lines.append(f"            x_delay_row{i}[0] <= x_in[{i}*DATA_WIDTH +: DATA_WIDTH];")
        lines.append(f"            v_delay_row{i}[0] <= valid_in[{i}];")
        lines.append(f"            lw_delay_row{i}[0] <= load_weight[{i}];")
        for k in range(1, i):
            lines.append(f"            x_delay_row{i}[{k}] <= x_delay_row{i}[{k-1}];")
            lines.append(f"            v_delay_row{i}[{k}] <= v_delay_row{i}[{k-1}];")
            lines.append(f"            lw_delay_row{i}[{k}] <= lw_delay_row{i}[{k-1}];")
        lines.append(f"        end")
        lines.append(f"    end")
        lines.append(f"    assign x_skewed[{i}*DATA_WIDTH +: DATA_WIDTH] = x_delay_row{i}[{i-1}];")
        lines.append(f"    assign v_skewed[{i}] = v_delay_row{i}[{i-1}];")
        lines.append(f"    assign lw_skewed[{i}] = lw_delay_row{i}[{i-1}];")
        lines.append("")

    # Array Instance
    lines.append("    // 2. Systolic Array Instance")
    lines.append("    systolic_array #(")
    lines.append("        .N(N),")
    lines.append("        .DATA_WIDTH(DATA_WIDTH),")
    lines.append("        .ACC_WIDTH(ACC_WIDTH)")
    lines.append("    ) u_array (")
    lines.append("        .clk(clk),")
    lines.append("        .rst_n(rst_n),")
    lines.append("        .load_weight(lw_skewed),")
    lines.append("        .valid_in(v_skewed),")
    lines.append("        .x_in(x_skewed),")
    lines.append("        .y_in(y_in),")
    lines.append("        .y_out(y_notskewed),")
    lines.append("        .valid_out(v_notskewed)")
    lines.append("    );")
    lines.append("")

    # Output deskew
    lines.append("    // 3. Output De-skew Buffers")
    for j in range(N):
        delay = (N - 1) - j
        if delay == 0:
            lines.append(f"    assign y_out[{j}*ACC_WIDTH +: ACC_WIDTH] = y_notskewed[{j}*ACC_WIDTH +: ACC_WIDTH];")
            lines.append(f"    assign valid_out[{j}] = v_notskewed[{j}];")
        else:
            lines.append(f"    reg [ACC_WIDTH-1:0] y_delay_col{j} [0:{delay-1}];")
            lines.append(f"    reg [{delay-1}:0] v_out_delay_col{j};")
            lines.append(f"    always @(posedge clk or negedge rst_n) begin")
            lines.append(f"        if (!rst_n) begin")
            lines.append(f"            v_out_delay_col{j} <= 0;")
            for k in range(delay):
                lines.append(f"            y_delay_col{j}[{k}] <= {{ACC_WIDTH{{1'b0}}}};")
            lines.append(f"        end else begin")
            lines.append(f"            y_delay_col{j}[0] <= y_notskewed[{j}*ACC_WIDTH +: ACC_WIDTH];")
            lines.append(f"            v_out_delay_col{j}[0] <= v_notskewed[{j}];")
            for k in range(1, delay):
                lines.append(f"            y_delay_col{j}[{k}] <= y_delay_col{j}[{k-1}];")
                lines.append(f"            v_out_delay_col{j}[{k}] <= v_out_delay_col{j}[{k-1}];")
            lines.append(f"        end")
            lines.append(f"    end")
            lines.append(f"    assign y_out[{j}*ACC_WIDTH +: ACC_WIDTH] = y_delay_col{j}[{delay-1}];")
            lines.append(f"    assign valid_out[{j}] = v_out_delay_col{j}[{delay-1}];")
        lines.append("")

    lines.append("endmodule")
    return "\n".join(lines)


if __name__ == '__main__':
    with open('../rtl/systolic_array.v', 'w') as f:
        f.write(gen_systolic_array() + '\n')
    
    with open('../rtl/systolic_core.v', 'w') as f:
        f.write(gen_systolic_core() + '\n')
