`timescale 1ns / 1ps

module npu_unit #(
    parameter AXI_WIDTH = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM Slave Interface (Unified)
    input  wire [3:0]  avs_address,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire        avs_read,
    output wire [31:0] avs_readdata,
    output wire        avs_readdatavalid,

    // Avalon-ST Sink Interface (from Memory/MSGDMA)
    input  wire [63:0] st_sink_data,
    input  wire        st_sink_valid,
    output wire        st_sink_ready,
    input  wire        st_sink_startofpacket,
    input  wire        st_sink_endofpacket,
    input  wire [2:0]  st_sink_empty,

    // Avalon-ST Source Interface (to Memory/MSGDMA)
    output wire [63:0] st_source_data,
    output wire        st_source_valid,
    input  wire        st_source_ready,
    output wire        st_source_startofpacket,
    output wire        st_source_endofpacket,
    output wire [2:0]  st_source_empty
);

    // Pipeline avs_readdata due to NPU CTRL having 1 cycle latency
    wire [31:0] ctrl_readdata;
    wire        ctrl_readdatavalid;
    assign avs_readdata = ctrl_readdata;
    assign avs_readdatavalid = ctrl_readdatavalid;

    // ------------------------------------------------------------------
    // 1. Centralized Control Unit
    // ------------------------------------------------------------------

    // Control <-> Sequencer
    wire        seq_start;
    wire [1:0]  seq_mode;
    wire [31:0] seq_total_rows;
    wire        seq_busy;
    wire        seq_done;
    wire        weight_latch_en;

    // The DMA and Sequencer wires have been removed as they are now handled by MSGDMA via Avalon-ST.
    // Control <-> NPU Stream (Mode Control etc.)
    // TODO: Connect seq_start or seq_mode to the stream controller if mode switching is needed.


    // Control <-> Legacy PE (CSR Testing Only)
    wire        csr_pe_load_weight;
    wire        csr_pe_valid_in;
    wire [7:0]  csr_pe_x_in;
    wire [31:0] csr_pe_y_in;
    wire [7:0]  csr_pe_x_out;
    wire [31:0] csr_pe_y_out;
    wire        csr_pe_valid_out;

    npu_ctrl u_npu_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .address        (avs_address),
        .write          (avs_write),
        .writedata      (avs_writedata),
        .read           (avs_read),
        .readdata       (ctrl_readdata),
        .readdatavalid  (ctrl_readdatavalid),
        
        .seq_start      (seq_start),
        .seq_mode       (seq_mode),
        .seq_total_rows (seq_total_rows),
        .seq_busy       (seq_busy),
        .seq_done       (seq_done),
        .weight_latch_en(weight_latch_en),
        
        .pe_load_weight (csr_pe_load_weight),
        .pe_valid_in    (csr_pe_valid_in),
        .pe_x_in        (csr_pe_x_in),
        .pe_y_in        (csr_pe_y_in),
        .pe_x_out       (csr_pe_x_out),
        .pe_y_out       (csr_pe_y_out),
        .pe_valid_out   (csr_pe_valid_out)
    );

    // ------------------------------------------------------------------
    // 2. NPU Stream Controller (replaces DMA & Sequencer)
    // ------------------------------------------------------------------
    wire [63:0] pe_din;
    wire        pe_valid_in;
    wire        pe_ready_in; 
    wire [255:0] pe_dout;
    wire        pe_valid_out;
    wire        pe_ready_out;

    npu_stream_ctrl u_npu_stream_ctrl (
        .clk                     (clk),
        .rst_n                   (rst_n),
        
        // Avalon-ST Sink
        .st_sink_data            (st_sink_data),
        .st_sink_valid           (st_sink_valid),
        .st_sink_ready           (st_sink_ready),
        .st_sink_startofpacket   (st_sink_startofpacket),
        .st_sink_endofpacket     (st_sink_endofpacket),
        .st_sink_empty           (st_sink_empty),

        // Avalon-ST Source
        .st_source_data          (st_source_data),
        .st_source_valid         (st_source_valid),
        .st_source_ready         (st_source_ready),
        .st_source_startofpacket (st_source_startofpacket),
        .st_source_endofpacket   (st_source_endofpacket),
        .st_source_empty         (st_source_empty),

        // NPU Global Configuration
        .seq_total_rows          (seq_total_rows),

        // NPU PE Interface
        .pe_din                  (pe_din),
        .pe_valid_in             (pe_valid_in),
        .pe_ready_in             (pe_ready_in),

        .pe_dout                 (pe_dout),
        .pe_valid_out            (pe_valid_out),
        .pe_ready_out            (pe_ready_out)
    );

    // ------------------------------------------------------------------
    systolic_core #(
        .N(8),
        .DATA_WIDTH(8),
        .ACC_WIDTH(32)
    ) u_systolic_core (
        .clk             (clk),
        .rst_n           (rst_n),
        .load_weight_in  ({8{seq_mode[0] & pe_valid_in}}),
        .valid_in        (pe_valid_in),
        .ready_out       (pe_ready_in), // connect upstream ready to stream controller
        .x_in            (pe_din[63:0]),
        .y_in            (256'd0),
        .weight_latch_en (weight_latch_en),
        .y_out           (pe_dout),
        .valid_out       (pe_valid_out),
        .ready_in        (pe_ready_out) // stream controller pulling outputs
    );

    // ------------------------------------------------------------------
    // 5. Legacy Single MAC PE (Test Mode via CSR)
    // ------------------------------------------------------------------
    mac_pe u_mac_pe (
        .clk              (clk),
        .rst_n            (rst_n),
        .valid_in_x       (csr_pe_valid_in),
        .ready_out_x      (),
        .weight_shift_in  (csr_pe_load_weight),
        .x_in             (csr_pe_x_in),
        .valid_out_x      (),
        .ready_in_x       (1'b1),
        .weight_shift_out (),
        .x_out            (csr_pe_x_out),
        
        .valid_in_y       (csr_pe_valid_in),
        .ready_out_y      (),
        .y_in             (csr_pe_y_in),
        .valid_out_y      (csr_pe_valid_out),
        .ready_in_y       (1'b1),
        .y_out            (csr_pe_y_out),
        
        .weight_latch_en  (1'b1) // Always latch immediately for single CSR test mode
    );

endmodule
