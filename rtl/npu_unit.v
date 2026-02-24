`timescale 1ns / 1ps

module npu_unit #(
    parameter AXI_WIDTH = 64
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

    // Avalon-MM Read Master Interface (DMA)
    input  wire        dma_rd_m_waitrequest,
    input  wire [AXI_WIDTH-1:0] dma_rd_m_readdata,
    input  wire        dma_rd_m_readdatavalid,
    output wire [4:0]  dma_rd_m_burstcount,
    output wire [31:0] dma_rd_m_address,
    output wire        dma_rd_m_read,

    // Avalon-MM Write Master Interface (DMA)
    input  wire        dma_wr_m_waitrequest,
    output wire [4:0]  dma_wr_m_burstcount,
    output wire [31:0] dma_wr_m_address,
    output wire        dma_wr_m_write,
    output wire [AXI_WIDTH-1:0] dma_wr_m_writedata
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

    // Control <-> DMA
    wire [31:0] dma_rd_addr;
    wire [31:0] dma_rd_len;
    wire        dma_rd_start;
    wire [31:0] dma_wr_addr;
    wire [31:0] dma_wr_len;
    wire        dma_wr_start;
    wire        dma_rd_busy;
    wire        dma_rd_done;
    wire        dma_wr_busy;
    wire        dma_wr_done;

    // Control <-> Legacy PE
    wire        pe_load_weight;
    wire        pe_valid_in;
    wire [7:0]  pe_x_in;
    wire [31:0] pe_y_in;
    wire [7:0]  pe_x_out;
    wire [31:0] pe_y_out;
    wire        pe_valid_out;

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
        
        .dma_rd_addr    (dma_rd_addr),
        .dma_rd_len     (dma_rd_len),
        .dma_rd_start   (dma_rd_start),
        .dma_wr_addr    (dma_wr_addr),
        .dma_wr_len     (dma_wr_len),
        .dma_wr_start   (dma_wr_start),
        .dma_rd_busy    (dma_rd_busy),
        .dma_rd_done    (dma_rd_done),
        .dma_wr_busy    (dma_wr_busy),
        .dma_wr_done    (dma_wr_done),
        
        .pe_load_weight (pe_load_weight),
        .pe_valid_in    (pe_valid_in),
        .pe_x_in        (pe_x_in),
        .pe_y_in        (pe_y_in),
        .pe_x_out       (pe_x_out),
        .pe_y_out       (pe_y_out),
        .pe_valid_out   (pe_valid_out)
    );

    // ------------------------------------------------------------------
    // 2. Data Move Engine (DMA)
    // ------------------------------------------------------------------
    wire [AXI_WIDTH-1:0] dma_data_to_npu;
    wire        dma_data_to_npu_valid;
    wire [AXI_WIDTH-1:0] dma_data_from_npu;
    wire        dma_data_from_npu_valid;

    npu_dma #(.AXI_WIDTH(AXI_WIDTH)) u_npu_dma (
        .clk                  (clk),
        .rst_n                (rst_n),
        
        // Control interface
        .rd_addr              (dma_rd_addr),
        .rd_len               (dma_rd_len),
        .rd_start_pulse       (dma_rd_start),
        .wr_addr              (dma_wr_addr),
        .wr_len               (dma_wr_len),
        .wr_start_pulse       (dma_wr_start),
        .rd_busy              (dma_rd_busy),
        .rd_done              (dma_rd_done),
        .wr_busy              (dma_wr_busy),
        .wr_done              (dma_wr_done),
        
        // Masters
        .rd_m_waitrequest     (dma_rd_m_waitrequest),
        .rd_m_readdata        (dma_rd_m_readdata),
        .rd_m_readdatavalid   (dma_rd_m_readdatavalid),
        .rd_m_burstcount      (dma_rd_m_burstcount),
        .rd_m_address         (dma_rd_m_address),
        .rd_m_read            (dma_rd_m_read),
        
        .wr_m_waitrequest     (dma_wr_m_waitrequest),
        .wr_m_burstcount      (dma_wr_m_burstcount),
        .wr_m_address         (dma_wr_m_address),
        .wr_m_write           (dma_wr_m_write),
        .wr_m_writedata       (dma_wr_m_writedata),
        
        // NPU Data stream
        .data_to_npu          (dma_data_to_npu),
        .data_to_npu_valid    (dma_data_to_npu_valid),
        .data_to_npu_ready    (dma_data_to_npu_ready),
        .data_from_npu        (dma_data_from_npu),
        .data_from_npu_valid  (dma_data_from_npu_valid),
        .data_from_npu_ready  (dma_data_from_npu_ready)
    );

    // ------------------------------------------------------------------
    // 3. NPU Sequencer
    // ------------------------------------------------------------------
    wire        core_load_weight;
    wire [7:0]  core_valid_in;
    wire [63:0] core_x_in;
    wire [255:0]core_y_in;
    wire [255:0]core_y_out;
    wire [7:0]  core_valid_out;

    npu_sequencer #(
        .N(8),
        .DATA_WIDTH(8),
        .AXI_WIDTH(AXI_WIDTH)
    ) u_npu_sequencer (
        .clk                  (clk),
        .rst_n                (rst_n),
        .start                (seq_start),
        .mode                 (seq_mode),
        .total_rows           (seq_total_rows),
        .busy                 (seq_busy),
        .done                 (seq_done),

        // DMA stream
        .dma_data_in          (dma_data_to_npu),
        .dma_data_in_valid    (dma_data_to_npu_valid),
        .dma_data_in_ready    (dma_data_to_npu_ready),

        .dma_data_out         (dma_data_from_npu),
        .dma_data_out_valid   (dma_data_from_npu_valid),
        .dma_data_out_ready   (dma_data_from_npu_ready),

        // Core
        .core_load_weight     (core_load_weight),
        .core_valid_in        (core_valid_in),
        .core_x_in            (core_x_in),
        .core_y_in            (core_y_in),
        .core_y_out           (core_y_out),
        .core_valid_out       (core_valid_out)
    );

    // ------------------------------------------------------------------
    // --- 4. Systolic Array 8x8 Core ---
    // ------------------------------------------------------------------
    systolic_core #(
        .N(8),
        .DATA_WIDTH(8),
        .ACC_WIDTH(32)
    ) u_systolic_core (
        .clk          (clk),
        .rst_n        (rst_n),
        .load_weight  (core_load_weight),
        .valid_in     (core_valid_in),
        .x_in         (core_x_in),
        .y_in         ({8*32{1'b0}}),
        .y_out        (core_y_out),
        .valid_out    (core_valid_out)
    );

    // ------------------------------------------------------------------
    // 5. Legacy Single MAC PE (Test Mode)
    // ------------------------------------------------------------------
    mac_pe u_mac_pe (
        .clk          (clk),
        .rst_n        (rst_n),
        .load_weight  (pe_load_weight),
        .valid_in     (pe_valid_in),
        .x_in         (pe_x_in),
        .y_in         (pe_y_in),
        .x_out        (pe_x_out),
        .y_out        (pe_y_out),
        .valid_out    (pe_valid_out)
    );

endmodule
