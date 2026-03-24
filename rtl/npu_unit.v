`timescale 1ns / 1ps

module npu_unit #(
    parameter AXI_WIDTH = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM Slave Interface (Unified)
    input  wire [9:0]  avs_address,      // Expanded to 10-bit for Bias SRAM
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
    output wire [2:0]  st_source_empty,

    // Post-Processor Avalon-MM Master (Read)
    output wire [31:0] pp_read_address,
    output wire        pp_read_read,
    input  wire [31:0] pp_read_readdata,
    input  wire        pp_read_waitrequest,
    input  wire        pp_read_readdatavalid,
    output wire [7:0]  pp_read_burstcount,

    // Post-Processor Avalon-MM Master (Write)
    output wire [31:0] pp_write_address,
    output wire        pp_write_write,
    output wire [31:0] pp_write_writedata,
    output wire [3:0]  pp_write_byteenable,
    input  wire        pp_write_waitrequest,
    output wire [7:0]  pp_write_burstcount
);

    // Pipeline avs_readdata due to NPU CTRL having 1 cycle latency
    wire [31:0] ctrl_readdata;
    wire        ctrl_readdatavalid;
    assign avs_readdata = ctrl_readdata;
    assign avs_readdatavalid = ctrl_readdatavalid;

    // ------------------------------------------------------------------
    // 1. Centralized Control Unit
    // ------------------------------------------------------------------

    wire        seq_start;
    wire [1:0]  seq_mode;
    wire [31:0] seq_total_rows;
    wire        seq_busy;
    wire        seq_done;
    wire        weight_latch_en;
    wire [4:0]  shift_val;

    wire        csr_pe_load_weight;
    wire        csr_pe_valid_in;
    wire [7:0]  csr_pe_x_in;
    wire [31:0] csr_pe_y_in;
    wire [7:0]  csr_pe_x_out;
    wire [31:0] csr_pe_y_out;
    wire        csr_pe_valid_out;

    wire        accum_start;
    wire        accum_done;
    wire [31:0] accum_src_addr;
    wire [31:0] accum_dst_addr;
    wire [31:0] accum_num_elements;

    wire [8:0]  pp_bias_addr;
    wire [31:0] pp_bias_rdata;

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
        .shift_val      (shift_val),
        
        .pe_load_weight (csr_pe_load_weight),
        .pe_valid_in    (csr_pe_valid_in),
        .pe_x_in        (csr_pe_x_in),
        .pe_y_in        (csr_pe_y_in),
        .pe_x_out       (csr_pe_x_out),
        .pe_y_out       (csr_pe_y_out),
        .pe_valid_out   (csr_pe_valid_out),
        
        .accum_start    (accum_start),
        .accum_done     (accum_done),
        .accum_src_addr (accum_src_addr),
        .accum_dst_addr (accum_dst_addr),
        .accum_num_elements (accum_num_elements),

        .pp_bias_addr   (pp_bias_addr),
        .pp_bias_rdata  (pp_bias_rdata)
    );

    // ------------------------------------------------------------------
    // 2. Data Flow Signals
    // ------------------------------------------------------------------

    // Sink -> Array
    wire [63:0] pe_din;
    wire        pe_valid_in;
    wire        pe_ready_in; 

    // Array -> Accumulator
    wire [255:0] pe_dout;
    wire         pe_valid_out;
    wire         pe_ready_out;

    // Accumulator -> Post-Processor
    wire [31:0] accum_pp_data;
    wire        accum_pp_valid;
    wire [3:0]  accum_pp_channel;
    wire        accum_pp_ready;

    // Post-Processor -> Stream Source
    wire [7:0]  pp_out_data;
    wire        pp_out_valid;
    wire        pp_out_ready;

    // ------------------------------------------------------------------
    // 3. NPU Stream Controller
    // ------------------------------------------------------------------
    npu_stream_ctrl u_npu_stream_ctrl (
        .clk                     (clk),
        .rst_n                   (rst_n),
        
        .st_sink_data            (st_sink_data),
        .st_sink_valid           (st_sink_valid),
        .st_sink_ready           (st_sink_ready),
        .st_sink_startofpacket   (st_sink_startofpacket),
        .st_sink_endofpacket     (st_sink_endofpacket),
        .st_sink_empty           (st_sink_empty),

        .st_source_data          (st_source_data),
        .st_source_valid         (st_source_valid),
        .st_source_ready         (st_source_ready),
        .st_source_startofpacket (st_source_startofpacket),
        .st_source_endofpacket   (st_source_endofpacket),
        .st_source_empty         (st_source_empty),

        .seq_total_rows          (seq_total_rows),

        .pe_din                  (pe_din),
        .pe_valid_in             (pe_valid_in),
        .pe_ready_in             (pe_ready_in),

        .pp_data                 (pp_out_data),
        .pp_valid                (pp_out_valid),
        .pp_ready                (pp_out_ready)
    );

    // ------------------------------------------------------------------
    // 4. Systolic Array Core (8x8)
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
        .ready_out       (pe_ready_in), 
        .x_in            (pe_din[63:0]),
        .y_in            (256'd0),
        .weight_latch_en (weight_latch_en),
        .y_out           (pe_dout),
        .valid_out       (pe_valid_out),
        .ready_in        (pe_ready_out) 
    );

    // ------------------------------------------------------------------
    // 5. OCM Accumulator (Hardware Fusion)
    // ------------------------------------------------------------------
    npu_ocm_accumulator u_npu_ocm_accumulator (
        .clk                    (clk),
        .rst_n                  (rst_n),

        .accum_start            (accum_start),
        .accum_done             (accum_done),
        .accum_dst_addr         (accum_dst_addr),
        .accum_num_elements     (accum_num_elements),
        .seq_mode               (seq_mode),

        .pe_dout                (pe_dout),
        .pe_valid_out           (pe_valid_out),
        .pe_ready_out           (pe_ready_out),

        .pp_data_out            (accum_pp_data),
        .pp_valid_out           (accum_pp_valid),
        .pp_channel_idx         (accum_pp_channel),
        .pp_ready_in            (accum_pp_ready),
        
        .avm_read_address       (pp_read_address),
        .avm_read_read          (pp_read_read),
        .avm_read_readdata      (pp_read_readdata),      
        .avm_read_waitrequest   (pp_read_waitrequest),
        .avm_read_readdatavalid (pp_read_readdatavalid),
        .avm_read_burstcount    (pp_read_burstcount),
        
        .avm_write_address      (pp_write_address),
        .avm_write_write        (pp_write_write),
        .avm_write_writedata    (pp_write_writedata),
        .avm_write_byteenable   (pp_write_byteenable),
        .avm_write_waitrequest  (pp_write_waitrequest),  
        .avm_write_burstcount   (pp_write_burstcount)
    );

    // ------------------------------------------------------------------
    // 6. Post-Processor (Bias, Scale, ReLU)
    // ------------------------------------------------------------------
    npu_post_processor u_npu_post_processor (
        .clk             (clk),
        .rst_n           (rst_n),

        .in_data         (accum_pp_data),
        .in_valid        (accum_pp_valid),
        .in_channel_idx  (accum_pp_channel),
        .in_bias_base    (9'd0), // Base offset for bias RAM, mapped to 0 for now
        .in_ready        (accum_pp_ready),

        .pp_bias_addr    (pp_bias_addr),
        .pp_bias_rdata   (pp_bias_rdata),

        .out_data        (pp_out_data),
        .out_valid       (pp_out_valid),
        .out_ready       (pp_out_ready),

        .shift_val       (shift_val)
    );

    // ------------------------------------------------------------------
    // 7. Legacy Single MAC PE (Test Mode via CSR)
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
        
        .weight_latch_en  (1'b1)
    );

endmodule
