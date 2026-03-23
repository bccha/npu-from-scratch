`timescale 1ns / 1ps

module npu_ocm_accumulator (
    input  wire        clk,
    input  wire        rst_n,

    // Control Interface (from npu_ctrl)
    input  wire        accum_start,
    output reg         accum_done,
    input  wire [31:0] accum_dst_addr,       // Base address in OCM
    input  wire [31:0] accum_num_elements,   // Number of 32-bit words to process
    input  wire [1:0]  seq_mode,             // seq_mode[1]==0: ACCUM, seq_mode[1]==1: DRAIN

    // Interface from Systolic Array
    input  wire [255:0] pe_dout,
    input  wire         pe_valid_out,
    output reg          pe_ready_out,

    // Interface to Post-Processor (Drain Mode)
    output reg  [31:0]  pp_data_out,
    output reg          pp_valid_out,
    output reg  [3:0]   pp_channel_idx,      // 0 to 7 to indicate which channel within the row
    input  wire         pp_ready_in,         // PP is ready to accept

    // Avalon-MM Master (Read)
    output reg  [31:0]  avm_read_address,
    output reg          avm_read_read,
    input  wire [31:0]  avm_read_readdata,
    input  wire         avm_read_waitrequest,
    input  wire         avm_read_readdatavalid,
    output wire [7:0]   avm_read_burstcount,

    // Avalon-MM Master (Write)
    output reg  [31:0]  avm_write_address,
    output reg          avm_write_write,
    output reg  [31:0]  avm_write_writedata,
    output wire [3:0]   avm_write_byteenable,
    input  wire         avm_write_waitrequest,
    output wire [7:0]   avm_write_burstcount
);

    assign avm_read_burstcount = 8'd1;
    assign avm_write_burstcount = 8'd1;
    assign avm_write_byteenable = 4'b1111;

    // Mode Definitions
    wire mode_drain = seq_mode[1]; // 1 = DRAIN (OCM -> PP), 0 = ACCUM (Array -> OCM)

    // State Machine
    localparam STATE_IDLE            = 4'd0;
    
    // Accumulate States
    localparam STATE_ACC_WAIT_DATA   = 4'd1;
    localparam STATE_ACC_READ_CMD    = 4'd2;
    localparam STATE_ACC_WAIT_READ   = 4'd3;
    localparam STATE_ACC_WRITE_CMD   = 4'd4;
    
    // Drain States
    localparam STATE_DRN_READ_CMD    = 4'd5;
    localparam STATE_DRN_WAIT_READ   = 4'd6;
    localparam STATE_DRN_PUSH_PP     = 4'd7;

    reg [3:0] state;

    reg [31:0] base_ocm_addr;
    reg [255:0] latched_pe_dout;
    reg [3:0]  channel_cnt; // 0 to 7
    reg [31:0] elements_processed;

    // Critical Bugfix: Additions CANNOT mathematically survive over byte-swapping due to Carry-bit distortion!
    // We strictly execute native physical addition over the 32-bit line. 
    wire [31:0] current_src_word = latched_pe_dout[channel_cnt * 32 +: 32];
    wire [31:0] sum_native = current_src_word + avm_read_readdata;
    wire [31:0] swapped_sum_result = sum_native;

    // ==========================================
    // 256-bit Input FIFO (Depth 8)
    // ==========================================
    reg [255:0] in_fifo [0:7];
    reg [2:0]   fifo_wr_ptr;
    reg [2:0]   fifo_rd_ptr;
    reg [3:0]   fifo_count;

    wire fifo_empty = (fifo_count == 4'd0);
    wire fifo_full  = (fifo_count >= 4'd8);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wr_ptr <= 3'd0;
            // fifo_rd_ptr and fifo_count are handled in the main state machine block
        end else if (pe_valid_out && !fifo_full) begin
            in_fifo[fifo_wr_ptr] <= pe_dout;
            fifo_wr_ptr <= fifo_wr_ptr + 1'b1;
        end
    end

    // Always accept data unless FIFO is full
    always @(*) begin
        pe_ready_out = !fifo_full;
    end

    // FIFO Counter Management logic
    wire fifo_push = pe_valid_out && !fifo_full;
    wire fifo_pop  = (state == STATE_ACC_WAIT_DATA) && !fifo_empty;

    // =======================================================
    // Main Pipeline State Machine
    // =======================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            accum_done <= 1'b0;
            
            avm_read_read <= 1'b0;
            avm_read_address <= 32'd0;
            
            avm_write_write <= 1'b0;
            avm_write_address <= 32'd0;
            avm_write_writedata <= 32'd0;
            
            pp_valid_out <= 1'b0;
            pp_data_out <= 32'd0;
            pp_channel_idx <= 4'd0;
            
            base_ocm_addr <= 32'd0;
            channel_cnt <= 4'd0;
            elements_processed <= 32'd0;
            latched_pe_dout <= 256'd0;
            
            fifo_rd_ptr <= 3'd0;
            fifo_count <= 4'd0;
        end else begin
            if (fifo_push && !fifo_pop) fifo_count <= fifo_count + 1'b1;
            else if (!fifo_push && fifo_pop) fifo_count <= fifo_count - 1'b1;

            case (state)
                STATE_IDLE: begin
                    pp_valid_out <= 1'b0;
                    if (accum_start) begin
                        accum_done <= 1'b0;
                        base_ocm_addr <= accum_dst_addr;
                        elements_processed <= 32'd0;
                        channel_cnt <= 4'd0;
                        
                        // We reset FIFO on start
                        fifo_rd_ptr <= 3'd0;
                        fifo_count <= 4'd0;

                        if (mode_drain) begin
                            state <= STATE_DRN_READ_CMD;
                            avm_read_address <= accum_dst_addr;
                            avm_read_read <= 1'b1;
                        end else begin
                            state <= STATE_ACC_WAIT_DATA;
                        end
                    end
                end

                // ==========================================
                // ACCUMULATE MODE (Array -> OCM RMW)
                // ==========================================
                STATE_ACC_WAIT_DATA: begin
                    if (!fifo_empty) begin
                        latched_pe_dout <= in_fifo[fifo_rd_ptr];
                        fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                        channel_cnt <= 4'd0;
                        
                        state <= STATE_ACC_READ_CMD;
                        avm_read_address <= base_ocm_addr;
                        avm_read_read <= 1'b1;
                    end else if (accum_done) begin
                        // Done state holds
                    end
                end

                STATE_ACC_READ_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0;
                        state <= STATE_ACC_WAIT_READ;
                    end
                end

                STATE_ACC_WAIT_READ: begin
                    if (avm_read_readdatavalid) begin
                        state <= STATE_ACC_WRITE_CMD;
                        avm_write_address <= avm_read_address;
                        avm_write_writedata <= swapped_sum_result;
                        avm_write_write <= 1'b1;
                    end
                end

                STATE_ACC_WRITE_CMD: begin
                    if (!avm_write_waitrequest) begin
                        avm_write_write <= 1'b0;
                        
                        if (channel_cnt == 4'd7) begin
                            // Finished this 256-bit row
                            base_ocm_addr <= base_ocm_addr + 32'd32; // Advance by 32 bytes
                            elements_processed <= elements_processed + 32'd8; // 8 words processed
                            
                            // Let's assume the run is done when software issues another command or we hit num_elements.
                            // But usually systolic arrays stream continuously until the sequence is done.
                            // We can check if we hit num_elements to auto-stop.
                            if ((elements_processed + 8) >= accum_num_elements) begin
                                accum_done <= 1'b1;
                                state <= STATE_IDLE;
                            end else begin
                                state <= STATE_ACC_WAIT_DATA;
                            end
                        end else begin
                            // Next channel word
                            channel_cnt <= channel_cnt + 1'b1;
                            state <= STATE_ACC_READ_CMD;
                            avm_read_address <= avm_read_address + 32'd4;
                            avm_read_read <= 1'b1;
                        end
                    end
                end

                // ==========================================
                // DRAIN MODE (OCM -> Post-Processor)
                // ==========================================
                STATE_DRN_READ_CMD: begin
                    if (!avm_read_waitrequest) begin
                        avm_read_read <= 1'b0;
                        state <= STATE_DRN_WAIT_READ;
                    end
                end

                STATE_DRN_WAIT_READ: begin
                    if (avm_read_readdatavalid) begin
                        // We must issue data to PP. But if PP is not ready, we hold it.
                        // Actually, wait_read guarantees the data arrived this cycle.
                        pp_data_out <= avm_read_readdata;
                        pp_valid_out <= 1'b1;
                        pp_channel_idx <= channel_cnt;
                        state <= STATE_DRN_PUSH_PP;
                    end
                end

                STATE_DRN_PUSH_PP: begin
                    if (pp_ready_in && pp_valid_out) begin
                        pp_valid_out <= 1'b0;
                        
                        if (channel_cnt == 4'd7) begin
                            channel_cnt <= 4'd0;
                        end else begin
                            channel_cnt <= channel_cnt + 1'b1;
                        end

                        elements_processed <= elements_processed + 1'b1;
                        if ((elements_processed + 1) >= accum_num_elements) begin
                            accum_done <= 1'b1;
                            state <= STATE_IDLE;
                        end else begin
                            state <= STATE_DRN_READ_CMD;
                            avm_read_address <= avm_read_address + 32'd4;
                            avm_read_read <= 1'b1;
                        end
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
