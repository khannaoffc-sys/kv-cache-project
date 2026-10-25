`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// kv_demo_top.v  -  on-board self-test demo for the KV-cache eviction unit
//
// What it does (no computer needed while it runs):
//   * Holds the real INT8 key vectors of Qwen2.5-0.5B layer 0 and layer 10
//     (512 tokens x 64 bytes each) and the EXPECTED kept-token indices
//     in on-chip ROMs, initialised from the same .mem files the ModelSim
//     testbench uses.
//   * When you press START it feeds the keys (4 bytes per clock) into
//     kv_evict_top, collects the 256 kept-token indices the hardware outputs,
//     and compares every one with the expected index.
//   * Shows PASS / FAIL on LEDs and the clock-cycle count on the 7-segment
//     displays.
//
// Port names follow Terasic's DE2-115 convention (CLOCK_50, KEY, SW, LEDR,
// LEDG, HEX0..HEX7). If your board is different, the names / counts may need
// to change. 7-segment displays are assumed ACTIVE-LOW (0 = segment lit),
// as on the DE-series boards.
//
//   KEY[0]  reset (press = reset)           KEY[1]  START (press = run)
//   SW[0]   layer: 0 = layer 0, 1 = layer 10   (read when START is pressed)
//   SW[1]   7-segment shows: 0 = clock cycles, 1 = number of outputs seen
//   LEDG[0] DONE     LEDG[1] PASS     LEDG[2] RUNNING
//   LEDR[0] FAIL     LEDR[16:9] error count (low 8 bits)   LEDR[17] = SW[0]
//   HEX5..HEX0  value in hex (cycle count: expect about 042803 = 272,387)
// -----------------------------------------------------------------------------
module kv_demo_top #(
    // Paths are relative to the Quartus project folder (fpga/quartus).
    // synth_demo.tcl copies the .mem files into fpga/quartus/mem/.
    // The ModelSim testbench overrides these with project-root paths.
    parameter KEYS0_FILE = "mem/layer0_keys_line.mem",
    parameter IDX0_FILE  = "mem/layer0_indices_topk.mem",
    parameter KEYS1_FILE = "mem/layer10_keys_line.mem",
    parameter IDX1_FILE  = "mem/layer10_indices_topk.mem",
    parameter SEQ_LEN    = 512,
    parameter HEAD_DIM   = 64,
    parameter LANES      = 4,
    parameter N_KEEP     = 256,
    parameter SCORE_W    = 21,
    parameter DEBOUNCE_CYCLES = 1000000,   // 20 ms at 50 MHz
    parameter MAX_CYCLES      = 1000000    // watchdog: 20 ms at 50 MHz
)(
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [17:0] SW,
    output wire [17:0] LEDR,
    output wire [8:0]  LEDG,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, HEX6, HEX7
);

    wire clk = CLOCK_50;

    function integer clog2;
        input [31:0] value;
        integer v;
        begin
            v = value - 1;
            clog2 = 0;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v = v >> 1;
            end
        end
    endfunction

    localparam BEATS  = HEAD_DIM / LANES;
    localparam AW     = (clog2(SEQ_LEN) < 1) ? 1 : clog2(SEQ_LEN);
    localparam BW     = (clog2(BEATS)   < 1) ? 1 : clog2(BEATS);
    localparam IW     = (clog2(N_KEEP)  < 1) ? 1 : clog2(N_KEEP);
    localparam LINE_W = 8 * HEAD_DIM;

    // ------------------------------------------------------------------
    // Reset: power-on reset (first 15 clocks) OR reset button (KEY[0])
    // ------------------------------------------------------------------
    reg [3:0] por_cnt  = 4'd0;
    reg [1:0] rst_sync = 2'b00;
    always @(posedge clk) begin
        if (por_cnt != 4'hF) por_cnt <= por_cnt + 1'b1;
        rst_sync <= {rst_sync[0], KEY[0]};
    end
    wire sys_rst = (por_cnt != 4'hF) | ~rst_sync[1];

    // ------------------------------------------------------------------
    // START button (KEY[1], active low) with debounce
    // ------------------------------------------------------------------
    reg [1:0]  k1_sync     = 2'b11;
    reg        k1_stable   = 1'b1;
    reg [19:0] db_cnt      = 20'd0;
    reg        start_pulse = 1'b0;
    always @(posedge clk) begin
        k1_sync     <= {k1_sync[0], KEY[1]};
        start_pulse <= 1'b0;
        if (k1_sync[1] == k1_stable) begin
            db_cnt <= 20'd0;
        end else if (db_cnt == DEBOUNCE_CYCLES - 1) begin
            db_cnt    <= 20'd0;
            k1_stable <= k1_sync[1];
            if (k1_stable) start_pulse <= 1'b1;      // 1 -> 0 : pressed
        end else begin
            db_cnt <= db_cnt + 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // ROMs (initialised from the .mem files)
    // ------------------------------------------------------------------
    reg [LINE_W-1:0] rom0 [0:SEQ_LEN-1];
    reg [LINE_W-1:0] rom1 [0:SEQ_LEN-1];
    reg [15:0]       idx0 [0:N_KEEP-1];
    reg [15:0]       idx1 [0:N_KEEP-1];
    initial begin
        $readmemh(KEYS0_FILE, rom0);
        $readmemh(KEYS1_FILE, rom1);
        $readmemh(IDX0_FILE,  idx0);
        $readmemh(IDX1_FILE,  idx1);
    end

    reg  [AW-1:0]     fetch_addr = {AW{1'b0}};
    reg  [15:0]       out_cnt    = 16'd0;
    reg               layer_sel  = 1'b0;
    reg  [LINE_W-1:0] q0, q1;
    reg  [15:0]       e0, e1;
    always @(posedge clk) begin                 // registered (synchronous) reads
        q0 <= rom0[fetch_addr];
        q1 <= rom1[fetch_addr];
        e0 <= idx0[out_cnt[IW-1:0]];
        e1 <= idx1[out_cnt[IW-1:0]];
    end
    wire [LINE_W-1:0] line    = layer_sel ? q1 : q0;
    wire [15:0]       exp_idx = layer_sel ? e1 : e0;

    // ------------------------------------------------------------------
    // The unit under test
    // ------------------------------------------------------------------
    localparam [2:0] ST_IDLE = 3'd0,
                     ST_RST  = 3'd1,
                     ST_PRE  = 3'd2,
                     ST_FEED = 3'd3,
                     ST_WAIT = 3'd4;

    reg  [2:0]        state    = ST_IDLE;
    reg  [3:0]        small_cnt = 4'd0;
    reg  [AW-1:0]     tok       = {AW{1'b0}};
    reg  [BW-1:0]     beat      = {BW{1'b0}};
    reg  [LINE_W-1:0] line_reg  = {LINE_W{1'b0}};
    reg  [23:0]       cyc_cnt   = 24'd0;
    reg  [15:0]       err_cnt   = 16'd0;
    reg               finished  = 1'b0;
    reg               timed_out = 1'b0;

    wire               unit_rst = sys_rst | (state == ST_RST);
    wire               feeding  = (state == ST_FEED);
    wire [8*LANES-1:0] feed_data = line_reg[beat*8*LANES +: 8*LANES];

    wire        u_in_ready;
    wire        u_out_valid;
    wire [15:0] u_out_index;
    wire        u_done;

    kv_evict_top #(
        .SEQ_LEN (SEQ_LEN),
        .HEAD_DIM(HEAD_DIM),
        .LANES   (LANES),
        .N_KEEP  (N_KEEP),
        .SCORE_W (SCORE_W)
    ) u_unit (
        .clk      (clk),
        .rst      (unit_rst),
        .in_valid (feeding),
        .in_data  (feed_data),
        .in_ready (u_in_ready),
        .out_valid(u_out_valid),
        .out_index(u_out_index),
        .done     (u_done)
    );

    // ------------------------------------------------------------------
    // Controller: feeds tokens back-to-back (prefetching the next ROM line
    // while the current one is being sent), collects and checks the outputs.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (sys_rst) begin
            state      <= ST_IDLE;
            small_cnt  <= 4'd0;
            tok        <= {AW{1'b0}};
            beat       <= {BW{1'b0}};
            fetch_addr <= {AW{1'b0}};
            line_reg   <= {LINE_W{1'b0}};
            cyc_cnt    <= 24'd0;
            out_cnt    <= 16'd0;
            err_cnt    <= 16'd0;
            finished   <= 1'b0;
            timed_out  <= 1'b0;
            layer_sel  <= 1'b0;
        end else begin
            // Check stage: runs in every state.
            if (u_out_valid) begin
                out_cnt <= out_cnt + 1'b1;
                if (u_out_index != exp_idx && err_cnt != 16'hFFFF)
                    err_cnt <= err_cnt + 1'b1;
            end

            case (state)
                ST_IDLE: begin
                    if (start_pulse) begin
                        layer_sel <= SW[0];
                        small_cnt <= 4'd0;
                        cyc_cnt   <= 24'd0;
                        out_cnt   <= 16'd0;
                        err_cnt   <= 16'd0;
                        finished  <= 1'b0;
                        timed_out <= 1'b0;
                        state     <= ST_RST;
                    end
                end

                ST_RST: begin                      // hold the unit in reset 4 clocks
                    tok        <= {AW{1'b0}};
                    beat       <= {BW{1'b0}};
                    fetch_addr <= {AW{1'b0}};
                    small_cnt  <= small_cnt + 1'b1;
                    if (small_cnt == 4'd3) begin
                        small_cnt <= 4'd0;
                        state     <= ST_PRE;
                    end
                end

                ST_PRE: begin                      // wait for ROM line 0
                    small_cnt <= small_cnt + 1'b1;
                    if (small_cnt == 4'd1) begin
                        line_reg   <= line;        // token 0
                        fetch_addr <= {{(AW-1){1'b0}}, 1'b1};   // prefetch token 1
                        tok        <= {AW{1'b0}};
                        beat       <= {BW{1'b0}};
                        state      <= ST_FEED;
                    end
                end

                ST_FEED: begin
                    cyc_cnt <= cyc_cnt + 1'b1;
                    if (beat == BEATS - 1) begin
                        beat <= {BW{1'b0}};
                        if (tok == SEQ_LEN - 1) begin
                            state <= ST_WAIT;
                        end else begin
                            tok        <= tok + 1'b1;
                            line_reg   <= line;           // next token (already fetched)
                            fetch_addr <= tok + 2'd2;     // prefetch the one after
                        end
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end

                ST_WAIT: begin
                    cyc_cnt <= cyc_cnt + 1'b1;
                    if (u_done) begin
                        finished <= 1'b1;
                        state    <= ST_IDLE;
                    end else if (cyc_cnt >= MAX_CYCLES) begin
                        timed_out <= 1'b1;
                        finished  <= 1'b1;
                        state     <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Result and display
    // ------------------------------------------------------------------
    wire pass = finished & ~timed_out & (err_cnt == 16'd0) & (out_cnt == N_KEEP);
    wire fail = finished & ~pass;

    assign LEDG = {6'b000000, (state != ST_IDLE), pass, finished};
    assign LEDR = {SW[0], err_cnt[7:0], 8'b00000000, fail};

    wire [23:0] show = SW[1] ? {8'h00, out_cnt} : cyc_cnt;

    function [6:0] seg;                         // active-low {g,f,e,d,c,b,a}
        input [3:0] d;
        begin
            case (d)
                4'h0: seg = 7'b1000000;
                4'h1: seg = 7'b1111001;
                4'h2: seg = 7'b0100100;
                4'h3: seg = 7'b0110000;
                4'h4: seg = 7'b0011001;
                4'h5: seg = 7'b0010010;
                4'h6: seg = 7'b0000010;
                4'h7: seg = 7'b1111000;
                4'h8: seg = 7'b0000000;
                4'h9: seg = 7'b0010000;
                4'hA: seg = 7'b0001000;
                4'hB: seg = 7'b0000011;
                4'hC: seg = 7'b1000110;
                4'hD: seg = 7'b0100001;
                4'hE: seg = 7'b0000110;
                default: seg = 7'b0001110;      // F
            endcase
        end
    endfunction

    assign HEX0 = seg(show[3:0]);
    assign HEX1 = seg(show[7:4]);
    assign HEX2 = seg(show[11:8]);
    assign HEX3 = seg(show[15:12]);
    assign HEX4 = seg(show[19:16]);
    assign HEX5 = seg(show[23:20]);
    assign HEX6 = 7'b1111111;                   // blank
    assign HEX7 = 7'b1111111;                   // blank

endmodule
