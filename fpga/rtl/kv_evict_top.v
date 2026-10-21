`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// kv_evict_top.v
// KV-cache eviction unit for ONE head (mirrors KnormPress, on INT8 keys).
//
// Phase 1 (LOAD):   stream SEQ_LEN tokens in (LANES INT8 values per clock).
//                   A sum-of-squares score per token is stored in a RAM.
// Phase 2 (SELECT): for every token i, count how many tokens j have a smaller
//                   score (or an equal score and a smaller index j < i).
//                   If that count < N_KEEP, token i is KEPT.
//                   This equals a stable sort by (score, index), the same rule
//                   as the Python golden model (torch.argsort(stable=True)).
// Output:           kept token indices, in ASCENDING index order, one per
//                   clock with out_valid = 1. `done` pulses after the last one.
//
// Send exactly SEQ_LEN tokens. After `done` the unit is ready for the next head.
// Selection costs about SEQ_LEN * (SEQ_LEN + 4) clocks (simple version).
// -----------------------------------------------------------------------------
module kv_evict_top #(
    parameter SEQ_LEN  = 512,
    parameter HEAD_DIM = 64,
    parameter LANES    = 4,      // HEAD_DIM must be a multiple of LANES
    parameter N_KEEP   = 256,    // 512 * (1 - 0.5) at ratio 0.5
    parameter SCORE_W  = 21
)(
    input  wire                clk,
    input  wire                rst,         // synchronous, active high
    input  wire                in_valid,
    input  wire [8*LANES-1:0]  in_data,
    output wire                in_ready,    // high during LOAD
    output reg                 out_valid,
    output reg  [15:0]         out_index,
    output reg                 done
);

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

    localparam IDX_W = (clog2(SEQ_LEN) < 1) ? 1 : clog2(SEQ_LEN);
    localparam CNT_W = IDX_W + 1;

    localparam [2:0] S_LOAD    = 3'd0,
                     S_FETCH_I = 3'd1,
                     S_LATCH_I = 3'd2,
                     S_SCAN    = 3'd3,
                     S_DRAIN   = 3'd4,
                     S_DECIDE  = 3'd5,
                     S_DONE    = 3'd6;

    reg  [2:0]          state;
    reg  [IDX_W-1:0]    wr_idx;      // next score slot to write (LOAD)
    reg  [IDX_W-1:0]    i_idx;       // token being decided
    reg  [IDX_W-1:0]    j_idx;       // scan address
    reg  [IDX_W-1:0]    j1;          // j_idx delayed by 1 (matches RAM read latency)
    reg                 v1;          // compare stage valid
    reg  [CNT_W-1:0]    cnt;         // how many tokens rank below token i
    reg  [SCORE_W-1:0]  score_i;     // score of token i

    reg  [SCORE_W-1:0]  score_mem [0:SEQ_LEN-1];
    reg  [SCORE_W-1:0]  rd_data;
    wire [IDX_W-1:0]    rd_addr = (state == S_FETCH_I) ? i_idx : j_idx;

    wire                score_valid;
    wire [SCORE_W-1:0]  score;

    assign in_ready = (state == S_LOAD);

    kv_sumsq_unit #(
        .HEAD_DIM(HEAD_DIM),
        .LANES   (LANES),
        .SCORE_W (SCORE_W)
    ) u_sumsq (
        .clk        (clk),
        .rst        (rst),
        .in_valid   (in_valid & in_ready),
        .in_data    (in_data),
        .score_valid(score_valid),
        .score      (score)
    );

    // Score RAM: one write port (LOAD), one synchronous read port (SELECT).
    always @(posedge clk) begin
        if (score_valid) score_mem[wr_idx] <= score;
        rd_data <= score_mem[rd_addr];
    end

    // Token j is "below" token i: smaller score, or equal score and earlier index.
    wire lt = (rd_data < score_i) || ((rd_data == score_i) && (j1 < i_idx));

    always @(posedge clk) begin
        out_valid <= 1'b0;
        done      <= 1'b0;
        v1        <= 1'b0;
        if (rst) begin
            state     <= S_LOAD;
            wr_idx    <= {IDX_W{1'b0}};
            i_idx     <= {IDX_W{1'b0}};
            j_idx     <= {IDX_W{1'b0}};
            j1        <= {IDX_W{1'b0}};
            cnt       <= {CNT_W{1'b0}};
            score_i   <= {SCORE_W{1'b0}};
            out_index <= 16'd0;
        end else begin
            // Compare stage: rd_data now holds score_mem[j1].
            if (v1 && lt) cnt <= cnt + 1'b1;

            case (state)
                S_LOAD: begin
                    if (score_valid) begin
                        if (wr_idx == SEQ_LEN-1) begin
                            wr_idx <= {IDX_W{1'b0}};
                            i_idx  <= {IDX_W{1'b0}};
                            state  <= S_FETCH_I;
                        end else begin
                            wr_idx <= wr_idx + 1'b1;
                        end
                    end
                end

                S_FETCH_I: begin                 // rd_addr = i_idx this cycle
                    state <= S_LATCH_I;
                end

                S_LATCH_I: begin                 // rd_data = score_mem[i_idx]
                    score_i <= rd_data;
                    j_idx   <= {IDX_W{1'b0}};
                    cnt     <= {CNT_W{1'b0}};
                    state   <= S_SCAN;
                end

                S_SCAN: begin                    // issue read of score_mem[j_idx]
                    v1 <= 1'b1;
                    j1 <= j_idx;
                    if (j_idx == SEQ_LEN-1) state <= S_DRAIN;
                    else                    j_idx <= j_idx + 1'b1;
                end

                S_DRAIN: begin                   // last compare happens this cycle
                    state <= S_DECIDE;
                end

                S_DECIDE: begin                  // cnt is final
                    if (cnt < N_KEEP) begin
                        out_valid <= 1'b1;
                        out_index <= i_idx;
                    end
                    if (i_idx == SEQ_LEN-1) begin
                        state <= S_DONE;
                    end else begin
                        i_idx <= i_idx + 1'b1;
                        state <= S_FETCH_I;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_LOAD;             // ready for the next head
                end

                default: state <= S_LOAD;
            endcase
        end
    end

    // Simulation-only sanity checks on the parameters.
    // synthesis translate_off
    initial begin
        if (HEAD_DIM % LANES != 0)
            $display("ERROR: HEAD_DIM (%0d) must be a multiple of LANES (%0d)", HEAD_DIM, LANES);
        if (N_KEEP > SEQ_LEN)
            $display("ERROR: N_KEEP (%0d) is larger than SEQ_LEN (%0d)", N_KEEP, SEQ_LEN);
        if ((HEAD_DIM * 16384) >= (1 << SCORE_W))
            $display("ERROR: SCORE_W (%0d) is too small for HEAD_DIM %0d", SCORE_W, HEAD_DIM);
    end
    // synthesis translate_on

endmodule
