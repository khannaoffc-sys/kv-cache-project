`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tb_kv_evict.v
// Feeds the real INT8 keys (layerN_keys_int8.mem) into kv_evict_top and checks:
//   1) every token score  == layerN_scores_int32.mem   (bit-exact)
//   2) every kept index   == layerN_indices_topk.mem   (same order, ascending)
// Run from the repo root so the default paths work, or pass +KEYS= +SCORES= +IDX=.
// Only ratio 0.5 (N_KEEP = 256) golden indices exist, so keep N_KEEP = 256.
// -----------------------------------------------------------------------------
module tb_kv_evict;

    parameter SEQ_LEN  = 512;
    parameter HEAD_DIM = 64;
    parameter LANES    = 4;
    parameter N_KEEP   = 256;
    parameter SCORE_W  = 21;

    localparam BEATS      = HEAD_DIM / LANES;
    localparam MAX_CYCLES = SEQ_LEN*BEATS + SEQ_LEN*(SEQ_LEN+8) + 2000;

    reg clk = 1'b0;
    always #5 clk = ~clk;                 // 100 MHz

    reg                  rst      = 1'b1;
    reg                  in_valid = 1'b0;
    reg  [8*LANES-1:0]   in_data  = {8*LANES{1'b0}};
    wire                 in_ready, out_valid, done;
    wire [15:0]          out_index;

    kv_evict_top #(
        .SEQ_LEN (SEQ_LEN),
        .HEAD_DIM(HEAD_DIM),
        .LANES   (LANES),
        .N_KEEP  (N_KEEP),
        .SCORE_W (SCORE_W)
    ) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_index(out_index), .done(done)
    );

    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    reg [7:0]  key_mem   [0:SEQ_LEN*HEAD_DIM-1];
    reg [31:0] exp_score [0:SEQ_LEN-1];
    reg [15:0] exp_idx   [0:N_KEEP-1];
    reg [15:0] got_idx   [0:SEQ_LEN-1];
    reg [8*256-1:0] f_keys, f_scores, f_idx;

    // Collect the kept-index stream.
    integer n_out = 0;
    always @(posedge clk) begin
        if (out_valid) begin
            if (n_out < SEQ_LEN) got_idx[n_out] = out_index;
            n_out = n_out + 1;
        end
    end

    integer t, bt, l, i;
    integer c_start, c_loaded, c_done;
    integer score_err, idx_err;
    reg [8*LANES-1:0] word;
    reg bad;

    initial begin
        score_err = 0;
        idx_err   = 0;

        if (!$value$plusargs("KEYS=%s",   f_keys))   f_keys   = "results/fpga/layer0_keys_int8.mem";
        if (!$value$plusargs("SCORES=%s", f_scores)) f_scores = "results/fpga/layer0_scores_int32.mem";
        if (!$value$plusargs("IDX=%s",    f_idx))    f_idx    = "results/fpga/layer0_indices_topk.mem";

        $readmemh(f_keys,   key_mem);
        $readmemh(f_scores, exp_score);
        $readmemh(f_idx,    exp_idx);

        bad = ((^key_mem[SEQ_LEN*HEAD_DIM-1]) === 1'bx) |
              ((^exp_score[SEQ_LEN-1])        === 1'bx) |
              ((^exp_idx[N_KEEP-1])           === 1'bx);
        if (bad) begin
            $display("RESULT: FAIL (a .mem file could not be read or is too short; check the paths)");
            $finish;
        end

        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);

        c_start = cycle;
        for (t = 0; t < SEQ_LEN; t = t + 1) begin
            for (bt = 0; bt < BEATS; bt = bt + 1) begin
                for (l = 0; l < LANES; l = l + 1)
                    word[8*l +: 8] = key_mem[t*HEAD_DIM + bt*LANES + l];
                in_data  <= word;
                in_valid <= 1'b1;
                @(posedge clk);
            end
        end
        in_valid <= 1'b0;
        c_loaded = cycle;

        @(posedge done);
        c_done = cycle;
        repeat (3) @(posedge clk);

        // 1) scores, bit-exact
        for (i = 0; i < SEQ_LEN; i = i + 1) begin
            if ((dut.score_mem[i] !== exp_score[i][SCORE_W-1:0]) ||
                ((exp_score[i] >> SCORE_W) != 0)) begin
                score_err = score_err + 1;
                if (score_err <= 5)
                    $display("  score mismatch, token %0d: hardware %0d, expected %0d",
                             i, dut.score_mem[i], exp_score[i]);
            end
        end

        // 2) kept indices
        if (n_out != N_KEEP) begin
            idx_err = idx_err + 1;
            $display("  kept-index count: hardware %0d, expected %0d", n_out, N_KEEP);
        end
        for (i = 0; i < N_KEEP; i = i + 1) begin
            if (got_idx[i] !== exp_idx[i]) begin
                idx_err = idx_err + 1;
                if (idx_err <= 5)
                    $display("  index mismatch at position %0d: hardware %0d, expected %0d",
                             i, got_idx[i], exp_idx[i]);
            end
        end

        $display("-----------------------------------------------");
        $display("SEQ_LEN=%0d HEAD_DIM=%0d LANES=%0d N_KEEP=%0d", SEQ_LEN, HEAD_DIM, LANES, N_KEEP);
        $display("Score errors : %0d of %0d", score_err, SEQ_LEN);
        $display("Index errors : %0d (kept %0d, expected %0d)", idx_err, n_out, N_KEEP);
        $display("Load phase   : %0d cycles", c_loaded - c_start);
        $display("Select phase : %0d cycles", c_done - c_loaded);
        $display("Total        : %0d cycles = about %0d us at 100 MHz",
                 c_done - c_start, (c_done - c_start) / 100);
        if (score_err == 0 && idx_err == 0)
            $display("RESULT: PASS (bit-exact match with the Python golden model)");
        else
            $display("RESULT: FAIL");
        $finish;
    end

    // Watchdog: fail instead of hanging forever.
    initial begin
        repeat (MAX_CYCLES) @(posedge clk);
        $display("RESULT: FAIL (timeout: the design never raised 'done')");
        $finish;
    end

endmodule
