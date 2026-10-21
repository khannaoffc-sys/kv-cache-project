`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// kv_sumsq_unit.v
// Streaming sum-of-squares unit for one KV-cache key vector.
//
// Takes LANES signed INT8 values per clock. After HEAD_DIM/LANES beats it
// produces ONE score = sum of squares of the HEAD_DIM values, then starts the
// next token automatically.
//
// SCORE_W = 21 bits: worst case is HEAD_DIM * 128 * 128 = 2^20 (if every value
// is -128), which needs 21 bits. (With only +-127 values, 20 bits would do.)
// -----------------------------------------------------------------------------
module kv_sumsq_unit #(
    parameter HEAD_DIM = 64,
    parameter LANES    = 4,      // HEAD_DIM must be a multiple of LANES
    parameter SCORE_W  = 21
)(
    input  wire                clk,
    input  wire                rst,          // synchronous, active high
    input  wire                in_valid,
    input  wire [8*LANES-1:0]  in_data,      // lane 0 = bits [7:0], two's complement
    output reg                 score_valid,  // 1-cycle pulse, one per token
    output reg  [SCORE_W-1:0]  score
);

    localparam BEATS = HEAD_DIM / LANES;

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

    localparam BEAT_W = (clog2(BEATS) < 1) ? 1 : clog2(BEATS);

    reg [BEAT_W-1:0]   beat_cnt;
    reg [SCORE_W-1:0]  acc;

    // Sum of squares of the LANES bytes in the current beat (combinational).
    reg [SCORE_W-1:0]  beat_sum;
    reg signed [7:0]   b;
    reg [15:0]         sq;
    integer            l;

    always @* begin
        beat_sum = {SCORE_W{1'b0}};
        for (l = 0; l < LANES; l = l + 1) begin
            b        = in_data[8*l +: 8];
            sq       = b * b;               // signed x signed, 16-bit result (max 16384)
            beat_sum = beat_sum + sq;
        end
    end

    always @(posedge clk) begin
        score_valid <= 1'b0;
        if (rst) begin
            beat_cnt <= {BEAT_W{1'b0}};
            acc      <= {SCORE_W{1'b0}};
            score    <= {SCORE_W{1'b0}};
        end else if (in_valid) begin
            if (beat_cnt == BEATS-1) begin
                score       <= acc + beat_sum;
                score_valid <= 1'b1;
                acc         <= {SCORE_W{1'b0}};
                beat_cnt    <= {BEAT_W{1'b0}};
            end else begin
                acc      <= acc + beat_sum;
                beat_cnt <= beat_cnt + 1'b1;
            end
        end
    end

endmodule
