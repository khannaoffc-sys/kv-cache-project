`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tb_kv_demo_top.v  -  ModelSim testbench for the on-board demo wrapper.
// Run from the PROJECT ROOT (so the results/fpga/*.mem paths resolve):
//   vlib work
//   vlog fpga/rtl/kv_sumsq_unit.v fpga/rtl/kv_evict_top.v fpga/rtl/kv_demo_top.v fpga/tb/tb_kv_demo_top.v
//   vsim -c tb_kv_demo_top -do "run -all; quit -f"
//
// Two copies of the demo are simulated side by side:
//   dut      : correct expected indices            -> must show PASS on both layers
//   dut_bad  : layer-0 expected indices deliberately wrong (layer-10's file)
//              -> must show FAIL on layer 0, proving the checker is not vacuous
// -----------------------------------------------------------------------------
module tb_kv_demo_top;

    reg         CLOCK_50 = 1'b0;
    reg  [3:0]  KEY      = 4'b1111;      // active low, all released
    reg  [17:0] SW       = 18'd0;

    always #10 CLOCK_50 = ~CLOCK_50;     // 50 MHz

    wire [17:0] LEDR,  LEDR_b;
    wire [8:0]  LEDG,  LEDG_b;
    wire [6:0]  H0, H1, H2, H3, H4, H5, H6, H7;
    wire [6:0]  B0, B1, B2, B3, B4, B5, B6, B7;

    localparam DEB = 4;

    kv_demo_top #(
        .KEYS0_FILE("results/fpga/layer0_keys_line.mem"),
        .IDX0_FILE ("results/fpga/layer0_indices_topk.mem"),
        .KEYS1_FILE("results/fpga/layer10_keys_line.mem"),
        .IDX1_FILE ("results/fpga/layer10_indices_topk.mem"),
        .DEBOUNCE_CYCLES(DEB)
    ) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDR(LEDR), .LEDG(LEDG),
        .HEX0(H0), .HEX1(H1), .HEX2(H2), .HEX3(H3),
        .HEX4(H4), .HEX5(H5), .HEX6(H6), .HEX7(H7)
    );

    kv_demo_top #(
        .KEYS0_FILE("results/fpga/layer0_keys_line.mem"),
        .IDX0_FILE ("results/fpga/layer10_indices_topk.mem"),   // WRONG on purpose
        .KEYS1_FILE("results/fpga/layer10_keys_line.mem"),
        .IDX1_FILE ("results/fpga/layer10_indices_topk.mem"),
        .DEBOUNCE_CYCLES(DEB)
    ) dut_bad (
        .CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDR(LEDR_b), .LEDG(LEDG_b),
        .HEX0(B0), .HEX1(B1), .HEX2(B2), .HEX3(B3),
        .HEX4(B4), .HEX5(B5), .HEX6(B6), .HEX7(B7)
    );

    integer errors = 0;
    integer waited;

    task check;
        input cond;
        input [8*60-1:0] msg;
        begin
            if (cond) $display("  [PASS] %0s", msg);
            else begin
                $display("  [FAIL] %0s", msg);
                errors = errors + 1;
            end
        end
    endtask

    task press_reset;
        begin
            KEY[0] = 1'b0; repeat (10) @(posedge CLOCK_50);
            KEY[0] = 1'b1; repeat (30) @(posedge CLOCK_50);
        end
    endtask

    task press_start_and_wait;
        begin
            KEY[1] = 1'b0; repeat (DEB + 10) @(posedge CLOCK_50);
            KEY[1] = 1'b1; repeat (DEB + 10) @(posedge CLOCK_50);
            repeat (200) @(posedge CLOCK_50);          // let the old DONE clear
            waited = 0;
            while (!LEDG[0] && waited < 2000000) begin
                @(posedge CLOCK_50);
                waited = waited + 1;
            end
            repeat (20) @(posedge CLOCK_50);
        end
    endtask

    initial begin
        $display("=== kv_demo_top testbench ===");
        press_reset;

        // ---------------- layer 0 ----------------
        SW[0] = 1'b0;
        $display("Layer 0:");
        press_start_and_wait;
        $display("  cycles = %0d, outputs = %0d, errors(dut) = %0d, errors(bad) = %0d",
                 dut.cyc_cnt, dut.out_cnt, dut.err_cnt, dut_bad.err_cnt);
        check(LEDG[0] == 1'b1,   "dut: DONE LED on");
        check(LEDG[1] == 1'b1,   "dut: PASS LED on");
        check(LEDR[0] == 1'b0,   "dut: FAIL LED off");
        check(dut.out_cnt == 256, "dut: exactly 256 outputs");
        // 7-segment: cycle count 272,387 = 0x042803 on HEX5..HEX0 (active-low segments)
        check({H5,H4,H3,H2,H1,H0} == {7'b1000000,7'b0011001,7'b0100100,7'b0000000,7'b1000000,7'b0110000},
              "dut: 7-segment shows 042803");
        SW[1] = 1'b1;  repeat (5) @(posedge CLOCK_50);
        check({H5,H4,H3,H2,H1,H0} == {7'b1000000,7'b1000000,7'b1000000,7'b1111001,7'b1000000,7'b1000000},
              "dut: 7-segment shows 000100 (256 outputs) with SW[1]");
        SW[1] = 1'b0;  repeat (5) @(posedge CLOCK_50);
        check(dut_bad.finished == 1'b1, "bad: finished");
        check(LEDG_b[1] == 1'b0, "bad: PASS LED off (wrong expectations caught)");
        check(LEDR_b[0] == 1'b1, "bad: FAIL LED on");

        // ---------------- layer 10 ----------------
        SW[0] = 1'b1;
        $display("Layer 10:");
        press_start_and_wait;
        $display("  cycles = %0d, outputs = %0d, errors(dut) = %0d, errors(bad) = %0d",
                 dut.cyc_cnt, dut.out_cnt, dut.err_cnt, dut_bad.err_cnt);
        check(LEDG[1] == 1'b1,   "dut: PASS LED on");
        check(LEDR[0] == 1'b0,   "dut: FAIL LED off");
        check(dut.out_cnt == 256, "dut: exactly 256 outputs");
        check(LEDG_b[1] == 1'b1, "bad: PASS on layer 10 (its layer-10 files are correct)");

        // ---------------- back to layer 0, run again after a reset ----------------
        press_reset;
        SW[0] = 1'b0;
        $display("Layer 0 again, after a reset:");
        press_start_and_wait;
        check(LEDG[1] == 1'b1,   "dut: PASS LED on again");

        if (errors == 0) $display("\nALL PASS  (wrapper verified in simulation)");
        else             $display("\nFAILED: %0d check(s) failed", errors);
        $finish;
    end

    // global watchdog
    initial begin
        #200000000;                       // 200 ms of simulated time
        $display("TIMEOUT: testbench ran too long");
        $finish;
    end

endmodule
