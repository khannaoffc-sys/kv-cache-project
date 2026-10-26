# synth_demo.tcl - compile the on-board demo (kv_demo_top) for the Terasic DE2-115
# (Cyclone IV E, EP4CE115F29C7) and produce output_files/kv_demo_top.sof
#
# Before running:
#   1. From the project root:  python src/make_line_mem.py
#      (creates results/fpga/layer0_keys_line.mem and layer10_keys_line.mem)
#   2. fpga/quartus/pins_de2_115.tcl must exist. It holds the pin assignments and
#      must be checked against the DE2-115 user manual before use.
#
# Run from the fpga/quartus folder:
#   & "C:\altera_lite\25.1std\quartus\bin64\quartus_sh.exe" -t synth_demo.tcl

load_package project
load_package flow

set top kv_demo_top

# ---- 0. make sure everything needed is there --------------------------------
foreach f {pins_de2_115.tcl kv_demo_top.sdc ../rtl/kv_sumsq_unit.v ../rtl/kv_evict_top.v ../rtl/kv_demo_top.v} {
    if {![file exists $f]} {
        puts "ERROR: missing file $f"
        exit 1
    }
}
set mems {layer0_keys_line.mem layer10_keys_line.mem layer0_indices_topk.mem layer10_indices_topk.mem}
foreach m $mems {
    if {![file exists ../../results/fpga/$m]} {
        puts "ERROR: missing ../../results/fpga/$m (run src/make_line_mem.py first)"
        exit 1
    }
}

# ---- 1. copy the ROM files next to the project (kv_demo_top.v reads mem/...) --
file mkdir mem
foreach m $mems {
    file copy -force ../../results/fpga/$m mem/$m
}

# ---- 2. project settings -------------------------------------------------------
project_new $top -overwrite

set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE115F29C7
set_global_assignment -name TOP_LEVEL_ENTITY $top

set_global_assignment -name VERILOG_FILE ../rtl/kv_sumsq_unit.v
set_global_assignment -name VERILOG_FILE ../rtl/kv_evict_top.v
set_global_assignment -name VERILOG_FILE ../rtl/kv_demo_top.v

set_global_assignment -name SDC_FILE kv_demo_top.sdc
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

# Pins that the design does not use: make them safe inputs (usual practice on DE boards).
# The exact setting name is from memory, so a failure here only prints a note and the
# compile carries on (unused pins then keep Quartus' default, which is fine for this demo).
if {[catch {set_global_assignment -name RESERVE_ALL_UNUSED_PINS_WEAK_PULLUP "AS INPUT TRI-STATED WITH WEAK PULL-UP"} err]} {
    puts "NOTE: could not set the unused-pin option ($err); continuing with the default."
}

# ---- 3. pin assignments (clock, buttons, switches, LEDs, 7-segment displays) ----
source pins_de2_115.tcl

# ---- 4. full compile -> output_files/kv_demo_top.sof -------------------------------
execute_flow -compile

project_close
puts "DONE. Read output_files/kv_demo_top.fit.summary and output_files/kv_demo_top.sta.rpt"
puts "The file to program into the board is output_files/kv_demo_top.sof"
