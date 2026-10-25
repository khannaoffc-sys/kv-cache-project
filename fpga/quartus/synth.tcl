# synth.tcl  -  compile the KV-cache eviction unit for Cyclone IV E (EP4CE115F29C7)
#
# Put this file and kv_evict_top.sdc in  fpga/quartus/  (next to fpga/rtl/).
# Run from that folder:
#     quartus_sh.exe -t synth.tcl
#
# Outputs go to fpga/quartus/output_files/ (fit.summary, sta.rpt, ...).

load_package project
load_package flow

set top kv_evict_top

# Re-create the project from scratch on every run so results are reproducible.
project_new $top -overwrite

set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE115F29C7
set_global_assignment -name TOP_LEVEL_ENTITY $top

# RTL (paths are relative to this folder)
set_global_assignment -name VERILOG_FILE ../rtl/kv_sumsq_unit.v
set_global_assignment -name VERILOG_FILE ../rtl/kv_evict_top.v

# Clock constraint
set_global_assignment -name SDC_FILE kv_evict_top.sdc

set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

# Full compile: analysis & synthesis, fitter, assembler, timing analysis
execute_flow -compile

project_close
puts "DONE. Now read output_files/kv_evict_top.fit.summary and output_files/kv_evict_top.sta.rpt"
