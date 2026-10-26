# kv_demo_top.sdc - timing constraints for the on-board demo
# The DE2-115 has a 50 MHz clock on CLOCK_50 (20 ns period).
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty

# Buttons and switches are slow, asynchronous human inputs; LEDs and the
# 7-segment displays are outputs nobody times. Leave them out of timing analysis.
set_false_path -from [get_ports {KEY[*] SW[*]}]
set_false_path -to [get_ports {LEDR[*] LEDG[*] HEX0[*] HEX1[*] HEX2[*] HEX3[*] HEX4[*] HEX5[*] HEX6[*] HEX7[*]}]
