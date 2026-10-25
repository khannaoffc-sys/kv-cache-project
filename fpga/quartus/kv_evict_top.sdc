# kv_evict_top.sdc
# 10 ns clock = 100 MHz target. Only the clock is constrained, so the Fmax
# summary reflects register-to-register paths inside the design (the input and
# output pins are left unconstrained on purpose; this is a standalone module).
create_clock -name clk -period 10.000 [get_ports clk]
derive_clock_uncertainty
