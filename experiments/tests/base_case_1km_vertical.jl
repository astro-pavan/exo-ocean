# Base case (P 10 d, ΔT 30 K, 1 km): sensitivity to vertical viscosity ν_v, diffusivity κ_v and vertical resolution, each with a 2° spin-up
# then 10 years at 1°. The surface forcing no longer depends on the top cell, so the resolution runs are clean, and the Gregory equilibrium
# test no longer stops weak-mixing runs early: a spin-up that hits the 150 yr cap instead is recorded as stopped_by = "stop_time".
#   julia --project=. experiments/tests/base_case_1km_vertical.jl
# Estimated ~4 h on an A100 (weak-κ_v spin-ups assumed to run to the cap).
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64, resolution = 1.0)
spin_up = merge(SPIN_UP, (; simulation_time = 150 * 365days))

# ν_v and κ_v varied separately and together, 16 levels
sweep(base; low_res_start = spin_up, prefix = "vmix1km", ν_vertical = [1e-4, 1e-5], κ_vertical = [1e-4, 1e-5])

# 32 levels (every 16-level cell split in two) at the default and the weak κ_v, to see whether numerical mixing matters
sweep(merge(base, (; vertical_stretching = sqrt(1.2))); low_res_start = spin_up, prefix = "vmix1km", n_depth = [32], κ_vertical = [1e-4, 1e-5])
