# Base case (P 10 d, ΔT 30 K, 1 km): does the default 2° spin-up + 10 years reproduce a run spun up from cold, and is 1° enough or is 0.5° needed?
#   julia --project=. experiments/tests/base_case_1km.jl
# Estimated ~6.5 h on an A100 (5–9 h, mostly the cold 0.5° run). Finished runs are skipped on a rerun.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64)

# 2° spin-up to deep equilibrium (shared by both resolutions), then 10 years at each resolution
sweep(base; prefix = "base1km", resolution = [1.0, 0.5])

# From cold to deep equilibrium at each resolution, with the spin-up's stopping test so the end states are comparable
cold = (; simulation_time = 300 * 365days, equilibrium_tol = EQUILIBRIUM_TOL * base.T_day, checkpoint_interval = 10 * 365days)
sweep(merge(base, cold); low_res_start = nothing, prefix = "base1km_cold", resolution = [1.0, 0.5])
