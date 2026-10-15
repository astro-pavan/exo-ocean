# 0.25° references for base_case_1km_resolution.jl, 8 years each from the same spin-ups (run that script first, or the spin-ups are
# redone here). Compare years 3–8 with the 1° and 0.5° runs.
#   julia --project=. experiments/tests/base_case_1km_resolution_ref.jl
# Estimated ~5 h on an A100 (~1.7 h per planet); drop a planet to fit a shorter job.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

base = (; ocean_depth = 1000.0, T_night = 0.0, n_depth = 16, float_type = Float64, simulation_time = 8 * 365days)

sweep(base; prefix = "res1km", rotational_period = [3, 5], T_day = [30.0], resolution = [0.25])
sweep(base; prefix = "res1km", rotational_period = [10], T_day = [1.0], resolution = [0.25])
