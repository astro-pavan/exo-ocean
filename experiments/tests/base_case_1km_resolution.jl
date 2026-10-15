# Which horizontal resolution does each (P, ΔT) need at 1 km? A 2° spin-up per planet, then 10 years at 1° and at 0.5° from it, for the
# narrowest jets: set by fast rotation (P 3 and 5 d, ΔT 30 K) and by weak forcing (P 10 d, ΔT 1 K). P 10 d ΔT 30 K is in base_case_1km.jl.
# The 0.25° references are in base_case_1km_resolution_ref.jl, which reuses these spin-ups.
#   julia --project=. experiments/tests/base_case_1km_resolution.jl
# Estimated ~2.7 h on an A100.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

base = (; ocean_depth = 1000.0, T_night = 0.0, n_depth = 16, float_type = Float64)

sweep(base; prefix = "res1km", rotational_period = [3, 5], T_day = [30.0], resolution = [1.0, 0.5])
sweep(base; prefix = "res1km", rotational_period = [10], T_day = [1.0], resolution = [1.0, 0.5])
