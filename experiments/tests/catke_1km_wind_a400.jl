# base_case_1km_wind_a400.jl with CATKE vertical mixing (ν = κ = 1e-4 background unchanged): no wind and CLERO emulator winds at
# P0 = 10, 2 and 3 bar, each 10 years at 1°, all warm-started from the convective-adjustment no-wind spin-up base1km_spinup. Reusing it
# is fair: with no wind, 3-year 2° runs of the two schemes gave the same equatorial transport (experiments/tests/wind_mixing_test.jl).
# 10 bar runs first as its stress differs most; the 2 and 3 bar stresses are within ~20%. Uses the cached wind files.
#   EXO_EMULATOR_PYTHON=/data/pt426/ocean-venv/bin/python nice -n 19 julia --project=. -t 14 experiments/tests/catke_1km_wind_a400.jl
# Finished runs are skipped and cut-off runs resume on a rerun.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

use_GPU = get(ENV, "USE_GPU", "false") == "true"
base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64, use_GPU,
        vertical_mixing = :catke)
wind = merge(base, (; emulator_wind = true))
shared = (; low_res_start = nothing, warm_start = "base1km_spinup", prefix = "catke1km", resolution = [1.0])

sweep(wind; shared..., surface_pressure = [10])
sweep(base; shared...)
sweep(wind; shared..., surface_pressure = [2, 3])
