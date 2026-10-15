# Base case (P 10 d, ΔT 30 K, 1 km) with no wind and with CLERO emulator winds at P0 = 2, 3 and 10 bar, each with the default 2° spin-up
# (wind on in both stages) then 10 years at 1°, plus 0.5° at 10 bar (the strongest stress) to check the resolution choice with wind.
#   julia --project=. experiments/tests/base_case_1km_wind.jl
# Needs the emulator's Python env (`$EXO_EMULATOR_PYTHON`, with `clero`). Estimated ~2.5 h on an A100 (up to ~3.5 h).
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64)

# Solve the winds afresh so the emulator is tested too (overwrites any cached files)
foreach(P0 -> emulator_wind_file(base.rotational_period, base.T_day, base.T_night, P0; overwrite = true), (2, 3, 10))

sweep(base; prefix = "base1km", resolution = [1.0])    # no wind: the same run as in base_case_1km.jl, so skipped if that ran
sweep(merge(base, (; emulator_wind = true)); prefix = "wind1km", surface_pressure = [2, 3, 10], resolution = [1.0])
sweep(merge(base, (; emulator_wind = true)); prefix = "wind1km", surface_pressure = [10], resolution = [0.5])    # reuses the 10 bar spin-up
