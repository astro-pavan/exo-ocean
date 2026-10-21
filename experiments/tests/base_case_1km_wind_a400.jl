# Base case (P 10 d, ΔT 30 K, 1 km) with no wind and with CLERO emulator winds at P0 = 2, 3 and 10 bar, all four run for 10 years at 1°
# from one shared no-wind 2° spin-up: a cheap local look at how wind changes the jet and upper-ocean thermocline. The deep ocean keeps
# its no-wind equilibrium, so wind-driven deep changes need base_case_1km_wind.jl (a spin-up per pressure).
# Runs on the CPU by default, as the A400 also drives the displays; USE_GPU=true runs it on the GPU (~4 h, but the desktop stalls).
#   EXO_EMULATOR_PYTHON=/data/pt426/ocean-venv/bin/python nice -n 19 julia --project=. -t 14 experiments/tests/base_case_1km_wind_a400.jl
# Estimated ~8–10 h on 14 CPU threads. Finished runs are skipped and cut-off runs resume on a rerun.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

use_GPU = get(ENV, "USE_GPU", "false") == "true"
base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64, use_GPU)

# Solve the winds afresh so the emulator is tested too (overwrites any cached files)
foreach(P0 -> emulator_wind_file(base.rotational_period, base.T_day, base.T_night, P0; overwrite = true), (2, 3, 10))

sweep(base; prefix = "base1km", resolution = [1.0])    # the no-wind spin-up (base1km_spinup) and 10 years at 1°
sweep(merge(base, (; emulator_wind = true)); low_res_start = nothing, warm_start = "base1km_spinup",
      prefix = "wind1km_shared", surface_pressure = [2, 3, 10], resolution = [1.0])
