# Ocean forced by CLERO-emulated winds for a planet with surface pressure P0. Generate the wind file first (repo Python env; adjust the path if EXO_OCEAN_OUTPUT is set):
#   python src/climate/climate_emulator.py solve --P_rot 10 --T_day 30 --T_night 0 --P0 3 --out output/winds/P10_Td30_Tn0_P3.nc
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "emulator_wind.jl"))
using Oceananigans.Units

const P_rot, T_day, T_night, P0 = 10, 30.0, 0.0, 3
const wind_file = joinpath(directory, "winds", "P$(P_rot)_Td$(round(Int, T_day))_Tn$(round(Int, T_night))_P$(P0).nc")
const short = get(ENV, "SHORT_TEST", "0") == "1"  # coarse CPU checks: 5-day wind-only sign test + 60-day forced run

wind = grid -> emulator_wind_stress(wind_file, grid; P_rot)

if short
    ocean_simulation("test_emuwind_signcheck", P_rot, 1kilometer, R_Earth, 0.0, 0.0, 5days;
                     n_lon=90, n_lat=40, n_depth=8, use_GPU=false, output_interval=1days, wind_stress=wind)
    ocean_simulation("test_emuwind_short", P_rot, 1kilometer, R_Earth, T_day, T_night, 60days;
                     n_lon=90, n_lat=40, n_depth=8, use_GPU=false, output_interval=5days, wind_stress=wind)
else
    ocean_simulation("emuwind_P_$(P_rot)_dT_$(T_day - T_night)_P0_$(P0)_D_1", P_rot, 1kilometer, R_Earth, T_day, T_night, 2*365days;
                     n_lon=360, n_lat=160, n_depth=16, checkpoint_interval=365days, wind_stress=wind)
end
