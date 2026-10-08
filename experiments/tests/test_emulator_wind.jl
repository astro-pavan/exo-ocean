# Ocean forced by CLERO-emulated winds (ocean_simulation(...; emulator_wind = true, surface_pressure = P0)). The wind file is solved and
# cached automatically (Python env: $EXO_EMULATOR_PYTHON, default /data/pt426/ocean-venv/bin/python).
#   SHORT_TEST=1 julia experiments/tests/test_emulator_wind.jl   # coarse CPU checks
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))
using Oceananigans.Units

const P_rot, T_day, T_night, P0 = 10, 30.0, 0.0, 3
const short = get(ENV, "SHORT_TEST", "0") == "1"
const coarse = (n_lon = 90, n_lat = 40, n_depth = 8, use_GPU = false)

expect_error(f, label) = try
    f(); error("expected an error: $(label)")
catch err
    err isa ErrorException && startswith(err.msg, "expected an error") && rethrow()
    @info "OK, $(label) raised: $(first(sprint(showerror, err), 300))"
end

if short
    expect_error("missing surface_pressure") do
        ocean_simulation("x", P_rot, 1kilometer, R_Earth, T_day, T_night, 1days; coarse..., emulator_wind = true)
    end
    expect_error("wind_field together with emulator_wind") do
        ocean_simulation("x", P_rot, 1kilometer, R_Earth, T_day, T_night, 1days; coarse..., emulator_wind = true, surface_pressure = P0,
                         wind_field = (λ, φ, t) -> (0.0, 0.0))
    end
    expect_error("no emulator planet at 1 bar") do
        ocean_simulation("x", P_rot, 1kilometer, R_Earth, T_day, T_night, 1days; coarse..., emulator_wind = true, surface_pressure = 1)
    end
    # fresh solve (or cached file) and a short forced run; then an identical call that must reuse the cached wind file
    ocean_simulation("test_emuwind_short", P_rot, 1kilometer, R_Earth, T_day, T_night, 5days; coarse...,
                     output_interval = 1days, emulator_wind = true, surface_pressure = P0)
    ocean_simulation("test_emuwind_cached", P_rot, 1kilometer, R_Earth, T_day, T_night, 1days; coarse...,
                     output_interval = 1days, emulator_wind = true, surface_pressure = P0)
else
    ocean_simulation("emuwind_P_$(P_rot)_dT_$(T_day - T_night)_P0_$(P0)_D_1", P_rot, 1kilometer, R_Earth, T_day, T_night, 2 * 365days;
                     n_lon = 360, n_lat = 160, n_depth = 16, checkpoint_interval = 365days, emulator_wind = true, surface_pressure = P0)
end
