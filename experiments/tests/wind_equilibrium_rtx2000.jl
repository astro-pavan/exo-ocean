# Wind tests with CATKE at 2° (2° and 1° gave nearly the same CATKE jet at 2 bar), one chain per wind case run as three processes
# sharing the RTX 2000, each in priority order:
#   1. spin-up from rest with the wind on, to deep equilibrium (capped at 100 years), κ = 1e-4
#   2. (2 and 10 bar only) 10 years with 32 levels (stretching √1.2 halves every cell), warm-started from 1
#   3. 80 years with background κ = 1e-5, warm-started from 1 (deep adjustment is slow, so check the drift rather than trust a stop)
#   experiments/tests/wind_equilibrium_rtx2000.sh   launches the three chains in the background (or run one: julia --project=. -t 4 <this file> nowind|2|10)
# Estimated ~8–12 h, mostly the wind spin-ups, from scaled A400 rates (RTX 2000 not yet measured). Finished runs are skipped and
# cut-off runs resume on a rerun. Winds come from the cache, or are solved with $EXO_EMULATOR_PYTHON.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

case = only(ARGS)
case in ("nowind", "2", "10") || error("case must be nowind, 2 or 10 (bar), not $(case)")

base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, resolution = 2.0, n_depth = 16,
        float_type = Float64, use_GPU = true, vertical_mixing = :catke, κ_vertical = 1e-4, prandtl = 1)
wind = case == "nowind" ? (; emulator_wind = false) : (; emulator_wind = true, surface_pressure = parse(Int, case))
tag = case == "nowind" ? "nowind" : "wind$(case)bar"
equilibrium_tol = EQUILIBRIUM_TOL * (base.T_day - base.T_night)

spin_up = "windeq_$(tag)_spinup"
simulation(merge(base, wind, (; name = spin_up, simulation_time = 100 * 365days, equilibrium_tol, checkpoint_interval = 10 * 365days)),
           nothing)
if case != "nowind"
    try    # the 16 → 32 level warm start is untested, so a failure here shouldn't stop stage 3
        simulation(merge(base, wind, (; name = "windeq_$(tag)_nz32", n_depth = 32, vertical_stretching = sqrt(1.2),
                                      simulation_time = 10 * 365days)), nothing, spin_up)
    catch err
        @error "windeq_$(tag)_nz32 failed; continuing to the κ = 1e-5 run" exception = (err, catch_backtrace())
    end
end
simulation(merge(base, wind, (; name = "windeq_$(tag)_kv1e-5", κ_vertical = 1e-5, prandtl = 10, simulation_time = 80 * 365days, equilibrium_tol,
                              checkpoint_interval = 10 * 365days)), nothing, spin_up)
