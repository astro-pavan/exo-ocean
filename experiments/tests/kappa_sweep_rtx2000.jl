# Sweep of the background vertical diffusivity κ at Prandtl number 10 (ν = 10κ), with CATKE, at 2° and 16 levels, for no wind and
# CLERO winds at 2 and 10 bar: one chain per wind case, run as three processes sharing the RTX 2000. Each run warm-starts from the
# nearest finished κ, the first (κ = 1e-4) from that case's windeq_*_spinup (κ = 1e-4, Pr = 1) if it exists, otherwise from rest.
# Equilibrium is tested over 20-year windows to 0.03 °C, so the mean surface heat flux must fall below ~0.2 W/m² (the 10-year,
# 0.06 °C test stopped κ = 1e-5 runs at ~0.8 W/m² while still drifting); every run is capped so a chain stays within ~48 h.
#   experiments/tests/kappa_sweep_rtx2000.sh   launches the three chains in the background (or run one: julia --project=. -t 4 <this file> nowind|2|10)
# Estimated ~25–48 h for each wind chain at ~3.5–4.5 min per model year with three runs sharing the GPU, less for no wind. Finished
# runs are skipped and cut-off runs resume on a rerun. Winds come from the cache, or are solved with $EXO_EMULATOR_PYTHON.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

case = only(ARGS)
case in ("nowind", "2", "10") || error("case must be nowind, 2 or 10 (bar), not $(case)")

base = (; rotational_period = 10, ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, resolution = 2.0, n_depth = 16,
        float_type = Float64, use_GPU = true, vertical_mixing = :catke, prandtl = 10)
wind = case == "nowind" ? (; emulator_wind = false) : (; emulator_wind = true, surface_pressure = parse(Int, case))
tag = case == "nowind" ? "nowind" : "wind$(case)bar"
convergence = (; equilibrium_tol = 0.03, equilibrium_window = 20 * 365days, checkpoint_interval = 10 * 365days)

name(κ) = run_name("kvsweep_$(tag)", (; κ_vertical = κ, prandtl = base.prandtl))
earlier = "windeq_$(tag)_spinup"
first_donor = is_finished(earlier) ? earlier : nothing

# (κ, cap in years, κ of the donor run), in priority order; the weakest κ adjusts slowest, so it runs last
plan = [(1e-4, 100, nothing), (1e-5, 200, 1e-4), (3e-5, 120, 1e-5), (3e-4, 80, 1e-4), (3e-6, 200, 1e-5)]

for (κ, years, from) in plan
    donor = !isnothing(from) && is_finished(name(from)) ? name(from) : first_donor
    max_Δt = base.prandtl * κ > 2e-3 ? 2.5hours : nothing    # explicit background viscosity: AB2 needs ν Δt / Δz² ≲ 1/4 in the 11 m top cell
    try
        simulation(merge(base, wind, convergence, (; name = name(κ), κ_vertical = κ, simulation_time = years * 365days, max_Δt)),
                   nothing, donor)
    catch err
        @error "$(name(κ)) failed; continuing with the next κ" exception = (err, catch_backtrace())
    end
end
