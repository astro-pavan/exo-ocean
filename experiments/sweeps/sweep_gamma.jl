# sweep_beta.jl without rotation and with ocean depth swept instead of rotation period: 72-run sweep of ΔT, depth and constant background κ, forced by
# SST relaxation only, as a warm-started tree (src/ocean/tree_sweep.jl; depth steps regrid the parent's spin-up). Each run spins up at 2° to deep
# equilibrium from its parent's spin-up, then runs 10 years at 1° (no rotation, so no jet to resolve). The launcher starts two processes: one walks
# the tree running spin-ups, the other runs main stages as their spin-ups finish. Every stage runs with ocean_simulation's defaults of Float32, WENO and
# no TKE transport.
#   scripts/launch.sh experiments/sweeps/sweep_gamma.jl                     starts both workers (or resumes the sweep)
#   julia --project=. -t 4 experiments/sweeps/sweep_gamma.jl                runs spin-up and main stage of each point in turn
#   julia --project=. experiments/sweeps/sweep_gamma.jl --plan              prints the tree and each case's levels, resolution and Δt caps
#launch gpu RTX 2000
#launch threads auto
#launch timestamp
#launch size ~10 GB
#launch conflicts sweep_alpha|sweep_beta|step_cost_benchmark|speedup_validation
#launch chain spinups --spin-ups
#launch chain mains --mains
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "tree_sweep.jl"))

const ΔT = [1.0, 3.0, 5.0, 10.0, 30.0, 50.0]    # with T_night = 0 °C, T_day = ΔT
const DEPTHS = [1000.0, 2000.0, 4000.0, 8000.0]
const KAPPA = [1e-5, 1e-4, 1e-3]
const TOP_CELL = ocean_z_faces(1000.0, 16)[2]    # 11.4 m, the 1 km sweeps' top cell

base = (; rotational_period = 0, T_night = 0.0, use_GPU = true, prandtl = 1, vertical_mixing = :catke, horizontal_mixing = :gm_redi,
        simulation_time = 10 * 365days)

# Levels that keep the 1 km sweeps' top cell at every depth with the same 1.2 stretching: 16, 20, 23, 27
n_depth(D) = round(Int, log(1 + D * 0.2 / TOP_CELL) / log(1.2))

numerics(point) = run_numerics(merge(base, point, (; n_depth = n_depth(point.ocean_depth))))
point_settings(point) = merge((; n_depth = n_depth(point.ocean_depth)), numerics(point).main)

# Warm starts test equilibrium over 10 years (20-year minimum run) instead of 20, except κ = 1e-5, which a 10-year test has stopped too early
stop_window_years(point, warm_started) = warm_started && point.κ_vertical >= 1e-4 ? 10 : 20

# The tolerance scales with the window and inversely with depth, so the heat-flux limit stays at the 1 km sweeps' (about 0.2 W/m² at ΔT = 30 K)
function spin_up(point, warm_started)
    years = stop_window_years(point, warm_started)
    tol = 0.001 * point.T_day * years / 20 * 1000 / point.ocean_depth
    return merge(SPIN_UP, (; equilibrium_tol = tol, equilibrium_window = years * 365days), numerics(point).spin_up)
end

plan_only = "--plan" in ARGS
stages = "--spin-ups" in ARGS ? :spin_ups : "--mains" in ARGS ? :mains : :all
tree_sweep(base; prefix = "norot", point_settings, spin_up, plan_only, stages,
           T_day = ΔT, ocean_depth = DEPTHS, κ_vertical = KAPPA)

if plan_only
    println("\nΔT  depth  κ  levels  resolution  main Δt cap (h)  spin-up Δt cap (h)")
    for T in ΔT, D in DEPTHS, κ in KAPPA
        n = numerics((; T_day = T, ocean_depth = D, κ_vertical = κ))
        println(T, "  ", D, "  ", κ, "  ", n_depth(D), "  ", n.main.resolution, "  ", round(n.main.max_Δt / hour, digits = 2), "  ",
                round(n.spin_up.max_Δt / hour, digits = 2))
    end
end
