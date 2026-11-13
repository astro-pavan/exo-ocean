# 432-run sweep of ΔT, rotation period, surface pressure and constant background κ for a 1 km ocean with the closed-form wind (src/ocean/pattern_wind.jl),
# as a warm-started tree (src/ocean/tree_sweep.jl). Each run spins up at 2° to deep equilibrium from its parent's spin-up (stop test over 20 years, or 10
# for warm starts with κ ≥ 1e-4), then runs 10 years at the resolution the jet needs. The launcher starts two processes: one walks the tree running
# spin-ups, the other runs main stages as their spin-ups finish, so the GPU never waits for a spin-up. Every stage runs with ocean_simulation's defaults of
# Float32, WENO and no TKE transport, validated against Float64 in docs/LLM/simulation_speedups.md §12.6–12.8.
#   scripts/launch.sh experiments/sweeps/single_depth_sweep_constant_kappa.jl                     starts both workers (or resumes the sweep)
#   julia --project=. -t 4 experiments/sweeps/single_depth_sweep_constant_kappa.jl                runs spin-up and main stage of each point in turn
#   julia --project=. experiments/sweeps/single_depth_sweep_constant_kappa.jl --plan              prints the tree and each case's resolution and Δt caps
#launch gpu RTX 2000
#launch threads auto
#launch size ~40 GB
#launch conflicts step_cost_benchmark|speedup_validation
#launch chain spinups --spin-ups
#launch chain mains --mains
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "tree_sweep.jl"))

const DEPTH = 1000.0
const ΔT = [1.0, 3.0, 5.0, 10.0, 30.0, 50.0]    # with T_night = 0 °C, T_day = ΔT
const P_ROT = [3, 5, 10, 30, 50, 100]
const P_SURF = [0.5, 1.0, 2.0, 4.0]
const KAPPA = [1e-5, 1e-4, 1e-3]

base = (; ocean_depth = DEPTH, T_night = 0.0, n_depth = 16, use_GPU = true, prandtl = 1, vertical_mixing = :catke, horizontal_mixing = :gm_redi,
        simulation_time = 10 * 365days, function_wind = true)

numerics(point) = run_numerics(merge(base, point))
point_settings(point) = numerics(point).main

# Warm starts test equilibrium over 10 years (20-year minimum run) instead of 20, except κ = 1e-5, which a 10-year test has stopped too early
stop_window_years(point, warm_started) = warm_started && point.κ_vertical >= 1e-4 ? 10 : 20

# The tolerance scales with the window, so the heat-flux limit stays the same (about 0.2 W/m² at ΔT = 30 K)
function spin_up(point, warm_started)
    years = stop_window_years(point, warm_started)
    return merge(SPIN_UP, (; equilibrium_tol = 0.001 * point.T_day * years / 20, equilibrium_window = years * 365days), numerics(point).spin_up)
end

plan_only = "--plan" in ARGS
stages = "--spin-ups" in ARGS ? :spin_ups : "--mains" in ARGS ? :mains : :all
const PREDICTOR_AXES = (:T_day, :κ_vertical)    # offline tests on finished runs cut the start error along these, but not along rotation period
tree_sweep(base; prefix = "constkappa1km", point_settings, spin_up, plan_only, stages, predictor_axes = PREDICTOR_AXES,
           T_day = ΔT, rotational_period = P_ROT, surface_pressure = P_SURF, κ_vertical = KAPPA)

if plan_only
    println("\nΔT  P_rot  κ  resolution  main Δt cap (h)  spin-up Δt cap (h)")
    for T in ΔT, P in P_ROT, κ in KAPPA
        n = numerics((; T_day = T, rotational_period = P, κ_vertical = κ))
        println(T, "  ", P, "  ", κ, "  ", n.main.resolution, "  ", round(n.main.max_Δt / hour, digits = 2), "  ", round(n.spin_up.max_Δt / hour, digits = 2))
    end
end
