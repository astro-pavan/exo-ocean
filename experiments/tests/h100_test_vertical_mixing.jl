# Test 1: how much do the background vertical viscosity ν_v and diffusivity κ_v set the jet? (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_vertical_mixing.jl
# 200 m, P 10 d, ΔT 30 K, WENO5, Float64. 1°: (ν_v, κ_v) = (1e-5, 1e-5), (3e-5, 3e-5), (1e-4, 1e-4), (3e-4, 3e-4), and each varied
# alone, (1e-5, 1e-4) and (1e-4, 1e-5); 0.5° at (1e-5, 1e-5) checks the resolution need at weak mixing. Each stops at deep equilibrium
# (⟨T⟩ trend < 0.005 °C/yr over 5 yr) or 40 yr; equilibration slows as κ_v falls (τ ~ H²/κ_v), so the weakest cases hit the cap.
# ~1 h on an H100 (cap 1 h, shared budget ledger). Output: $EXO_OCEAN_OUTPUT/ocean/vmix_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict(1.0 => 0.22, 0.5 => 0.6)    # H100 min per model year at 200 m (Float32; ×1.3 for Float64 in the estimate)
drift = (tol = 0.005, window = 5)
mix(res, ν, κ, est) = case("vertical", res, 200, 10, 30; prefix = "vmix", ν_v = ν, κ_v = κ, FT = Float64, years = 40, est_years = est,
                           drift, cost = cost[res])

cases = [mix(1.0, 1e-5, 1e-5, 40), mix(1.0, 3e-5, 3e-5, 30), mix(1.0, 1e-4, 1e-4, 15), mix(1.0, 3e-4, 3e-4, 10),
         mix(1.0, 1e-5, 1e-4, 15), mix(1.0, 1e-4, 1e-5, 40), mix(0.5, 1e-5, 1e-5, 40)]

budget = h100_budget("vertical_mixing", 1.0)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "vmix")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
