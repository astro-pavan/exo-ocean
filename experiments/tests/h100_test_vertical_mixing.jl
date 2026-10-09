# Test 1: how much do the background vertical viscosity ν_v and diffusivity κ_v set the jet, down to near-molecular κ_v, and does the
# answer depend on vertical resolution? (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_vertical_mixing.jl
# 200 m, P 10 d, ΔT 30 K, WENO5, Float64; 9 levels unless "nz" (18 or 36 levels split every default cell into 2 or 4).
#   1°: (ν_v, κ_v) = (1e-5, 1e-5), (3e-5, 3e-5), (1e-4, 1e-4), (3e-4, 3e-4); one varied alone: (1e-5, 1e-4), (1e-4, 1e-5);
#       weak κ_v: (1e-4, 1e-6), (1e-4, 1e-7), (1e-7, 1e-7)   (heat's molecular diffusivity is ~1.4e-7 m²/s)
#   1° vertical resolution: 18 levels at (1e-4, 1e-4) and (1e-4, 1e-7); 36 levels at (1e-4, 1e-7)
#   0.5°: (1e-5, 1e-5)
# Each stops at deep equilibrium (⟨T⟩ trend < 0.005 °C/yr over 5 yr) or 40 yr (36 levels: 30 yr). ~2 h on an H100 (cap 2 h, shared
# budget ledger). Output: $EXO_OCEAN_OUTPUT/ocean/vmix_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict(1.0 => 0.22, 0.5 => 0.6)    # H100 min per model year at 200 m, 9 levels (Float32; ×1.3 Float64, × level factor in the estimate)
drift = (tol = 0.005, window = 5)
mix(res, ν, κ, est; nz_factor = 1, years = 40) =
    case("vertical", res, 200, 10, 30; prefix = "vmix", ν_v = ν, κ_v = κ, nz_factor, FT = Float64, years, est_years = est, drift,
         cost = cost[res])

cases = [mix(1.0, 1e-5, 1e-5, 40), mix(1.0, 3e-5, 3e-5, 30), mix(1.0, 1e-4, 1e-4, 15), mix(1.0, 3e-4, 3e-4, 10),
         mix(1.0, 1e-5, 1e-4, 15), mix(1.0, 1e-4, 1e-5, 40),
         mix(1.0, 1e-4, 1e-6, 40), mix(1.0, 1e-4, 1e-7, 40), mix(1.0, 1e-7, 1e-7, 40),
         mix(1.0, 1e-4, 1e-4, 15; nz_factor = 2), mix(1.0, 1e-4, 1e-7, 40; nz_factor = 2), mix(1.0, 1e-4, 1e-7, 30; nz_factor = 4, years = 30),
         mix(0.5, 1e-5, 1e-5, 40)]

budget = h100_budget("vertical_mixing", 2.0)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "vmix")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
