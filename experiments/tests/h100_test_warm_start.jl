# Test 4: can a fine run warm-started from a coarse equilibrium be trusted, and how coarse can the donor be?
# (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_warm_start.jl
# 500 m (13 levels, τ_eq ≈ 10 yr), P 10 d, ΔT 30 K, WENO5, Float64. Reference: 0.5° from cold to deep equilibrium (⟨T⟩ trend
# < 0.002 °C/yr over 10 yr; cap 80 yr). Donors: 4°, 2°, 1° from cold to the same stop. Then 0.5° warm-started (regridded) from each
# donor for 12 yr, compared year by year with the reference. Warm starts wait for their donor; runs are scheduled so the reference
# and the donor chain proceed side by side. ~1.4 h on an H100 (cap 1.5 h, shared budget ledger). Output: ocean/warmtest_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict(0.5 => 1.0, 1.0 => 0.35, 2.0 => 0.12, 4.0 => 0.08)    # H100 min per model year at 500 m (Float32), scaled from 200 m
drift = (tol = 0.002, window = 10)
cold(res) = case("cold", res, 500, 10, 30; prefix = "warmtest", FT = Float64, drift, years = 80, est_years = 50, cost = cost[res])
warm(donor) = case("warm", 0.5, 500, 10, 30; prefix = "warmtest", FT = Float64, years = 12, cost = cost[0.5],
                   pickup = (name = donor.name, res = donor.res, depth = 500))

donors = [cold(4.0), cold(2.0), cold(1.0)]
cases = [cold(0.5); donors; [warm(d) for d in donors]]

budget = h100_budget("warm_start", 1.5)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "warmtest")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
