# Test 5: which grid does each rotation period and temperature contrast need? (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_grid_requirements.jl
# 200 m, WENO5, Float64, 10 yr. Fills the gaps in the wenoval set (P 3/10/50 d at ΔT 30 K, P 3 d ΔT 50 K, P 10 d ΔT 1 K):
#   P 5 d,  ΔT 30 K at 1°, 0.5°, 0.25°   (between the converged P 10 d and unconverged P 3 d)
#   P 10 d, ΔT 10 K at 1°, 0.5°, 0.25°   (intermediate forcing)
#   P 10 d, ΔT 1 K  at 0.25°             (completes the weak-forcing 1°/0.5° pair)
#   P 30 d, ΔT 30 K at 1°, 0.5°          (slow rotation, broad jet)
# ~1.8 h on an H100 (cap 2 h, shared budget ledger). Output: ocean/gridtest_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict(1.0 => 0.22, 0.5 => 0.6, 0.25 => 3.0)
grid_case(res, P, dT) = case("grid", res, 200, P, dT; prefix = "gridtest", FT = Float64, years = 10, cost = cost[res])

cases = [[grid_case(r, 5, 30) for r in (1.0, 0.5, 0.25)];
         [grid_case(r, 10, 10) for r in (1.0, 0.5, 0.25)];
         grid_case(0.25, 10, 1);
         [grid_case(r, 30, 30) for r in (1.0, 0.5)]]

budget = h100_budget("grid_requirements", 2.0)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "gridtest")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
