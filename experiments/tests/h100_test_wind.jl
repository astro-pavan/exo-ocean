# Test 3: how much does CLERO emulator wind stress change the jet? (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_wind.jl
# P 10 d, ΔT 30 K, WENO5, Float64. 200 m at 1°, 15 yr: surface pressure P0 = 2, 3, 10 bar (area-mean stress 0.10, 0.08, 0.22 N/m²:
# denser air outweighs the weaker winds at 10 bar), and P0 = 3 bar with the stress × 0.5 and × 2 (bulk-formula uncertainty);
# 0.5° at P0 = 10 bar (strongest stress; compare wenoval_0.5deg_D200_P10_dT30).
# 1 km at 1°, P0 = 3 bar, 25 yr, warm-started from the A400 WENO run wenoval_1deg_D1000_P10_dT30_warm (its no-wind control).
# No-wind 200 m baselines: wenoval_{1,0.5}deg_D200_P10_dT30. Needs winds/P10_Td30_Tn0_P{2,3,10}_exocam.nc and the A400 checkpoint
# on the pod (the pod has no Python to solve winds). ~0.6 h on an H100 (cap 0.75 h, shared budget ledger). Output: ocean/wind_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict((1.0, 200) => 0.22, (0.5, 200) => 0.6, (1.0, 1000) => 0.5)
wind(res, depth, P0; scale = 1.0, years = 15, pickup = nothing) =
    case("wind", res, depth, 10, 30; prefix = "wind", P0, wind_scale = scale, FT = Float64, years, pickup, cost = cost[(res, depth)])

cases = [[wind(1.0, 200, P0) for P0 in (2, 3, 10)];
         [wind(1.0, 200, 3; scale) for scale in (0.5, 2.0)];
         wind(0.5, 200, 10);
         wind(1.0, 1000, 3; years = 25, pickup = (name = "wenoval_1deg_D1000_P10_dT30_warm", res = 1.0, depth = 1000))]

budget = h100_budget("wind", 0.75)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "wind")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
