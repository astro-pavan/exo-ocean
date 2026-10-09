# Test 2: how much does the horizontal tracer diffusivity κ_h set the jet, and does its effect shrink with resolution?
# (context: docs/LLM/h100_sensitivity_tests.md)
#   julia --project=. experiments/tests/h100_test_tracer_diffusivity.jl
# 200 m, P 10 d, ΔT 30 K, WENO5, Float64. 1°: κ_h = 0, 100, 300, 1000, 3000 m²/s (UpwindBiased(3) tracers) and WENO5 tracers with
# κ_h = 0; 0.5°: 300, 1000, 3000; 15 yr. 0.25°: 300 for 10 yr, to compare with the existing 0.25° κ_h = 1000 Float64 run
# (wenoval_0.25deg_D200_P10_dT30_f64, 10 yr). ~1.25 h on an H100 (cap 1.25 h, shared budget ledger). Output: ocean/khtest_*.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cost = Dict(1.0 => 0.22, 0.5 => 0.6, 0.25 => 3.0)
kh(res, κ; tracer = :upwind3, years = 15) = case("kappa_h", res, 200, 10, 30; prefix = "khtest", κ, tracer, FT = Float64, years,
                                                 cost = cost[res])

cases = [[kh(1.0, κ) for κ in (0, 100, 300, 1000, 3000)];
         kh(1.0, 0; tracer = :weno5);
         [kh(0.5, κ) for κ in (300, 1000, 3000)];
         kh(0.25, 300; years = 10)]

budget = h100_budget("tracer_diffusivity", 1.25)
todo, _, summary_path = plan(cases, "h100"; n_workers = 2, prefix = "khtest")
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path; budget_hours = budget)
