# Restarts the two 1 km WENO5 runs of weno_validation_a400.jl on the local CPU (P 10 d, ΔT 30 K), both at once:
#   1°, Float32: continues wenoval_1deg_D1000_P10_dT30_warm from its 40-yr checkpoint for 60 yr (model clock restarts at 0, so +40 yr)
#   0.5°, Float64: restarts from the 1° biharmonic equilibrium (deepeq_P_10_dT_30.0_D_1) for 25 yr; the Float32 run went NaN at 6.6 yr
#   julia --project=. experiments/tests/weno_validation_a400_restart.jl
include(joinpath(@__DIR__, "weno_validation_common.jl"))

cases = [case("deep", 1.0, 1000, 10, 30; prefix = "wenoval1km", cpu = true, years = 60, cost = 7,
              pickup = (name = "wenoval_1deg_D1000_P10_dT30_warm", res = 1.0, depth = 1000)),
         case("deep", 0.5, 1000, 10, 30; prefix = "wenoval1km", cpu = true, warm = true, FT = Float64, years = 25, cost = 35)]

todo, _, summary_path = plan(cases, "cpu"; n_workers = 2, prefix = "wenoval1km")
start_workers(2; threads = 10)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, nothing, summary_path)
