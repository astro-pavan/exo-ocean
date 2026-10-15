# Does WENO5 at 1° converge at the sweep's depth? 1 km, P 10 d, ΔT 30 K, warm-started from the 1° biharmonic equilibrium
# (deepeq_P_10_dT_30.0_D_1), on the local machine: 0.5° on the A400 and 1° on the CPU, at the same time.
#   julia --project=. experiments/tests/weno_validation_a400.jl
# The 0.5° run covers the first 25 yr (the resolution comparison); the 1° run continues to 100 yr to show where WENO 1° settles.
# Estimated ~3–4 days for the 0.5° run and ~2–3 days for the 1° run (±50%); ~0.6 GB. Checkpoints every 10 model years.
include(joinpath(@__DIR__, "weno_validation_common.jl"))

# Minutes per model year at 1 km: A400 from the measured centred cost (8.5) × ~2.5 for WENO, × ~8 for 0.5°; CPU (16 threads) ≈ 1.3× the A400 at 1°
cases = [case("deep", 0.5, 1000, 10, 30; years = 25, cost = 200, warm = true),
         case("deep", 1.0, 1000, 10, 30; years = 100, cost = 40, cpu = true, warm = true)]

todo, donor, summary_path = plan(cases, "a400"; n_workers = 2)
start_workers(2; threads = 16)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, donor, summary_path)
