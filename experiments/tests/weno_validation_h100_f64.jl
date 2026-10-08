# Reruns the weno_validation_h100.jl cases that hit the Float32 WENO NaN, with a Float64 grid, for a rented H100 (~1–1.5 h).
#   julia --project=. experiments/tests/weno_validation_h100_f64.jl
# 200 m, 10 model years, two runs at a time; output names end in _f64:
#   0.25°, P 10 d, ΔT 30 K   (baseline reference)
#   0.25°, P 3 d,  ΔT 30 K   (narrow-jet reference)
#   0.5°,  P 50 d, ΔT 30 K   (completes the P 50 d 1° vs 0.5° pair)
# Estimated ~1.6 h one at a time, ~1–1.5 h as run; ~3.5 GB (~1.5 GB without checkpoints; Float64 files are twice the size). Copy back: rsync "…/ocean/wenoval_*_f64*".
include(joinpath(@__DIR__, "weno_validation_common.jl"))

# H100 minutes per model year in Float32 from the wenoval runs (0.25°: 26.6 min for 7.2 yr at P 3 d, shared GPU); ×1.3 for Float64 in estimate_minutes
cost = Dict(0.25 => 3.0, 0.5 => 0.6)
f64(res, P) = case("reference", res, 200, P, 30; years = 10, cost = cost[res], FT = Float64)

cases = [f64(0.25, 10), f64(0.25, 3), f64(0.5, 50)]

todo, donor, summary_path = plan(cases, "h100_f64"; n_workers = 2)
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, donor, summary_path; budget_hours = 3)
