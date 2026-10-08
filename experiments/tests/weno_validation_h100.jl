# Is WENO5 at 1° good enough for the sweeps? Cold-start resolution checks across the sweep's P and ΔT range, for a rented H100 (≤ 4 h).
#   julia --project=. experiments/tests/weno_validation_h100.jl
# 200 m ocean, 10 model years per run (mean T within ~0.25 °C of equilibrium), two runs at a time, none started after 3.25 h:
#   baseline  P 10 d, ΔT 30 K at 1°, 0.5°, 0.25°
#   edges     P 3 d ΔT 30 K (1°, 0.5°, 0.25°: narrowest jet); P 3 d ΔT 50 K, P 10 d ΔT 1 K, P 50 d ΔT 30 K (1°, 0.5°)
#   kappa     P 10 d, ΔT 30 K at 1° with κ_h = 300 and 3000 m²/s (default 1000)
# Estimated ~2.8 h one at a time, ~2–2.5 h as run; ~3.2 GB (~1.3 GB without checkpoints). Copy back: rsync "…/ocean/wenoval_*".
include(joinpath(@__DIR__, "weno_validation_common.jl"))

# H100 minutes per model year at 200 m, measured in the jetres runs (0.25°: bih10 × 1.2 for WENO)
cost = Dict(1.0 => 0.22, 0.5 => 0.6, 0.25 => 4.4)
h100(group, res, P, dT; κ = 1e3) = case(group, res, 200, P, dT; κ, years = 10, cost = cost[res])

cases = [
    [h100("baseline", r, 10, 30) for r in (1.0, 0.5, 0.25)];
    [h100("edges", r, 3, 30) for r in (1.0, 0.5, 0.25)];
    [h100("edges", r, P, dT) for (P, dT) in ((3, 50), (10, 1), (50, 30)) for r in (1.0, 0.5)];
    [h100("kappa", 1.0, 10, 30; κ) for κ in (300, 3000)]]

todo, donor, summary_path = plan(cases, "h100"; n_workers = 2)
start_workers(2; threads = 4)
@everywhere workers() include($VALIDATION_COMMON)
run_all(todo, donor, summary_path; budget_hours = 3.25)
