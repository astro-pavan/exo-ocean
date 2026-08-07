include("../src/ocean_sim.jl")
using Oceananigans.Units

# Minimal experiment set (A + B) probing the baroclinic equatorial-jet picture,
# anchored on the already-equilibrated reference run coriolis_P_10_dT_30.0_D_1.
#
#   Set A — depth sweep (is the jet baroclinic / set by the warm layer, not the
#           full ocean depth?). Vary D at fixed P = 10 d.
#   Set B — slow-rotation sweep (how does the rotating v∝c² jet connect to the
#           non-rotating v∝c flow?). Vary P at fixed D = 1 km.
#
# All at day-night contrast dT = 30 °C (T_day = 30, T_night = 0): the strongest
# clean jet, which also equilibrates fastest (τ_eq ∝ dT^-1/2). Runs stop early
# via the built-in ⟨KE⟩ convergence check in ocean_sim.jl, so `max_time` is only
# a safety cap. Run from the repo root (paths are relative to ./output).

const T_night = 0
const dT      = 30.0                 # day-night contrast; T_day = T_night + dT
const T_day   = T_night + dT
const max_time = 1000days            # cap; convergence check normally stops sooner

# Find the most-recent checkpoint for a run prefix, for warm-starting (Set B).
function latest_checkpoint(prefix)
    ckpt_dir = joinpath(directory, "ocean", "checkpoints")   # `directory` from ocean_sim.jl
    files = filter(f -> startswith(f, prefix * "_iteration") && endswith(f, ".jld2"),
                   readdir(ckpt_dir))
    isempty(files) && error("no checkpoint matching '$prefix' in $ckpt_dir")
    iters = [parse(Int, match(r"_iteration(\d+)\.jld2$", f).captures[1]) for f in files]
    return joinpath(ckpt_dir, files[argmax(iters)])
end

# ---------------------------------------------------------------------------
# Set A — depth sweep (rotating jet, baroclinic test), P = 10 d.
# n_depth is raised with D so the near-surface grid is the SAME at every depth
# (~11 m top cell, ~5 cells across the ~76 m warm layer); a fixed n_depth would
# leave deep runs under-resolving the warm layer. D = 1 km (n_depth = 16) already
# exists as coriolis_P_10_dT_30.0_D_1 and serves as the control — not re-run here.
# Different depth ⇒ different grid, so these cold-start (no warm-start possible).
# ---------------------------------------------------------------------------
const P_A = 10
const set_A = [(0.5, 13), (2.0, 20), (4.0, 24)]   # (depth [km], n_depth)

for (depth, nz) in set_A
    ocean_simulation("coriolis_P_$(P_A)_dT_$(dT)_D_$(depth)", P_A, depth * kilometer,
                     R_Earth, T_day, T_night, max_time,
                     n_lat=160, n_lon=360, n_depth=nz, use_GPU=false,
                     checkpoint_interval=356days)
end

# ---------------------------------------------------------------------------
# Set B — slow-rotation extension, D = 1 km.
# P = 3,5,10,30,50 d already exist; the non-rotating limit is OWTG_T_30_D_1.
# Add slow rotation, warm-started from the equilibrated P = 10 run (identical
# grid), so only the circulation re-adjusts rather than rebuilding the thermocline.
# ---------------------------------------------------------------------------
const depth_B = 1
const set_B = [100, 300]

donor = try
    latest_checkpoint("coriolis_P_10_dT_$(dT)_D_$(depth_B)")
catch e
    @warn "No P=10 checkpoint for warm-start ($e). Set B will cold-start (slower)."
    nothing
end

for P in set_B
    ocean_simulation("coriolis_P_$(P)_dT_$(dT)_D_$(depth_B)", P, depth_B * kilometer,
                     R_Earth, T_day, T_night, max_time,
                     n_lat=160, n_lon=360, n_depth=16, use_GPU=false,
                     pickup=donor, checkpoint_interval=356days)
end
