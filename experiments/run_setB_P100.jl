include("../src/ocean_sim.jl")
using Oceananigans.Units

# One-off: the first Set B run (P = 100 d, dT = 30, D = 1 km), warm-started from
# the equilibrated P = 10 checkpoint — a smoke test that the pickup path works
# before committing the whole set_AB_sweep.jl. Run from the repo root.

const T_night = 0
const dT      = 30.0
const T_day   = T_night + dT

function latest_checkpoint(prefix)
    ckpt_dir = joinpath(directory, "ocean", "checkpoints")
    files = filter(f -> startswith(f, prefix * "_iteration") && endswith(f, ".jld2"),
                   readdir(ckpt_dir))
    isempty(files) && error("no checkpoint matching '$prefix' in $ckpt_dir")
    iters = [parse(Int, match(r"_iteration(\d+)\.jld2$", f).captures[1]) for f in files]
    return joinpath(ckpt_dir, files[argmax(iters)])
end

donor = latest_checkpoint("coriolis_P_10_dT_$(dT)_D_1")
@info "WARMSTART_DONOR = $donor"

ocean_simulation("coriolis_P_100_dT_$(dT)_D_1", 100, 1 * kilometer, R_Earth,
                 T_day, T_night, 1000days,
                 n_lat=160, n_lon=360, n_depth=16, use_GPU=false,
                 pickup=donor, checkpoint_interval=356days)
