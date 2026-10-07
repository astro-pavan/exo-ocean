# Integrate a tidally locked ocean to deep thermal equilibrium (stops when the trend of volume-mean T falls below a tolerance).
#   julia experiments/tests/deep_equilibrium.jl <depth_km> [options]
# Options:
#   --res DEG              horizontal resolution in degrees (default 1)
#   --cpu                  run on the CPU (set threads with julia -t N)
#   --P DAYS, --dT K       rotation period and day-night contrast (default 10 d, 30 K; T_night = 0)
#   --tau-biharmonic DAYS  grid-scale damping time of the biharmonic viscosity (default 10)
#   --max-dt-hours H       timestep cap (default from ocean_simulation)
#   --pickup RUN|FILE      warm start from a checkpoint file, or the latest checkpoint of a run name
#   --pickup-depth KM, --pickup-res DEG   donor grid if it differs (levels follow the same rule as here)
#   --drift-tol X          stop tolerance in °C/yr (default ≈ 0.02 °C / τ_eq, τ_eq ≈ 9.5 yr × (H / 500 m)^1.4)
#   --max-years Y          cap on model years (default 1000)
#   --no-3d                no 3D snapshots (zonal means and checkpoints only)
#   --name NAME            override the run name
#   --dry-run              print the configuration and exit
#
# Runs made with the earlier one-off scripts (CFL 0.1 then; the default is now 0.2):
#   deepeq_P_10_dT_30.0_D_0.5                  0.5
#   deepeq_P_10_dT_30.0_D_1                    1 --drift-tol 6e-4
#   deepeq_P_10_dT_30.0_D_2_from1km            2 --pickup deepeq_P_10_dT_30.0_D_1 --pickup-depth 1 --name deepeq_P_10_dT_30.0_D_2_from1km
#   deepeq2deg_P_10_dT_30.0_D_{0.5,1.0}        {0.5,1.0} --res 2 [--drift-tol 6e-4 for 1 km]
#   deepeq4deg_P_10_dT_30.0_D_{0.5,1.0,2.0}    {0.5,1.0,2.0} --res 4 --cpu [--drift-tol 6e-4 for 1 km]
#   deepeq4deg_tb{100,300}_P_10_dT_30.0_D_1.0  1.0 --res 4 --cpu --tau-biharmonic {100,300} --max-dt-hours 2.4 --drift-tol 6e-4
#   warmfinish_from{2,4}deg_P_10_dT_30.0_D_0.5 0.5 --cpu --pickup deepeq{2,4}deg_P_10_dT_30.0_D_0.5 --pickup-depth 0.5 --pickup-res {2,4} --no-3d --name warmfinish_from{2,4}deg_P_10_dT_30.0_D_0.5
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))
using Oceananigans.Units

isempty(ARGS) && error("usage: julia experiments/tests/deep_equilibrium.jl <depth_km> [options]; see the header of this file")
depth_str = ARGS[1]
depth = parse(Float64, depth_str) * kilometer
opts = Dict{String, String}()
flags = Set{String}()
let i = 2
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--cpu", "--no-3d", "--dry-run")
            push!(flags, a); i += 1
        else
            startswith(a, "--") && i < length(ARGS) || error("bad option $(a)")
            opts[a] = ARGS[i + 1]; i += 2
        end
    end
end
opt(key, default) = haskey(opts, key) ? parse(Float64, opts[key]) : default
tidy(x) = isinteger(x) ? Int(x) : x

# Smallest number of levels whose top cell is at most 11.5 m (9, 13, 16, 20, 24, 25 levels for 0.2, 0.5, 1, 2, 4, 5 km)
levels_for(H) = findfirst(n -> last(ocean_z_faces(H, n)) <= 11.5, 1:60)

res = opt("--res", 1.0)
P = tidy(opt("--P", 10.0))
dT = opt("--dT", 30.0)
τ_bih = opt("--tau-biharmonic", 10.0)
n_depth = levels_for(depth)
τ_eq = max(2.0, 9.5 * (depth / 500)^1.4)                          # years, from the measured τ ∝ H^1.4
drift_tol = opt("--drift-tol", round(0.02 / τ_eq, sigdigits = 1))
drift_window = (depth <= 500 ? 10 : 20) * 365days

pickup, pickup_grid, from = nothing, nothing, ""
if haskey(opts, "--pickup")
    p = opts["--pickup"]
    if isfile(p)
        pickup = p
    else
        dir = joinpath(directory, "ocean", "checkpoints")
        files = filter(f -> startswith(f, p * "_iteration") && endswith(f, ".jld2"), readdir(dir))
        isempty(files) && error("no checkpoint for run $(p) in $(dir)")
        pickup = joinpath(dir, files[argmax([parse(Int, match(r"_iteration(\d+)\.jld2$", f)[1]) for f in files])])
    end
    donor_depth = opt("--pickup-depth", depth / kilometer) * kilometer
    donor_res = opt("--pickup-res", res)
    if donor_depth != depth || donor_res != res
        pickup_grid = (n_lon = round(Int, 360 / donor_res), n_lat = round(Int, 160 / donor_res),
                       n_depth = levels_for(donor_depth), ocean_depth = donor_depth)
    end
    from = "_from_" * replace(basename(pickup), r"_iteration\d+\.jld2$" => "")
end

prefix = "deepeq" * (res == 1 ? "" : "$(tidy(res))deg") * (τ_bih == 10 ? "" : "_tb$(tidy(τ_bih))")
name = get(opts, "--name", "$(prefix)_P_$(P)_dT_$(dT)_D_$(depth_str)$(from)")

kwargs = (n_lon = round(Int, 360 / res), n_lat = round(Int, 160 / res), n_depth,
          use_GPU = !("--cpu" in flags), τ_biharmonic = τ_bih * days,
          pickup, pickup_grid,
          T_drift_tol = drift_tol, T_drift_window = drift_window,
          output_interval = "--no-3d" in flags ? nothing : 10 * 365days,
          zonal_mean_interval = 73days, checkpoint_interval = 10 * 365days)
haskey(opts, "--max-dt-hours") && (kwargs = merge(kwargs, (; max_Δt = opt("--max-dt-hours", 0.0) * hours)))

@info "Deep-equilibrium run $(name): depth $(depth) m, $(n_depth) levels, $(res)°, P = $(P) d, ΔT = $(dT) K, " *
      "stop below $(drift_tol) °C/yr over $(drift_window / 365days) yr (τ_eq ≈ $(round(τ_eq, digits=1)) yr)" *
      (isnothing(pickup) ? "" : ", warm start from $(pickup)" * (isnothing(pickup_grid) ? "" : " regridded from $(pickup_grid)"))
"--dry-run" in flags && (println(name, "\n", kwargs); exit())

ocean_simulation(name, P, depth, R_Earth, dT, 0.0, opt("--max-years", 1000.0) * 365days; kwargs...)
