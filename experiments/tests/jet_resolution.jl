# Jet formation vs resolution and dissipation scheme (200 m ocean, P = 10 d, ΔT = 30 K, cold start). One run per call:
#   julia experiments/tests/jet_resolution.jl <scheme> <resolution_deg> [years]     (JETRES_CPU=1 to run on the CPU)
# The full matrix, benchmark and smoke modes are driven by experiments/tests/jet_resolution.sh.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))
using Oceananigans.Units

const SCHEMES = Dict(
    "bih10"    => (momentum_advection = :centered, τ_biharmonic = 10days,  tracer_advection = :upwind3, κ_horizontal = 1e3),
    "bih100"   => (momentum_advection = :centered, τ_biharmonic = 100days, tracer_advection = :upwind3, κ_horizontal = 1e3),
    "weno9"    => (momentum_advection = :weno9,    τ_biharmonic = nothing, tracer_advection = :upwind3, κ_horizontal = 1e3),
    "weno5"    => (momentum_advection = :weno5,    τ_biharmonic = nothing, tracer_advection = :upwind3, κ_horizontal = 1e3),
    "weno9_tr" => (momentum_advection = :weno9,    τ_biharmonic = nothing, tracer_advection = :weno5,   κ_horizontal = 0.0))

scheme = ARGS[1]
haskey(SCHEMES, scheme) || error("unknown scheme $(scheme); choose from $(join(sort(collect(keys(SCHEMES))), ", "))")
res = parse(Float64, ARGS[2])
years = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5.0
kw = SCHEMES[scheme]

duration = years * 365days
cap = 3hours * res  # ≈ 2× the CFL-limited Δt at this resolution, so barotropic substeps aren't oversized
max_Δt = isnothing(kw.τ_biharmonic) ? cap : min(kw.τ_biharmonic / 100, cap)

ocean_simulation("jetres_$(scheme)_$(res)deg_D200", 10, 200meters, R_Earth, 30.0, 0.0, duration;
                 n_lon = round(Int, 360 / res), n_lat = round(Int, 160 / res), n_depth = 9,
                 use_GPU = get(ENV, "JETRES_CPU", "0") != "1",
                 kw..., max_Δt,
                 output_interval = duration,                       # 3D snapshots at the start and the end only
                 zonal_mean_interval = min(36.5days, duration / 10),
                 checkpoint_interval = duration)                   # final state, for extending a run later
