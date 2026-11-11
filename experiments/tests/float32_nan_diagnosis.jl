# Is the Float32 NaN the WENO-Z weight overflow in the momentum divergence flux? CPU, 4°, 8 levels, 1 km, P = 10 d, ΔT = 30 K, 2 bar function wind,
# GM-Redi + CATKE, 5 model years per case, from rest or warm-started from a 60-day Float64 donor; plus an overflow census on the donor's state.
#   EXO_OCEAN_OUTPUT=<scratch> julia --project=. -t 8 experiments/tests/float32_nan_diagnosis.jl [case name filters]
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))
include(joinpath(@__DIR__, "float32_weno_fix.jl"))
using Oceananigans.Advection: beta_loop, global_smoothness_indicator, C★
using Oceananigans.BoundaryConditions: fill_halo_regions!

const RES = 4.0
physics = (; rotational_period = 10, T_day = 30.0, T_night = 0.0, ocean_depth = 1000.0, n_depth = 8, κ_vertical = 1e-4, prandtl = 1, use_GPU = false,
           vertical_mixing = :catke, horizontal_mixing = :gm_redi, function_wind = true, surface_pressure = 2, resolution = RES)
const DONOR = "f32diag_donor"
off = (; tke_advection = false, tke_gm_redi = false)

# (name, float type, WENO_FIX mode, options, warm start?)
const CASES = [
    ("f32_weno_warm", Float32, :off, (;), true),
    ("f32_weno_rest", Float32, :off, (;), false),
    ("f32_weno_convadj_rest", Float32, :off, (; vertical_mixing = :convective_adjustment), false),
    ("f32_upwind3_warm", Float32, :off, (; advection = :upwind3), true),
    ("f32_up3mom_warm", Float32, :off, (; advection = :upwind3_momentum), true),
    ("f32_fix_warm", Float32, :float32, (;), true),
    ("f32_fix_rest", Float32, :float32, (;), false),
    ("f32_fix_tkeon_implicit_warm", Float32, :float32, (; implicit_vertical_diffusion = true), true),
    ("f32_fix_tkeoff_implicit_warm", Float32, :float32, merge(off, (; implicit_vertical_diffusion = true)), true),
    ("f64_weno_warm", Float64, :off, (;), true),
    ("f64_fix_warm", Float64, :always, (;), true),
]

kw(p) = Base.structdiff(p, NamedTuple{(:rotational_period, :ocean_depth, :T_day, :T_night)})
cap(p) = timestep_cap(p, RES)

function make_donor()
    is_finished(DONOR) && return
    p = merge(physics, (; float_type = Float64))
    ocean_simulation(DONOR, p.rotational_period, p.ocean_depth, R_Earth, p.T_day, p.T_night, 60days; kw(p)..., max_Δt = cap(p),
                     output_interval = 30days, checkpoint_interval = 60days)
end

function run_case(name, FT, mode, options, warm)
    name = "f32diag_" * name
    is_finished(name) && return
    p = merge(physics, (; float_type = FT), options)
    WENO_FIX[] = mode
    t0 = time()
    try
        ocean_simulation(name, p.rotational_period, p.ocean_depth, R_Earth, p.T_day, p.T_night, 5 * 365days; kw(p)..., max_Δt = cap(p),
                         output_interval = 365days, wall_time_limit = 20 * 60, (warm ? (; warm_start = DONOR) : (;))...)
    finally
        WENO_FIX[] = :float32
    end
    @info "Case $(name) took $(round(time() - t0, digits = 1)) s"
end

# Largest WENO-Z α numerator (τ/(β+ϵ))² over the 5-point windows of each row of d, against Float32's overflow threshold
function max_ratio(d, dims)
    s = WENO(Float64; order = 5)
    worst, over = 0.0, 0
    n = size(d, dims)
    for idx in CartesianIndices(d)
        i = idx[dims]
        3 <= i <= n - 2 || continue
        at(o) = d[Base.setindex(Tuple(idx), i + o, dims)...]
        ψ = ((at(-2), at(-1), at(0)), (at(-1), at(0), at(1)), (at(0), at(1), at(2)))
        β = beta_loop(s, ψ)
        τ = global_smoothness_indicator(Val(3), β)
        r = maximum(τ / (b + 1e-8) for b in β)^2
        worst = max(worst, r)
        over += r * C★(s, Val(0)) > floatmax(Float32)
    end
    return worst, over
end

# Divergence smoothness δx(Ax u) + δy(Ay v) from the donor's last 3D snapshot; how often would Float32 weights overflow, with and without the scale?
function census()
    p = read_parameters(DONOR)["grid"]
    grid = ocean_grid(CPU(); n_lon = p["n_lon"], n_lat = p["n_lat"], n_depth = p["n_depth"], ocean_depth = p["ocean_depth"], planet_radius = R_Earth,
                      float_type = Float64, r = p["vertical_stretching"], max_lat = p["max_lat"])
    u, v = XFaceField(grid), YFaceField(grid)
    NCDataset(joinpath(directory, "$(DONOR).nc")) do ds
        n = size(ds["u"], 4)
        set!(u, Array(ds["u"][:, :, :, n]))
        set!(v, Array(ds["v"][:, :, :, n]))
    end
    fill_halo_regions!((u, v))
    d = Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Center}(divergence_smoothness, grid, u, v)))))
    @info "Census: |div| median $(round(sort(abs.(vec(d)))[end ÷ 2], sigdigits = 3)), max $(round(maximum(abs, d), sigdigits = 3)) m³/s; exact zeros $(count(iszero, d)) of $(length(d))"
    for (label, scale) in (("unscaled", 1.0), ("scaled", DIVERGENCE_SCALE))
        for (dir, dims) in (("x", 1), ("y", 2))
            worst, over = max_ratio(scale .* d, dims)
            @info "Census $(label) $(dir): max (τ/(β+ϵ))² = $(round(worst, sigdigits = 3)), windows over Float32 overflow: $(over)"
        end
    end
    # Float32 rounding: a stencil whose values differ by less than their rounding gives β = 0 exactly, so count those
    d32 = Float32.(d)
    @info "Census: Float32 rounding makes $(count(==(0), diff(d32; dims = 1))) neighbouring x-pairs identical"
end

function summary_line(name)
    n = "f32diag_" * name
    !isfile(parameter_file(n)) && return "$(rpad(name, 32)) not run"
    !is_finished(n) && return "$(rpad(name, 32)) unfinished (wall-time limit)"
    f = read_parameters(n)["finished"]
    status = blew_up(n) ? "NaN" : "clean, $(round(f["model_time"] / 365days, digits = 2)) yr ($(f["stopped_by"]))"
    return "$(rpad(name, 32)) $(status)"
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(joinpath(directory, "checkpoints"))
    make_donor()
    census()
    chosen = filter(c -> all(occursin(a, c[1]) for a in ARGS), CASES)
    for c in chosen
        try
            run_case(c...)
        catch err
            @error "Case $(c[1]) failed" exception = (err, catch_backtrace())
        end
    end
    @info "Summary:\n" * join(summary_line.(first.(chosen)), "\n")
end
