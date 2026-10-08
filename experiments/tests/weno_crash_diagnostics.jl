# Why do the WENO5 runs go NaN on the H100? Reruns the crashed wenoval cases exactly, watches every field at every step around the
# known crash, and records where the first NaN appears and how the fields grew before it. ~1–1.5 h on an H100.
#   julia --project=. experiments/tests/weno_crash_diagnostics.jl            (optional: --only name1,name2)
# Cases (200 m, ΔT 30 K, WENO5 momentum, settings identical to weno_validation_h100.jl):
#   early_f32     0.25°, P 10 d   repeat of the crash at iteration 200–300 (velocities ~1e-5 m/s); watched from iteration 0
#   early_f64     the same with a Float64 grid: does single precision cause it?
#   early_cpu     the same on the CPU: is it specific to the GPU?
#   p50_f32       0.5°, P 50 d    repeat of the crash at iteration 33,500–33,600 (3.9 yr); watched over 33,000–34,000
#   p50_f64       the same with a Float64 grid, to iteration 50,000
#   p50_bih300    the same plus a weak biharmonic viscosity (τ = 300 d), to iteration 50,000: does grid-scale damping prevent it?
#   p3_f32        0.25°, P 3 d    repeat of the crash at iteration 88,000–88,100 (7.2 yr); watched over 87,600–88,200
# Output in $EXO_OCEAN_OUTPUT/ocean/diag/: <case>.log, <case>_history.csv (per-step max |field|, its grid index, NaN count),
# <case>_blowup.nc (all fields in a ±20-cell box around the first NaN, at the first bad step and the step before), summary.txt.
include(joinpath(@__DIR__, "weno_validation_common.jl"))
using NCDatasets

const DIAG_FT = Ref{DataType}(Float32)
const DIAG_HOOK = Ref{Any}(nothing)

# Same grid as ocean_sim.jl with a selectable float type (overrides the Float32 version for this script only)
function ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo = (3, 3, 3))
    z_faces, _ = ocean_z_faces(ocean_depth, n_depth)
    return LatitudeLongitudeGrid(arch, DIAG_FT[]; size = (n_lon, n_lat, n_depth), halo, longitude = (-180, 180),
                                 latitude = (-80, 80), z = z_faces, radius = planet_radius, topology = (Periodic, Bounded, Bounded))
end

# Replaces the output writers: installs the watch, plus no-op callbacks on the original output schedules, because
# Oceananigans shortens Δt to land on scheduled times and the rerun must take exactly the same steps as the H100 run
add_output_writers!(simulation, name; kwargs...) = DIAG_HOOK[](simulation)

mutable struct Watch
    from::Int
    to::Int
    fields::Any
    prev::Any
    rows::Vector{String}
    blowup::Any
end

function watch!(sim, W)
    it = iteration(sim)
    W.from <= it <= W.to || return nothing
    cur = map(f -> Array(interior(f)), W.fields)
    row = [string(it), @sprintf("%.5f", time(sim) / 86400), @sprintf("%.3f", sim.Δt / 60)]
    bad = Symbol[]
    for (k, a) in pairs(cur)
        n = count(!isfinite, a)
        m, I = findmax(x -> isfinite(x) ? abs(x) : zero(x), a)
        push!(row, @sprintf("%.6g", m), join(Tuple(I), " "), string(n))
        n > 0 && push!(bad, k)
    end
    push!(W.rows, join(row, ","))
    if !isempty(bad)
        W.blowup = (; iteration = it, bad, cur, prev = W.prev)
        sim.running = false
    end
    W.prev = cur
    return nothing
end

# (λ, φ, z) of grid index I of field f; z = 0 for 2D fields
function locate(f, I)
    ℓ = Oceananigans.Fields.location(f)
    λ = Array(λnodes(f.grid, ℓ[1]()))[I[1]]
    φ = Array(φnodes(f.grid, ℓ[2]()))[I[2]]
    z = ℓ[3] === Nothing || size(f, 3) == 1 ? 0.0 : Array(znodes(f.grid, ℓ[3]()))[I[3]]
    return round(Float64(λ), digits = 2), round(Float64(φ), digits = 2), round(Float64(z), digits = 1)
end

function save_blowup(path, W)
    b = W.blowup
    first_bad = findfirst(!isfinite, b.cur[b.bad[1]])
    i0, j0 = first_bad[1], first_bad[2]
    NCDataset(path, "c") do ds
        ds.attrib["iteration"] = b.iteration
        ds.attrib["first_bad_fields"] = join(string.(b.bad), ",")
        for (k, a) in pairs(b.cur)
            f = W.fields[k]
            Nx = size(a, 1)
            is = [mod1(i, Nx) for i in i0-20:i0+20]                    # periodic in longitude
            js = max(1, j0 - 20):min(size(a, 2), j0 + 20)
            dims = ("$(k)_i", "$(k)_j", "$(k)_k")
            for (d, n) in zip(dims, (length(is), length(js), size(a, 3)))
                defDim(ds, d, n)
            end
            ℓ = Oceananigans.Fields.location(f)
            defVar(ds, "$(k)_lon", Float64.(Array(λnodes(f.grid, ℓ[1]()))[is]), (dims[1],))
            defVar(ds, "$(k)_lat", Float64.(Array(φnodes(f.grid, ℓ[2]()))[js]), (dims[2],))
            defVar(ds, "$(k)_first_bad", Float64.(a[is, js, :]), dims)
            isnothing(b.prev) || defVar(ds, "$(k)_last_good", Float64.(b.prev[k][is, js, :]), dims)
        end
    end
end

# Plain-language summary of one case
function verdict(c, W, nan_logged, iterations_run)
    lines = String[]
    if isnothing(W.blowup)
        crashed = nan_logged ? "went NaN outside the watched window (iteration > $(W.to) or < $(W.from)); see the log" :
                               "no NaN by iteration $(iterations_run)"
        push!(lines, "$(c.name): $(crashed). H100 crash was at $(c.expected).")
        return lines
    end
    b = W.blowup
    push!(lines, "$(c.name): NaN first at iteration $(b.iteration) (H100: $(c.expected)); first bad fields: $(join(b.bad, ", "))")
    for k in b.bad
        a = b.cur[k]
        idx = findall(!isfinite, a)
        lo, hi = Tuple(minimum(idx)), Tuple(maximum(idx))
        push!(lines, "  $(k): $(length(idx)) bad cells, index box $(lo)–$(hi); first at (λ, φ, z) = $(locate(W.fields[k], idx[1]))")
    end
    if !isnothing(b.prev)
        for (k, a) in pairs(b.prev)
            m, I = findmax(abs, a)
            push!(lines, @sprintf("  step before: max |%s| = %.4g at (λ, φ, z) = %s", k, m, locate(W.fields[k], I)))
        end
    end
    # Growth of max |u| over the last 50 watched steps: steady exponential growth = instability, a jump = something abrupt
    us = [parse(Float64, split(r, ",")[4]) for r in W.rows]
    if length(us) >= 3 && us[max(1, end - 50)] > 0
        n = min(50, length(us) - 1)
        a, z = us[end-n], us[end-1]
        push!(lines, @sprintf("  max |u| over the last %d good steps: %.4g → %.4g (×%.3g)", n - 1, a, z, z / a))
    end
    return lines
end

function diagnose(c, out_dir)
    DIAG_FT[] = c.FT
    W = Watch(c.from, c.to, nothing, nothing, String[], nothing)
    DIAG_HOOK[] = function (sim)
        m = sim.model
        fs = m.free_surface
        W.fields = (u = m.velocities.u, v = m.velocities.v, w = m.velocities.w, T = m.tracers.T,
                    η = fs.displacement, U = fs.barotropic_velocities.U, V = fs.barotropic_velocities.V)
        sim.stop_iteration = c.stop
        sim.callbacks[:watch] = Callback(s -> watch!(s, W), IterationInterval(1))
        sim.callbacks[:zonal_schedule] = Callback(s -> nothing, TimeInterval(36.5days))
        sim.callbacks[:output_schedule] = Callback(s -> nothing, TimeInterval(10 * 365days))
        return nothing
    end

    nan, t0 = Ref(false), time()
    iterations_run = Ref(0)
    open(joinpath(out_dir, "$(c.name).log"), "w") do io
        with_logger(RunLogger(io, c.name, nan)) do
            try
                ocean_simulation(c.name, c.P, 200.0, R_Earth, 30.0, 0.0, 10 * 365days;
                                 n_lon = round(Int, 360 / c.res), n_lat = round(Int, 160 / c.res), n_depth = levels_for(200),
                                 use_GPU = c.gpu, momentum_advection = :weno5, τ_biharmonic = c.τ, κ_horizontal = 1e3,
                                 max_Δt = min(3hours * c.res, default_max_Δt(c.τ, c.P), default_max_Δt(nothing, c.P)),
                                 output_interval = nothing, zonal_mean_interval = nothing, checkpoint_interval = nothing)
            catch err
                @error "Run failed" exception = (err, catch_backtrace())
            end
        end
    end
    isempty(W.rows) || open(joinpath(out_dir, "$(c.name)_history.csv"), "w") do io
        println(io, "iteration,time_days,dt_min," * join(("max_$(k),idx_$(k),nbad_$(k)" for k in keys(W.fields)), ","))
        foreach(r -> println(io, r), W.rows)
    end
    isnothing(W.blowup) || save_blowup(joinpath(out_dir, "$(c.name)_blowup.nc"), W)
    log_iterations = [parse(Int, m[1]) for m in eachmatch(r"iteration = (\d+)", read(joinpath(out_dir, "$(c.name).log"), String))]
    lines = verdict(c, W, nan[], isempty(log_iterations) ? c.stop : maximum(log_iterations))
    push!(lines, @sprintf("  (%.1f min)", (time() - t0) / 60))
    GC.gc(true)
    CUDA_AVAILABLE && CUDA.reclaim()
    return lines
end

cases = [
    (name = "early_f32",  res = 0.25, P = 10, FT = Float32, gpu = true,  τ = nothing,  from = 0,      to = 600,    stop = 600,    expected = "200–300"),
    (name = "early_f64",  res = 0.25, P = 10, FT = Float64, gpu = true,  τ = nothing,  from = 0,      to = 600,    stop = 600,    expected = "200–300"),
    (name = "early_cpu",  res = 0.25, P = 10, FT = Float32, gpu = false, τ = nothing,  from = 0,      to = 400,    stop = 400,    expected = "200–300"),
    (name = "p50_f32",    res = 0.5,  P = 50, FT = Float32, gpu = true,  τ = nothing,  from = 33_000, to = 34_000, stop = 40_000, expected = "33,500–33,600"),
    (name = "p50_f64",    res = 0.5,  P = 50, FT = Float64, gpu = true,  τ = nothing,  from = 33_000, to = 34_000, stop = 50_000, expected = "33,500–33,600 (Float32)"),
    (name = "p50_bih300", res = 0.5,  P = 50, FT = Float32, gpu = true,  τ = 300days, from = 33_000, to = 34_000, stop = 50_000, expected = "33,500–33,600 (no viscosity)"),
    (name = "p3_f32",     res = 0.25, P = 3,  FT = Float32, gpu = true,  τ = nothing,  from = 87_600, to = 88_200, stop = 90_000, expected = "88,000–88,100")]

only = findfirst(==("--only"), ARGS)
isnothing(only) || (cases = filter(c -> c.name in split(ARGS[only + 1], ","), cases))

out_dir = joinpath(directory, "ocean", "diag")
mkpath(out_dir)
summary_path = joinpath(out_dir, "summary.txt")
for c in cases
    println("\n=== $(c.name) ===")
    lines = diagnose(c, out_dir)
    foreach(println, lines)
    open(io -> foreach(l -> println(io, l), lines), summary_path, "a")
end
println("\nSummary: $(summary_path)")
