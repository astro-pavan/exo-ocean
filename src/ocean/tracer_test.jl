# Diapycnal diffusivity of a converged run from passive tracers released on isotherms (a virtual Ledwell et al. 1993, Nature 364, 701,
# release; Hill et al. 2012, Ocean Modelling 45–46, 14). The tracers feel the run's own advection and closures, so κ counts explicit,
# CATKE, GM-Redi and numerical mixing, like the Walin heat-budget κ.
isdefined(@__MODULE__, :ocean_simulation) || include("ocean_sim.jl")
using Oceananigans.AbstractOperations: Integral, ∂z
using Oceananigans.Operators: Vᶜᶜᶜ, Azᶜᶜᶜ

# Wind file of a run, also when it was recorded on another machine
resolve_wind_file(path) = isfile(path) ? path : joinpath(directory, "winds", basename(path))

# The run's model rebuilt from its parameter file, with passive tracers `extra` added
function rebuild_model(p, extra; use_GPU = true)
    planet, physics, time = p["planet"], p["physics"], p["time"]
    g = donor_grid(p)
    get(physics, "wind_field", "nothing") == "nothing" || error("$(p["run"]["name"]) used a custom wind_field, which isn't recorded")
    arch = select_architecture(use_GPU)
    momentum, tracers_adv = WENOVectorInvariant(order = 5), WENO(order = 5)
    closure = ocean_closure(; κ_horizontal = physics["κ_horizontal"], κ_vertical = physics["κ_vertical"], prandtl = run_prandtl(p),
                            vertical_mixing = run_vertical_mixing(p), horizontal_mixing = run_horizontal_mixing(p))
    halo = something(g.halo, required_halo(momentum, tracers_adv, closure, g.max_lat))
    grid = ocean_grid(arch; g.n_lon, g.n_lat, g.n_depth, g.ocean_depth, planet_radius = planet["planet_radius"], halo, g.float_type,
                      r = g.vertical_stretching, max_lat = g.max_lat)
    tidal = run_tidal_parameters(p)
    tidal_κ = isnothing(tidal) ? nothing : Field{Center, Center, Face}(grid)
    isnothing(tidal) || (closure = ocean_closure(; κ_horizontal = physics["κ_horizontal"], κ_vertical = physics["κ_vertical"], prandtl = run_prandtl(p),
                                                 vertical_mixing = run_vertical_mixing(p), horizontal_mixing = run_horizontal_mixing(p), tidal = (; κ = tidal_κ, tidal.prandtl)))

    P_rot = planet["rotational_period"]
    s_w = get(physics, "wind_current_coupling", 1.0)
    scale = get(physics, "wind_stress_scale", 1.0)
    wind_file = physics["emulator_wind"] ? resolve_wind_file(physics["wind_file"]) : nothing
    wind_stress = isnothing(wind_file) ? nothing :
                  s_w == 1 ? emulator_wind_stress(wind_file, grid; P_rot, scale) : emulator_current_feedback(wind_file, grid, s_w; P_rot, scale)
    u_bcs, v_bcs = velocity_boundary_conditions(nothing, wind_stress)
    relaxation = (; T_day = planet["T_day"], T_night = planet["T_night"], W = physics["heat_exchange_coefficient"] / (rho_seawater * cp_seawater))
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(surface_relaxation, discrete_form = true, parameters = relaxation))

    salinity = physics["salinity"] == "nothing" ? nothing : physics["salinity"]
    eos = LinearEquationOfState(thermal_expansion = 2e-4, haline_contraction = 0.0)
    buoyancy = isnothing(salinity) ? SeawaterBuoyancy(equation_of_state = eos, constant_salinity = 35.0) : SeawaterBuoyancy(equation_of_state = eos)
    max_Δt = time["max_Δt"]
    rotation_rate = P_rot == 0 ? 0 : omega_Earth / P_rot
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection = momentum, tracer_advection = tracers_adv,
                                        coriolis = HydrostaticSphericalCoriolis(; rotation_rate), buoyancy,
                                        tracers = ((isnothing(salinity) ? (:T,) : (:T, :S))..., extra...), closure,
                                        free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = max_Δt),
                                        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs))
    return model, relaxation, isnothing(salinity), tidal, tidal_κ
end

# Sorted (Walin) reference profile of T: T at reference depth z* and z* of a temperature, z* = −H + V(T' < T) / A
function reference_profile(T, grid)
    cpu = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu)
    V = [Vᶜᶜᶜ(i, j, k, cpu) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    A = sum(Azᶜᶜᶜ(i, j, Nz, cpu) for i in 1:Nx, j in 1:Ny)
    H = -znodes(cpu, Face())[1]
    order = sortperm(vec(T))
    Ts, cumV = vec(T)[order], cumsum(vec(V)[order])
    T_at(z) = Ts[clamp(searchsortedfirst(cumV, (z + H) * A), 1, length(Ts))]
    z_of(Tq) = -H + (searchsortedfirst(Ts, Tq) > 1 ? cumV[searchsortedfirst(Ts, Tq) - 1] : 0.0) / A
    dTdz(z, δ) = (T_at(min(z + δ, 0.0)) - T_at(max(z - δ, -H))) / (min(z + δ, 0.0) - max(z - δ, -H))
    return T_at, z_of, dTdz
end

# Tracer-weighted median of T, its half-width (q84 − q16)/2 and the mean (∂T/∂z)² of the tracer between those quantiles; a warm tail reaching the surface can't dominate them
function robust_moments(c, T, Tz², V)
    w = vec(max.(c, 0) .* V)
    order = sortperm(vec(T))
    Ts, F = vec(T)[order], cumsum(w[order]) ./ sum(w)
    function q(f)
        i = clamp(searchsortedfirst(F, f), 2, length(Ts))
        return Ts[i-1] + (Ts[i] - Ts[i-1]) * (f - F[i-1]) / max(F[i] - F[i-1], eps())
    end
    q16, q84 = q(0.16), q(0.84)
    core = q16 .<= vec(T) .<= q84
    return q(0.5), (q84 - q16) / 2, sum(w[core] .* vec(Tz²)[core]) / sum(w[core])
end

# Cell thickness at depth z
cell_thickness(grid, z) = (zf = Array(znodes(on_architecture(CPU(), grid), Face())); k = clamp(searchsortedlast(zf, z), 1, length(zf) - 1); zf[k+1] - zf[k])

"""
    tracer_release(run; release_depths = (-150.0, -300.0, -600.0), years = 1, σ_z = nothing, latitudes = (-90, 90), use_GPU = true,
                   diagnostic_interval = 5days, output_interval = 0.5 * 365days, skip = 0.25 * 365days, fit_end = 365days,
                   surface_depth = 60.0, name = "<run>_tracers")

Release one passive tracer per depth in `release_depths` into the converged run `run`, rebuilt from its parameter file and started
from its latest checkpoint (clock reset to 0), and run for `years`. Tracer n starts as exp(−(T − T*ₙ)²/2σ_T²) on the isotherm T*ₙ
found at reference depth zₙ of the sorted T profile, with σ_T = |dT/dz*| σ_z (σ_z defaults to the cell thickness there), between
`latitudes`, and zero in columns where the isotherm outcrops (top-cell T < T* + 3σ_T).

Every `diagnostic_interval` it records each tracer's mass, mean T̄ and variance σ_T² of T weighted by the tracer, its weighted ⟨(∂T/∂z)²⟩,
its share in the top `surface_depth` m, and the median, half-width (q84 − q16)/2 and core ⟨(∂T/∂z)²⟩ of `robust_moments` (to `<name>_moments.csv`),
and T and the tracers every `output_interval` (to `<name>.nc`). At the end κ = ½ dσ²/dt / G² from a least-squares slope between `skip` and `fit_end`.
`kappa_robust` uses the half-width and core gradient and is the one to quote: the variance is dominated by the few % of tracer that reaches warm
surface water, where heating rather than mixing changes T. `kappa_reference` (reference-profile gradient at the median, comparable with the Walin κ)
and `kappa_local` (tracer-weighted ∂T/∂z) use the variance and are kept for comparison. All three are written to `<name>_kappa.toml` and returned.
"""
function tracer_release(run; release_depths = (-150.0, -300.0, -600.0), years = 1, σ_z = nothing, latitudes = (-90, 90), use_GPU = true,
                        diagnostic_interval = 5days, output_interval = 0.5 * 365days, skip = 0.25 * 365days, fit_end = 365days,
                        surface_depth = 60.0, name = "$(run)_tracers")
    p = read_parameters(run)
    blew_up(run) && error("$(run) blew up (NaN)")
    is_finished(run) && p["finished"]["stopped_by"] == "equilibrium" || @warn "$(run) didn't stop at deep equilibrium; κ may include its drift"
    tracer_names = Tuple(Symbol("c$(n)") for n in eachindex(release_depths))
    model, relaxation, no_salinity, tidal, tidal_κ = rebuild_model(p, tracer_names; use_GPU)
    grid = model.grid

    max_Δt = p["time"]["max_Δt"]
    simulation = Simulation(model, Δt = 2minutes, stop_time = years * 365days)
    checkpoint = latest_checkpoint(run)
    @info "Tracer release in $(run) from $(checkpoint)"
    restore_checkpoint!(simulation, checkpoint; drop = no_salinity ? (:S,) : (), reset_clock = false)
    simulation.model.clock.time = 0
    simulation.model.clock.iteration = 0    # the tracers' first step is forward Euler
    simulation.Δt = min(simulation.Δt, max_Δt)
    isnothing(tidal) || add_tidal_κ_callback!(simulation, tidal, tidal_κ)

    # Tracers released on isotherms, away from their outcrops
    T = Array(interior(model.tracers.T))
    T_at, z_of, dTdz = reference_profile(T, grid)
    φ = Array(φnodes(on_architecture(CPU(), grid), Center()))
    in_band = reshape(latitudes[1] .<= φ .<= latitudes[2], 1, :)
    releases = map(release_depths) do z
        σ = something(σ_z, cell_thickness(grid, z))
        T★, G = T_at(z), dTdz(z, σ)
        σ_T = abs(G) * σ
        c = exp.(-(T .- T★) .^ 2 ./ (2σ_T^2)) .* (in_band .& (T[:, :, end] .>= T★ + 3σ_T))
        @info "Release at z* = $(z) m: T* = $(round(T★, sigdigits = 4)) °C, σ_T = $(round(σ_T, sigdigits = 3)) °C (σ_z = $(round(σ, digits = 1)) m), " *
              "$(round(100 * mean(T[:, :, end] .< T★ + 3σ_T), digits = 1))% of columns outcrop"
        (; z, T★, σ_T, σ_z = σ, c)
    end
    set!(model; (n => r.c for (n, r) in zip(tracer_names, releases))...)

    # Tracer-weighted moments of T, on the architecture
    Tf = model.tracers.T
    Tz² = @at (Center, Center, Center) ∂z(Tf)^2
    moments = map(tracer_names) do n
        c = model.tracers[n]
        (; c, M = Field(Integral(c)), MT = Field(Integral(c * Tf)), MT² = Field(Integral(c * Tf * Tf)), MG = Field(Integral(c * Tz²)))
    end
    cpu = on_architecture(CPU(), grid)
    V = [Vᶜᶜᶜ(i, j, k, cpu) for i in 1:size(cpu, 1), j in 1:size(cpu, 2), k in 1:size(cpu, 3)]
    near_surface = reshape(Array(znodes(cpu, Center())) .> -surface_depth, 1, 1, :)
    Tz²_field = Field(Tz²)
    scalar(f) = (compute!(f); Array(interior(f))[1])
    rows = Vector{Vector{Float64}}()
    function record(sim)
        row = [sim.model.clock.time]
        compute!(Tz²_field)
        T_now, Tz²_now = Array(interior(Tf)), Array(interior(Tz²_field))
        for m in moments
            M, MT, MT², MG = scalar(m.M), scalar(m.MT), scalar(m.MT²), scalar(m.MG)
            T̄ = MT / M
            c_now = Array(interior(m.c))
            surface_share = sum(c_now .* V .* near_surface) / M
            append!(row, (M, T̄, MT² / M - T̄^2, MG / M, surface_share, robust_moments(c_now, T_now, Tz²_now, V)...))
        end
        push!(rows, row)
    end

    simulation.callbacks[:moments] = Callback(record, TimeInterval(diagnostic_interval))
    simulation.callbacks[:progress] = Callback(progress_callback(model, surface_heat_flux(model, relaxation), nothing), IterationInterval(100))
    simulation.callbacks[:wizard] = Callback(TimeStepWizard(; cfl = p["time"]["cfl"], max_change = 1.05, max_Δt), IterationInterval(10))
    outputs = (; T = Tf, (n => model.tracers[n] for n in tracer_names)...)
    simulation.output_writers[:tracers] = NetCDFWriter(model, outputs; filename = joinpath(directory, "$(name).nc"),
                                                       schedule = TimeInterval(output_interval), overwrite_existing = true)
    record(simulation)
    run!(simulation)
    record(simulation)
    t_end = simulation.model.clock.time
    (isfinite(t_end) && !hasnan(model.tracers.T)) || error("tracer release in $(run) blew up (NaN) at $(prettytime(t_end))")

    header = ["t"; [string(n, "_", q) for n in tracer_names for q in ("M", "Tmean", "varT", "Tz2", "surface_share", "Tmedian", "halfwidth", "Tz2_core")]]
    open(joinpath(directory, "$(name)_moments.csv"), "w") do io
        println(io, join(header, ","))
        foreach(r -> println(io, join(r, ",")), rows)
    end

    # κ from the least-squares growth rates between `skip` and `fit_end`
    t = getindex.(rows, 1)
    fit = skip .<= t .<= fit_end
    count(fit) >= 3 || error("fewer than 3 diagnostics between skip = $(prettytime(skip)) and fit_end = $(prettytime(fit_end)); run longer or record more often")
    slope(y) = (x = t[fit]; sum((x .- mean(x)) .* (y[fit] .- mean(y[fit]))) / sum((x .- mean(x)) .^ 2))
    results = map(enumerate(releases)) do (n, r)
        col(q) = getindex.(rows, 1 + 8(n - 1) + q)
        rate = slope(col(3))
        z̄ = z_of(mean(col(6)[fit]))
        G_ref = dTdz(z̄, r.σ_z)
        κ_ref, κ_local = rate / 2 / G_ref^2, rate / 2 / mean(col(4)[fit])
        κ_robust = slope(col(7) .^ 2) / 2 / mean(col(8)[fit])
        surface_share = col(5)[findlast(fit)]
        @info "Tracer c$(n) (T* = $(round(r.T★, sigdigits = 4)) °C, now z* = $(round(z̄, digits = 1)) m): κ = $(round(κ_robust, sigdigits = 3)) m²/s (robust); " *
              "from the variance $(round(κ_ref, sigdigits = 3)) (reference dT/dz*), $(round(κ_local, sigdigits = 3)) (tracer-weighted ∂T/∂z); $(round(100 * surface_share, sigdigits = 2))% of the tracer in the top $(surface_depth) m at fit_end"
        Dict{String, Any}("release_depth" => r.z, "T_release" => r.T★, "sigma_T" => r.σ_T, "sigma_z" => r.σ_z, "T_mean" => mean(col(2)[fit]), "T_median" => mean(col(6)[fit]), "z_star" => z̄,
                          "dvarT_dt" => rate, "dTdz_reference" => G_ref, "kappa_robust" => κ_robust, "kappa_reference" => κ_ref, "kappa_local" => κ_local,
                          "mass_change" => col(1)[end] / col(1)[1] - 1, "surface_share_at_fit_end" => surface_share)
    end
    out = Dict{String, Any}("run" => run, "checkpoint" => checkpoint, "years" => years, "skip" => skip, "fit_end" => fit_end, "surface_depth" => surface_depth, "latitudes" => collect(latitudes),
                            "tracers" => Dict("c$(n)" => r for (n, r) in enumerate(results)))
    open(io -> TOML.print(io, out; sorted = true), joinpath(directory, "$(name)_kappa.toml"), "w")
    return out
end
