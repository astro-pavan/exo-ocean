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
    return model, relaxation, isnothing(salinity)
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

# Cell thickness at depth z
cell_thickness(grid, z) = (zf = Array(znodes(on_architecture(CPU(), grid), Face())); k = clamp(searchsortedlast(zf, z), 1, length(zf) - 1); zf[k+1] - zf[k])

"""
    tracer_release(run; release_depths = (-150.0, -300.0, -600.0), years = 3, σ_z = nothing, latitudes = (-90, 90), use_GPU = true,
                   diagnostic_interval = 5days, output_interval = 0.5 * 365days, skip = 0.25 * 365days, name = "<run>_tracers")

Release one passive tracer per depth in `release_depths` into the converged run `run`, rebuilt from its parameter file and started
from its latest checkpoint (clock reset to 0), and run for `years`. Tracer n starts as exp(−(T − T*ₙ)²/2σ_T²) on the isotherm T*ₙ
found at reference depth zₙ of the sorted T profile, with σ_T = |dT/dz*| σ_z (σ_z defaults to the cell thickness there), between
`latitudes`, and zero in columns where the isotherm outcrops (top-cell T < T* + 3σ_T).

Every `diagnostic_interval` it records each tracer's mass, mean T̄ and variance σ_T² of T weighted by the tracer, its weighted ⟨(∂T/∂z)²⟩
and the share of the tracer in the top cell (to `<name>_moments.csv`), and T and the tracers every `output_interval` (to `<name>.nc`).
At the end κ = ½ dσ_T²/dt / G² from a least-squares slope after `skip`, with G either the reference-profile gradient at T̄ (comparable
with the Walin κ) or the tracer-weighted local ∂T/∂z; written to `<name>_kappa.toml` and returned.
"""
function tracer_release(run; release_depths = (-150.0, -300.0, -600.0), years = 3, σ_z = nothing, latitudes = (-90, 90), use_GPU = true,
                        diagnostic_interval = 5days, output_interval = 0.5 * 365days, skip = 0.25 * 365days, name = "$(run)_tracers")
    p = read_parameters(run)
    blew_up(run) && error("$(run) blew up (NaN)")
    is_finished(run) && p["finished"]["stopped_by"] == "equilibrium" || @warn "$(run) didn't stop at deep equilibrium; κ may include its drift"
    tracer_names = Tuple(Symbol("c$(n)") for n in eachindex(release_depths))
    model, relaxation, no_salinity = rebuild_model(p, tracer_names; use_GPU)
    grid = model.grid

    max_Δt = p["time"]["max_Δt"]
    simulation = Simulation(model, Δt = 2minutes, stop_time = years * 365days)
    checkpoint = latest_checkpoint(run)
    @info "Tracer release in $(run) from $(checkpoint)"
    restore_checkpoint!(simulation, checkpoint; drop = no_salinity ? (:S,) : (), reset_clock = false)
    simulation.model.clock.time = 0
    simulation.model.clock.iteration = 0    # the tracers' first step is forward Euler
    simulation.Δt = min(simulation.Δt, max_Δt)

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
    V_top = [Vᶜᶜᶜ(i, j, size(cpu, 3), cpu) for i in 1:size(cpu, 1), j in 1:size(cpu, 2)]
    scalar(f) = (compute!(f); Array(interior(f))[1])
    rows = Vector{Vector{Float64}}()
    function record(sim)
        row = [sim.model.clock.time]
        for m in moments
            M, MT, MT², MG = scalar(m.M), scalar(m.MT), scalar(m.MT²), scalar(m.MG)
            T̄ = MT / M
            top = sum(Array(interior(m.c, :, :, size(cpu, 3))) .* V_top) / M
            append!(row, (M, T̄, MT² / M - T̄^2, MG / M, top))
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

    header = ["t"; [string(n, "_", q) for n in tracer_names for q in ("M", "Tmean", "varT", "Tz2", "top_share")]]
    open(joinpath(directory, "$(name)_moments.csv"), "w") do io
        println(io, join(header, ","))
        foreach(r -> println(io, join(r, ",")), rows)
    end

    # κ from the least-squares growth rate of σ_T² after `skip`
    t = getindex.(rows, 1)
    fit = t .>= skip
    count(fit) >= 3 || error("fewer than 3 diagnostics after skip = $(prettytime(skip)); run longer or record more often")
    slope(y) = (x = t[fit]; sum((x .- mean(x)) .* (y[fit] .- mean(y[fit]))) / sum((x .- mean(x)) .^ 2))
    results = map(enumerate(releases)) do (n, r)
        col(q) = getindex.(rows, 1 + 5(n - 1) + q)
        rate = slope(col(3))
        T̄ = mean(col(2)[fit])
        z̄ = z_of(T̄)
        G_ref = dTdz(z̄, r.σ_z)
        κ_ref, κ_local = rate / 2 / G_ref^2, rate / 2 / mean(col(4)[fit])
        @info "Tracer c$(n) (T* = $(round(r.T★, sigdigits = 4)) °C, now z* = $(round(z̄, digits = 1)) m): κ = $(round(κ_ref, sigdigits = 3)) m²/s " *
              "(reference dT/dz*), $(round(κ_local, sigdigits = 3)) m²/s (tracer-weighted ∂T/∂z); top-cell share at the end $(round(col(5)[end], sigdigits = 2))"
        Dict{String, Any}("release_depth" => r.z, "T_release" => r.T★, "sigma_T" => r.σ_T, "sigma_z" => r.σ_z, "T_mean" => T̄, "z_star" => z̄,
                          "dvarT_dt" => rate, "dTdz_reference" => G_ref, "kappa_reference" => κ_ref, "kappa_local" => κ_local,
                          "mass_change" => col(1)[end] / col(1)[1] - 1, "top_share_end" => col(5)[end])
    end
    out = Dict{String, Any}("run" => run, "checkpoint" => checkpoint, "years" => years, "skip" => skip, "latitudes" => collect(latitudes),
                            "tracers" => Dict("c$(n)" => r for (n, r) in enumerate(results)))
    open(io -> TOML.print(io, out; sorted = true), joinpath(directory, "$(name)_kappa.toml"), "w")
    return out
end
