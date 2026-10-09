include("constants.jl")
using Oceananigans
using Oceananigans.Units
using Oceananigans.AbstractOperations: @at, Average
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: λnode, φnode
using NCDatasets
using Statistics

const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

const AMDGPU_AVAILABLE = !CUDA_AVAILABLE && try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const directory = get(ENV, "EXO_OCEAN_OUTPUT", joinpath(REPO_ROOT, "output"))

include("emulator_wind.jl")

#####
##### Grid
#####

# Exponentially stretched vertical grid (ratio r between adjacent cells) spanning ocean_depth; also returns the top-cell thickness
function ocean_z_faces(ocean_depth, n_depth; r = 1.2)
    Δz₀ = ocean_depth * (r - 1) / (r^n_depth - 1)
    return [-Δz₀ * (r^(n_depth - k) - 1) / (r - 1) for k in 0:n_depth], Δz₀
end

function ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo = (3, 3, 3), float_type = Float32)
    z_faces, _ = ocean_z_faces(ocean_depth, n_depth)
    return LatitudeLongitudeGrid(arch, float_type;
                                 size = (n_lon, n_lat, n_depth),
                                 halo,
                                 longitude = (-180, 180),
                                 latitude = (-80, 80),
                                 z = z_faces,
                                 radius = planet_radius,
                                 topology = (Periodic, Bounded, Bounded))
end

function select_architecture(use_GPU)
    use_GPU || return CPU()
    CUDA_AVAILABLE && return GPU()
    AMDGPU_AVAILABLE && return GPU()
    @warn "No functional GPU found, falling back to CPU."
    return CPU()
end

#####
##### Boundary conditions and closures
#####

@inline function bottom_drag_u(i, j, grid, clock, fields)
    u = @inbounds fields.u[i, j, 1]
    v = @inbounds fields.v[i, j, 1]
    return -C_Bottom_Drag * u * sqrt(u^2 + v^2)
end

@inline function bottom_drag_v(i, j, grid, clock, fields)
    u = @inbounds fields.u[i, j, 1]
    v = @inbounds fields.v[i, j, 1]
    return -C_Bottom_Drag * v * sqrt(u^2 + v^2)
end

# Quadratic bulk wind stress from wind_field(lon, lat, t) -> (u_wind, v_wind) in m/s
@inline function wind_stress_u(lon, lat, t, p)
    u_wind, v_wind = p.wind_field(lon, lat, t)
    return -rho_air * C_D_wind * sqrt(u_wind^2 + v_wind^2) * u_wind / rho_seawater
end

@inline function wind_stress_v(lon, lat, t, p)
    u_wind, v_wind = p.wind_field(lon, lat, t)
    return -rho_air * C_D_wind * sqrt(u_wind^2 + v_wind^2) * v_wind / rho_seawater
end

# Relax SST toward a day-night equilibrium (hottest at the substellar point) as a flux with piston velocity Δz_top / τ_relax (Haney 1971)
@inline function surface_relaxation(i, j, grid, clock, fields, p)
    k = grid.Nz
    lon_rad = λnode(i, j, k, grid, Center(), Center(), Center()) * π / 180
    lat_rad = φnode(i, j, k, grid, Center(), Center(), Center()) * π / 180
    T = @inbounds fields.T[i, j, k]
    T_eq = p.T_night + (p.T_day - p.T_night) * max(zero(lon_rad), cos(lon_rad) * cos(lat_rad))
    return p.Δz_top * (T - T_eq) / p.τ_relax  # positive flux cools
end

# wind_stress = (τx, τy): kinematic stress arrays at u- and v-points (e.g. from emulator_wind_stress), used as array-valued top fluxes
function velocity_boundary_conditions(wind_field, wind_stress = nothing)
    u_bottom = FluxBoundaryCondition(bottom_drag_u, discrete_form = true)
    v_bottom = FluxBoundaryCondition(bottom_drag_v, discrete_form = true)
    if !isnothing(wind_stress)
        return FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress[1]), bottom = u_bottom),
               FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress[2]), bottom = v_bottom)
    end
    isnothing(wind_field) && return FieldBoundaryConditions(bottom = u_bottom), FieldBoundaryConditions(bottom = v_bottom)
    p = (; wind_field)
    return FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress_u, parameters = p), bottom = u_bottom),
           FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress_v, parameters = p), bottom = v_bottom)
end

# τ_biharmonic = nothing omits the biharmonic viscosity and κ_horizontal = 0 the horizontal tracer diffusivity
function ocean_closure(; n_lon, n_lat, planet_radius, τ_biharmonic, κ_horizontal, ν_vertical = 1e-4, κ_vertical = 1e-4)
    R = Float64(planet_radius)
    Δλ = 2π / n_lon
    Δy = deg2rad(160) / n_lat * R
    ν₄(λ, φ, z, t) = min(Δλ * R * cosd(φ), Δy)^4 / τ_biharmonic  # same grid-scale damping time everywhere
    biharmonic = isnothing(τ_biharmonic) ? nothing : HorizontalScalarBiharmonicDiffusivity(ν = ν₄)
    horizontal = iszero(κ_horizontal) ? nothing : HorizontalScalarDiffusivity(κ = κ_horizontal)
    closures = (VerticalScalarDiffusivity(ν = ν_vertical, κ = κ_vertical), biharmonic, horizontal,
                ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0))
    return Tuple(c for c in closures if !isnothing(c))
end

function momentum_scheme(name)
    name == :centered && return VectorInvariant()
    name == :weno5 && return WENOVectorInvariant(order = 5)
    name == :weno9 && return WENOVectorInvariant()
    error("momentum_advection must be :centered, :weno5 or :weno9, got $(name)")
end

function tracer_scheme(name)
    name == :upwind3 && return UpwindBiased(order = 3)
    name == :weno5 && return WENO(order = 5)
    error("tracer_advection must be :upwind3 or :weno5, got $(name)")
end

# Smallest grid halo (at least 3) the schemes and closures need, checked on a small CPU grid of the same type
function required_halo(momentum, tracers, closure)
    probe = LatitudeLongitudeGrid(CPU(), Float32; size = (8, 8, 4), longitude = (-180, 180), latitude = (-80, 80), z = (-1, 0),
                                  topology = (Periodic, Bounded, Bounded))
    return Oceananigans.Grids.inflate_halo_size(3, 3, 3, probe, momentum, tracers, closure)
end

# Timestep cap: biharmonic stability if there is a biharmonic viscosity, otherwise inertial stability (f Δt ≤ 0.4 at 80°)
function default_max_Δt(τ_biharmonic, rotational_period)
    isnothing(τ_biharmonic) || return min(1days, τ_biharmonic / 100)
    rotational_period == 0 && return 1days
    return min(1days, 0.4 / (2 * omega_Earth / rotational_period * sind(80)))
end

#####
##### Warm starts
#####

# Remove `key` from every NamedTuple in a loaded checkpoint state (e.g. :S for a model without salinity)
drop_state_key(x, key) = x
drop_state_key(nt::NamedTuple, key) = (; (name => drop_state_key(value, key) for (name, value) in pairs(nt) if name != key)...)

# Restore a same-grid checkpoint, then restart the clock and Δt for a fresh run
function restore_checkpoint!(simulation, path; drop_salinity)
    state = Oceananigans.OutputWriters.load_checkpoint_state(path)
    drop_salinity && (state = drop_state_key(state, :S))
    Oceananigans.restore_prognostic_state!(simulation, state)
    simulation.model.clock.time = zero(simulation.model.clock.time)
    simulation.model.clock.iteration = 0
    simulation.Δt = 2minutes
    return nothing
end

# Warm start from a checkpoint on another grid; donor = (n_lon, n_lat, n_depth, ocean_depth) of the run that wrote it.
# T, u, v are interpolated; below a shallower donor's floor u = v = 0 and T continues the donor's thermocline exponentially.
function regrid_pickup!(model, pickup, donor; planet_radius, T_night = 0.0)
    m = Oceananigans.OutputWriters.load_checkpoint_state(pickup).model
    saved = size(parent(m.tracers.T.data))
    halo = (saved .- (donor.n_lon, donor.n_lat, donor.n_depth)) .÷ 2
    all(halo .> 0) || error("pickup_grid $(donor) doesn't match the checkpoint (T is $(saved) with halos)")
    src = ocean_grid(CPU(); donor.n_lon, donor.n_lat, donor.n_depth, donor.ocean_depth, planet_radius, halo)
    dst = on_architecture(CPU(), model.grid)

    T_src, u_src, v_src = CenterField(src), XFaceField(src), YFaceField(src)
    for (f, saved, name) in ((T_src, m.tracers.T, :T), (u_src, m.velocities.u, :u), (v_src, m.velocities.v, :v))
        data = Array(parent(saved.data))
        size(data) == size(parent(f)) || error("pickup_grid $(donor) doesn't match the checkpoint ($(name) is $(size(data)) with halos, expected $(size(parent(f))))")
        copyto!(parent(f), data)
        fill_halo_regions!(f)
    end

    T = regrid_to(dst, T_src, (Center(), Center(), Center()))
    u = regrid_to(dst, u_src, (Face(), Center(), Center()))
    v = regrid_to(dst, v_src, (Center(), Face(), Center()))

    zc = Array(znodes(dst, Center()))
    below = findall(zc .< -donor.ocean_depth)
    if !isempty(below)
        u[:, :, below] .= 0
        v[:, :, below] .= 0
        δ, z_r = thermocline_e_folding(T_src, src, donor.ocean_depth, T_night)
        if isfinite(δ)
            k_hi = findfirst(zc .>= z_r)
            k_lo = k_hi - 1
            s = (z_r - zc[k_lo]) / (zc[k_hi] - zc[k_lo])
            T_r = (1 - s) .* T[:, :, k_lo] .+ s .* T[:, :, k_hi]
            for k in 1:k_lo
                T[:, :, k] .= T_night .+ (T_r .- T_night) .* exp((zc[k] - z_r) / δ)
            end
            @info "Regridded warm start: exponential fill below $(round(z_r, digits=1)) m (e-folding depth $(round(δ, digits=1)) m)"
        else
            @warn "Regridded warm start: no e-folding fit to the donor profile; filling with each column's bottom value"
        end
    end

    set!(model, T = T, u = u, v = v)
    return nothing
end

# Trilinear interpolation onto the `loc` nodes of `grid`, with target depths clamped to the donor's range (interpolate! returns zeros beyond it)
function regrid_to(grid, field, loc)
    λs, φs, zs = Array(λnodes(grid, loc[1])), Array(φnodes(grid, loc[2])), Array(znodes(grid, loc[3]))
    z_src = Array(znodes(field.grid, Center()))
    z_lo, z_hi = extrema(z_src)
    out = zeros(eltype(grid), length(λs), length(φs), length(zs))
    Threads.@threads for k in eachindex(zs)
        z = clamp(zs[k], z_lo, z_hi)
        for j in eachindex(φs), i in eachindex(λs)
            out[i, j, k] = interpolate((λs[i], φs[j], z), field)
        end
    end
    return out
end

# e-folding depth of (area-mean T − T_night) over 0.3–0.8 H, and the deepest level of the fit; (NaN, NaN) if no fit
function thermocline_e_folding(T, grid, H, T_night)
    zs = Array(znodes(grid, Center()))
    w = cosd.(Array(φnodes(grid, Center())))
    T_mean = [sum(w' .* view(interior(T), :, :, k)) / (size(grid, 1) * sum(w)) for k in eachindex(zs)]
    fit = findall((zs .< -0.3H) .& (zs .> -0.8H) .& (T_mean .> T_night))
    length(fit) >= 2 || return NaN, NaN
    x, y = zs[fit], log.(T_mean[fit] .- T_night)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    slope > 0 || return NaN, NaN
    return 1 / slope, minimum(x)
end

#####
##### Diagnostics and output
#####

# Log progress every call; stop when the trend of volume-mean T over T_drift_window falls below T_drift_tol (°C/yr)
function progress_callback(model, T_drift_tol, T_drift_window)
    mean_T = Field(Average(model.tracers.T))
    mean_KE = Field(Average(@at (Center, Center, Center) (model.velocities.u^2 + model.velocities.v^2 + model.velocities.w^2) / 2))
    previous = Ref((t = 0.0, T = NaN))
    history = Tuple{Float64, Float64}[]

    function progress(sim)
        compute!(mean_T)
        compute!(mean_KE)
        T̄ = Array(interior(mean_T))[1]
        KE = Array(interior(mean_KE))[1]
        t = sim.model.clock.time
        dT̄dt = isfinite(previous[].T) && t > previous[].t ? (T̄ - previous[].T) / (t - previous[].t) * 365days : NaN
        previous[] = (; t, T = T̄)

        @info "Time: $(prettytime(t)), Δt: $(prettytime(sim.Δt)), CFL: $(round(AdvectiveCFL(sim.Δt)(sim.model), digits=3)), " *
              "⟨T⟩: $(round(T̄, digits=3)) °C (d⟨T⟩/dt: $(round(dT̄dt, digits=3)) °C/yr), ⟨KE⟩: $(round(KE, sigdigits=3)) m²/s²"

        if !isnothing(T_drift_tol) && isfinite(T̄)
            push!(history, (t, T̄))
            filter!(s -> t - s[1] <= T_drift_window, history)
            if length(history) >= 3 && history[end][1] - history[1][1] >= 0.99 * T_drift_window
                years = first.(history) ./ 365days
                temps = last.(history)
                trend = sum((years .- mean(years)) .* (temps .- mean(temps))) / sum((years .- mean(years)) .^ 2)
                if abs(trend) < T_drift_tol
                    @info "Deep equilibrium: ⟨T⟩ trend $(round(trend, sigdigits=2)) °C/yr (|.| < $(T_drift_tol)) over the last " *
                          "$(round((history[end][1] - history[1][1]) / 365days, digits=1)) years. Stopping at $(prettytime(t))."
                    sim.running = false
                end
            end
        end
    end

    return progress
end

function add_output_writers!(simulation, name; output_interval, zonal_mean_interval, checkpoint_interval, global_attributes = Dict())
    model = simulation.model
    ocean_dir = joinpath(directory, "ocean")
    mkpath(ocean_dir)
    fields = (; T = model.tracers.T, u = model.velocities.u, v = model.velocities.v, w = model.velocities.w)

    if !isnothing(output_interval)
        simulation.output_writers[:full_3d] = NetCDFWriter(model, fields; filename = joinpath(ocean_dir, "$(name).nc"),
                                                           schedule = TimeInterval(output_interval), overwrite_existing = true, global_attributes)
    end

    if !isnothing(zonal_mean_interval)
        zonal = map(f -> Field(Average(f, dims = 1)), fields)
        simulation.output_writers[:zonal_mean] = NetCDFWriter(model, zonal; filename = joinpath(ocean_dir, "$(name)_zonal.nc"),
                                                              schedule = TimeInterval(zonal_mean_interval), overwrite_existing = true, global_attributes)
    end

    if !isnothing(checkpoint_interval)
        checkpoint_dir = joinpath(ocean_dir, "checkpoints")
        mkpath(checkpoint_dir)
        simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = TimeInterval(checkpoint_interval), dir = checkpoint_dir,
                                                                prefix = name, cleanup = true, overwrite_existing = true)
    end

    return nothing
end

# Save the final state as a 3D snapshot if the run stopped between output times (e.g. at the drift stop)
function write_final_snapshot!(simulation)
    haskey(simulation.output_writers, :full_3d) || return nothing
    writer = simulation.output_writers[:full_3d]
    last_write = writer.schedule.first_actuation_time + writer.schedule.actuations * writer.schedule.interval
    simulation.model.clock.time > last_write + 1 && Oceananigans.write_output!(writer, simulation.model)
    return nothing
end

#####
##### Simulation
#####

"""
    ocean_simulation(name, rotational_period, ocean_depth, planet_radius, T_day, T_night, simulation_time; kwargs...)

Tidally locked ocean forced by relaxing SST toward a day-night equilibrium (T_day at the substellar point, T_night on the
nightside). `rotational_period` is in Earth days (0 for no rotation). Output goes to `\$EXO_OCEAN_OUTPUT/ocean/` (default `output/ocean/`).

Keywords:
- `n_lat`, `n_lon`, `n_depth`, `use_GPU`: grid and architecture.
- `float_type`: grid precision (Float64 avoids the Float32 WENO NaNs seen on the H100, at ~1.3× the cost there).
- `τ_relax`, `τ_biharmonic`, `κ_horizontal`: SST relaxation time, grid-scale damping time of the biharmonic viscosity
  (`nothing` for none), horizontal tracer diffusivity (0 for none).
- `ν_vertical`, `κ_vertical`: background vertical viscosity and tracer diffusivity (m²/s).
- `momentum_advection` (`:centered`, `:weno5`, `:weno9`), `tracer_advection` (`:upwind3`, `:weno5`): advection schemes.
- `wind_field`: `(lon, lat, t) -> (u_wind, v_wind)` in m/s for a bulk wind stress.
- `emulator_wind`, `surface_pressure`: force with CLERO-emulated winds of the planet whose emulated (T_day, T_night) match this run's at
  surface pressure `surface_pressure` (bar). The wind file is solved by `src/climate/climate_emulator.py` (Python env `\$EXO_EMULATOR_PYTHON`)
  and cached in `\$EXO_OCEAN_OUTPUT/winds/`; errors if no planet matches. `emulator_gcm` (`"exocam"`/`"um"`), `emulator_feh` (stellar
  [Fe/H]; default the nearby M-dwarf mean) and `emulator_device` (`"cpu"`/`"cuda"`) are passed through; `wind_stress_scale` multiplies the stress.
- `salinity`: carry a passive salinity tracer initialised to this value (`nothing`: temperature only).
- `cfl`, `max_Δt`: advective CFL target and timestep cap (default τ_biharmonic / 100 for biharmonic stability, else inertial stability).
- `pickup`: checkpoint to warm-start from; `pickup_grid = (; n_lon, n_lat, n_depth, ocean_depth)` if it was written on another grid.
- `T_drift_tol`, `T_drift_window`: stop when the trend of volume-mean T over the window falls below this (°C/yr).
- `output_interval`, `zonal_mean_interval`, `checkpoint_interval`: model time between 3D snapshots, zonal means and checkpoints (`nothing` for none).
"""
function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, T_day, T_night, simulation_time;
                          n_lat = 160, n_lon = 360, n_depth = 20, use_GPU = true, float_type = Float32,
                          τ_relax = 30days, τ_biharmonic = 10days, κ_horizontal = 1e3, ν_vertical = 1e-4, κ_vertical = 1e-4,
                          momentum_advection = :centered, tracer_advection = :upwind3,
                          wind_field = nothing, salinity = nothing,
                          emulator_wind = false, surface_pressure = nothing, emulator_gcm = "exocam", emulator_feh = nothing,
                          emulator_device = nothing, wind_stress_scale = 1.0,
                          cfl = 0.2, max_Δt = default_max_Δt(τ_biharmonic, rotational_period),
                          pickup = nothing, pickup_grid = nothing,
                          T_drift_tol = nothing, T_drift_window = 10 * 365days,
                          output_interval = 5 * 365days, zonal_mean_interval = nothing, checkpoint_interval = nothing)

    @info "Setting up simulation $(simulation_name)..."
    wind_file = nothing
    if emulator_wind  # solve before the grid exists, so the Python GPU job has released its memory
        isnothing(surface_pressure) && error("emulator_wind = true needs surface_pressure (bar)")
        isnothing(wind_field) || error("pass either wind_field or emulator_wind, not both")
        rotational_period > 0 || error("emulator_wind needs a rotating, tidally locked planet (rotational_period > 0)")
        wind_file = emulator_wind_file(rotational_period, T_day, T_night, surface_pressure; gcm = emulator_gcm, feh = emulator_feh, device = emulator_device)
    end
    arch = select_architecture(use_GPU)
    momentum = momentum_scheme(momentum_advection)
    tracers_adv = tracer_scheme(tracer_advection)
    closure = ocean_closure(; n_lon, n_lat, planet_radius, τ_biharmonic, κ_horizontal, ν_vertical, κ_vertical)
    halo = required_halo(momentum, tracers_adv, closure)
    grid = ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo, float_type)
    _, Δz_top = ocean_z_faces(ocean_depth, n_depth)

    rotation_rate = rotational_period == 0 ? 0 : omega_Earth / rotational_period
    eos = LinearEquationOfState(thermal_expansion = 2e-4, haline_contraction = 0.0)
    buoyancy = isnothing(salinity) ? SeawaterBuoyancy(equation_of_state = eos, constant_salinity = 35.0) : SeawaterBuoyancy(equation_of_state = eos)

    wind_stress = isnothing(wind_file) ? nothing : emulator_wind_stress(wind_file, grid; P_rot = rotational_period, scale = wind_stress_scale)
    u_bcs, v_bcs = velocity_boundary_conditions(wind_field, wind_stress)
    relaxation = (; T_day, T_night, τ_relax, Δz_top)
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(surface_relaxation, discrete_form = true, parameters = relaxation))

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = momentum,
                                        tracer_advection = tracers_adv,
                                        coriolis = HydrostaticSphericalCoriolis(; rotation_rate),
                                        buoyancy,
                                        tracers = isnothing(salinity) ? (:T,) : (:T, :S),
                                        closure,
                                        free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = max_Δt),
                                        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs))

    if isnothing(pickup)
        set!(model, T = T_night)
    elseif !isnothing(pickup_grid)
        @info "Warm-starting from $(pickup), regridded from $(pickup_grid)"
        regrid_pickup!(model, pickup, pickup_grid; planet_radius, T_night)
    end
    isnothing(salinity) || set!(model, S = salinity)

    simulation = Simulation(model, Δt = 2minutes, stop_time = simulation_time)
    if !isnothing(pickup) && isnothing(pickup_grid)
        @info "Warm-starting from $(pickup)"
        restore_checkpoint!(simulation, pickup; drop_salinity = isnothing(salinity))
    end

    simulation.callbacks[:progress] = Callback(progress_callback(model, T_drift_tol, T_drift_window), IterationInterval(100))
    simulation.callbacks[:wizard] = Callback(TimeStepWizard(; cfl, max_change = 1.05, max_Δt), IterationInterval(10))
    global_attributes = isnothing(wind_file) ? Dict() : emulator_attributes(wind_file, wind_stress_scale)
    add_output_writers!(simulation, simulation_name; output_interval, zonal_mean_interval, checkpoint_interval, global_attributes)

    @info "Simulation setup complete. Starting the run..."
    run!(simulation, checkpoint_at_end = !isnothing(checkpoint_interval))
    write_final_snapshot!(simulation)
    @info "Simulation finished successfully!"

    return nothing
end
