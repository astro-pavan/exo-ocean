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
using TOML

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

function ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo = (3, 3, 3), float_type = Float32, r = 1.2, max_lat = 80)
    z_faces, _ = ocean_z_faces(ocean_depth, n_depth; r)
    return LatitudeLongitudeGrid(arch, float_type;
                                 size = (n_lon, n_lat, n_depth),
                                 halo,
                                 longitude = (-180, 180),
                                 latitude = (-max_lat, max_lat),
                                 z = z_faces,
                                 radius = planet_radius,
                                 topology = (Periodic, Bounded, Bounded))
end

# Number of cells of width `res` degrees across `span` degrees; errors unless they fit exactly
function grid_cells(span, res, direction)
    n = span / res
    isapprox(n, round(n); atol = 1e-6) || error("a $(res)° $(direction) resolution doesn't divide the $(span)° span into whole cells")
    return round(Int, n)
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

# Relax SST toward a day-night equilibrium (hottest at the substellar point) with a fixed piston velocity p.W = λ / ρc_p (Haney 1971)
@inline function surface_relaxation(i, j, grid, clock, fields, p)
    k = grid.Nz
    lon_rad = λnode(i, j, k, grid, Center(), Center(), Center()) * π / 180
    lat_rad = φnode(i, j, k, grid, Center(), Center(), Center()) * π / 180
    T = @inbounds fields.T[i, j, k]
    T_eq = p.T_night + (p.T_day - p.T_night) * max(zero(lon_rad), cos(lon_rad) * cos(lat_rad))
    return p.W * (T - T_eq)  # positive flux cools
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

# κ_horizontal = 0 omits the horizontal tracer diffusivity
function ocean_closure(; κ_horizontal, ν_vertical = 1e-4, κ_vertical = 1e-4)
    horizontal = iszero(κ_horizontal) ? nothing : HorizontalScalarDiffusivity(κ = κ_horizontal)
    closures = (VerticalScalarDiffusivity(ν = ν_vertical, κ = κ_vertical), horizontal,
                ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0))
    return Tuple(c for c in closures if !isnothing(c))
end

# Smallest grid halo (at least 3) the schemes and closures need, checked on a small CPU grid of the same type
function required_halo(momentum, tracers, closure, max_lat=80)
    probe = LatitudeLongitudeGrid(CPU(), Float32; size = (8, 8, 4), longitude = (-180, 180), latitude = (-max_lat, max_lat), z = (-1, 0),
                                  topology = (Periodic, Bounded, Bounded))
    return Oceananigans.Grids.inflate_halo_size(3, 3, 3, probe, momentum, tracers, closure)
end

# Timestep cap: otherwise inertial stability (f Δt ≤ 0.4 at the grid's highest latitude)
function default_max_Δt(rotational_period, max_lat = 80)
    rotational_period == 0 && return 1days
    return min(1days, 0.4 / (2 * omega_Earth / rotational_period * sind(max_lat)))
end

#####
##### Warm starts
#####

# Remove `key` from every NamedTuple in a loaded checkpoint state (e.g. :S for a model without salinity)
drop_state_key(x, key) = x
drop_state_key(nt::NamedTuple, key) = (; (name => drop_state_key(value, key) for (name, value) in pairs(nt) if name != key)...)

# Restore a same-grid checkpoint (model, Δt and the schedules of output writers already added); reset_clock starts a fresh run from it
function restore_checkpoint!(simulation, path; drop_salinity, reset_clock)
    state = Oceananigans.OutputWriters.load_checkpoint_state(path)
    drop_salinity && (state = drop_state_key(state, :S))
    Oceananigans.restore_prognostic_state!(simulation, state)
    if reset_clock
        simulation.model.clock.time = zero(simulation.model.clock.time)
        simulation.model.clock.iteration = 0
        simulation.Δt = 2minutes
    end
    return nothing
end

# Warm start from a checkpoint on another grid, described by the donor run's parameter file (see `read_parameters`).
# T, u, v are interpolated; below a shallower donor's floor u = v = 0 and T continues the donor's thermocline exponentially.
function regrid_warm_start!(model, checkpoint, donor_parameters; planet_radius, T_night = 0.0)
    donor = donor_grid(donor_parameters)
    m = Oceananigans.OutputWriters.load_checkpoint_state(checkpoint).model
    saved = size(parent(m.tracers.T.data))
    halo = something(donor.halo, (saved .- (donor.n_lon, donor.n_lat, donor.n_depth)) .÷ 2)    # older parameter files have no halo
    all(halo .> 0) || error("the donor grid $(donor) doesn't match the checkpoint (T is $(saved) with halos)")

    src = ocean_grid(CPU(); donor.n_lon, donor.n_lat, donor.n_depth, donor.ocean_depth, planet_radius, halo,
                     float_type = donor.float_type, r = donor.vertical_stretching, max_lat = donor.max_lat)
    dst = on_architecture(CPU(), model.grid)

    T_src, u_src, v_src = CenterField(src), XFaceField(src), YFaceField(src)
    for (f, saved, name) in ((T_src, m.tracers.T, :T), (u_src, m.velocities.u, :u), (v_src, m.velocities.v, :v))
        data = Array(parent(saved.data))
        size(data) == size(parent(f)) || error("the donor grid $(donor) doesn't match the checkpoint ($(name) is $(size(data)) with halos, expected $(size(parent(f))))")
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
##### Parameter files
#####

parameter_file(name) = joinpath(directory, "$(name)_parameters.toml")

# Every input of a run, written at its start; times in seconds, other quantities SI; `nothing` is written as "nothing"
function write_parameters(name, sections)
    toml_value(x) = isnothing(x) ? "nothing" : x isa Union{Type, Function} ? string(x) : x isa Tuple ? collect(toml_value.(x)) : x
    data =Dict{String, Any}(String(k) => Dict{String, Any}(String(n) => toml_value(v) for (n, v) in pairs(nt)) for (k, nt) in pairs(sections))
    data["run"] = Dict("name" => name)
    mkpath(directory)
    open(io -> TOML.print(io, data; sorted = true), parameter_file(name), "w")
    return nothing
end

function read_parameters(name)
    path = parameter_file(name)
    isfile(path) || error("no parameter file $(path). For a run that predates parameter files, write one with a [grid] section: " *
                          "n_lon, n_lat, n_depth, ocean_depth, max_lat, vertical_stretching, float_type")
    return TOML.parsefile(path)
end

# Add an entry to a run's parameter file each time it is restarted
function record_restart!(name, entry)
    data = read_parameters(name)
    push!(get!(data, "restarts", Any[]), Dict{String, Any}(String(k) => string(v) for (k, v) in pairs(entry)))
    delete!(data, "finished")    # running again
    open(io -> TOML.print(io, data; sorted = true), parameter_file(name), "w")
    return nothing
end

# Mark a run as finished in its parameter file, with the model time reached (s) and what stopped it
function record_finished!(name, model_time, stopped_by)
    data = read_parameters(name)
    data["finished"] = Dict{String, Any}("model_time" => model_time, "stopped_by" => stopped_by)
    open(io -> TOML.print(io, data; sorted = true), parameter_file(name), "w")
    return nothing
end

is_finished(name) = isfile(parameter_file(name)) && haskey(read_parameters(name), "finished")

# Latest checkpoint <name>_iteration<N>.jld2 of a run
function latest_checkpoint(name)
    dir = joinpath(directory, "checkpoints")
    matches = isdir(dir) ? filter(!isnothing, [match(Regex("^\\Q$(name)\\E_iteration(\\d+)\\.jld2\$"), f) for f in readdir(dir)]) : []
    isempty(matches) && error("no checkpoint of $(name) in $(dir)")
    return joinpath(dir, argmax(m -> parse(Int, m[1]), matches).match)
end

# The donor's grid from its parameters, with defaults for keys older runs didn't vary
function donor_grid(p)
    g = p["grid"]
    float_types = Dict("Float32" => Float32, "Float64" => Float64)
    return (n_lon = g["n_lon"], n_lat = g["n_lat"], n_depth = g["n_depth"], ocean_depth = Float64(g["ocean_depth"]),
            max_lat = get(g, "max_lat", 80), vertical_stretching = get(g, "vertical_stretching", 1.2),
            float_type = float_types[get(g, "float_type", "Float32")], halo = haskey(g, "halo") ? Tuple(g["halo"]) : nothing)
end

same_grid(a, b) = all(getfield(a, k) == getfield(b, k) for k in (:n_lon, :n_lat, :n_depth, :ocean_depth, :max_lat, :vertical_stretching, :float_type))

#####
##### Diagnostics and output
#####

#####
##### Deep-equilibrium stopping
#####

# Area-mean net surface heat flux into the ocean (W/m², positive warms) from the SST restoring, as a function of the current state
function surface_heat_flux(model, relaxation)
    grid = model.grid
    λ, φ = Array(λnodes(grid, Center())), Array(φnodes(grid, Center()))
    area = diff(sind.(Array(φnodes(grid, Face()))))    # cell area ∝ Δ(sin φ) at uniform Δλ
    weights = [a for _ in λ, a in area] ./ (length(λ) * sum(area))
    T_eq = [relaxation.T_night + (relaxation.T_day - relaxation.T_night) * max(0, cosd(l) * cosd(p)) for l in λ, p in φ]
    Nz = size(grid, 3)
    return () -> rho_seawater * cp_seawater * relaxation.W * sum(weights .* (T_eq .- Array(view(interior(model.tracers.T), :, :, Nz))))
end

# Gregory-style projection (Gregory et al. 2004, GRL 31, L03205): regress H on ⟨T⟩, extrapolate to H = 0, and return the remaining
# change T∞ − ⟨T⟩ (°C) and the mean H; the change is NaN when H doesn't fall as ⟨T⟩ rises, i.e. no drift is left to fit, only noise
function projected_change(T, H)
    b = sum((T .- mean(T)) .* (H .- mean(H))) / sum((T .- mean(T)) .^ 2)
    return (b < 0 ? mean(T) - mean(H) / b - T[end] : NaN), mean(H)
end

# Deep-equilibrium test over the last `window` of model time: stop once the projected remaining change of ⟨T⟩ is below `tol` (°C),
# or, when only noise is left to fit, once the window-mean surface heat flux is below `flux_tol` (W/m²)
function equilibrium_check(tol, window; flux_tol = 0.1)
    history = NTuple{3, Float64}[]
    function check!(sim, t, T̄, H)
        push!(history, (t, T̄, H))
        filter!(s -> t - s[1] <= window, history)
        (length(history) >= 3 && t - history[1][1] >= 0.99 * window) || return NaN
        gap, H̄ = projected_change(getindex.(history, 2), getindex.(history, 3))
        if isfinite(gap) ? abs(gap) < tol : abs(H̄) < flux_tol
            @info "Deep equilibrium: projected remaining change of ⟨T⟩ $(round(gap, sigdigits=2)) °C (tolerance $(tol) °C), mean surface " *
                  "heat flux $(round(H̄, sigdigits=2)) W/m² over the last $(round(window / 365days, digits=1)) years. Stopping at $(prettytime(t))."
            sim.running = false
        end
        return gap
    end
    return check!
end

#####
##### Diagnostics and output
#####

# Log progress every call, and stop at deep equilibrium if `equilibrium` is an `equilibrium_check`
function progress_callback(model, heat_flux, equilibrium)
    mean_T = Field(Average(model.tracers.T))
    mean_KE = Field(Average(@at (Center, Center, Center) (model.velocities.u^2 + model.velocities.v^2 + model.velocities.w^2) / 2))

    function progress(sim)
        compute!(mean_T)
        compute!(mean_KE)
        T̄ = Array(interior(mean_T))[1]
        KE = Array(interior(mean_KE))[1]
        H = heat_flux()
        t = sim.model.clock.time
        gap = isnothing(equilibrium) || !isfinite(T̄) ? NaN : equilibrium(sim, t, T̄, H)

        @info "Time: $(prettytime(t)), Δt: $(prettytime(sim.Δt)), CFL: $(round(AdvectiveCFL(sim.Δt)(sim.model), digits=3)), " *
              "⟨T⟩: $(round(T̄, digits=3)) °C, surface heat flux: $(round(H, sigdigits=3)) W/m², ⟨KE⟩: $(round(KE, sigdigits=3)) m²/s²" *
              (isfinite(gap) ? ", projected ⟨T⟩ change: $(round(gap, sigdigits=2)) °C" : "")
    end

    return progress
end

# append = true continues a restarted run's existing files
function add_output_writers!(simulation, name; output_interval, snapshot_interval, checkpoint_interval, global_attributes = Dict(), append = false)
    model = simulation.model
    ocean_dir = joinpath(directory)
    mkpath(ocean_dir)
    fields = (; T = model.tracers.T, u = model.velocities.u, v = model.velocities.v, w = model.velocities.w)

    if !isnothing(output_interval)
        simulation.output_writers[:full_3d] = NetCDFWriter(model, fields; filename = joinpath(ocean_dir, "$(name).nc"),
                                                           schedule = TimeInterval(output_interval), overwrite_existing = !append, global_attributes)
    end

    if !isnothing(snapshot_interval)
        Nz = size(model.grid, 3)
        section = map(f -> Field(Average(f, dims = 1)), fields)    # zonal mean (lat × z)
        simulation.output_writers[:section] = NetCDFWriter(model, section; filename = joinpath(ocean_dir, "$(name)_section.nc"),
                                                           schedule = TimeInterval(snapshot_interval), overwrite_existing = !append, global_attributes)
        # top cell (lat × lon); w on the top cell's lower face. The writer slices, so the file's z coordinate matches the outputs
        simulation.output_writers[:surface] = NetCDFWriter(model, fields; filename = joinpath(ocean_dir, "$(name)_surface.nc"), indices = (:, :, Nz),
                                                           schedule = TimeInterval(snapshot_interval), overwrite_existing = !append, global_attributes)
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
nightside). `rotational_period` is in Earth days (0 for no rotation). Output goes to `\$EXO_OCEAN_OUTPUT/` (default `output/`), including
`<name>_parameters.toml` with every input of the run.

Keywords:
- `resolution`: grid spacing in degrees, either one number or `(Δlon, Δlat)`; it must divide 360° and 2`max_lat` into whole cells.
- `max_lat`: the grid spans latitudes ±`max_lat` (< 90), with walls there.
- `n_depth`, `use_GPU`: number of vertical levels and architecture.
- `vertical_stretching`: thickness ratio of adjacent vertical cells (√1.2 with 2× `n_depth` halves every default cell).
- `float_type`: grid precision (Float64 avoids the Float32 WENO NaNs seen on the H100, at ~1.3× the cost there).
- `κ_horizontal`: horizontal horizontal tracer diffusivity (0 for none).
- `ν_vertical`, `κ_vertical`: background vertical viscosity and tracer diffusivity (m²/s).
- `wind_field`: `(lon, lat, t) -> (u_wind, v_wind)` in m/s for a bulk wind stress.
- `emulator_wind`, `surface_pressure`: force with CLERO-emulated winds of the planet whose emulated (T_day, T_night) match this run's at
  surface pressure `surface_pressure` (bar). The wind file is solved by `src/climate/climate_emulator.py` (Python env `\$EXO_EMULATOR_PYTHON`)
  and cached in `\$EXO_OCEAN_OUTPUT/winds/`; errors if no planet matches. Uses the ExoCAM emulator and the nearby M-dwarf mean [Fe/H];
  `wind_stress_scale` multiplies the stress.
- `salinity`: carry a passive salinity tracer initialised to this value (`nothing`: temperature only).
- `cfl`, `max_Δt`: advective CFL target and timestep cap (inertial stability).
- `restart`: name of a run to continue from its latest checkpoint for a further `simulation_time` (`simulation_name` must be the same).
  The grid comes from its parameter file (grid keywords and `ocean_depth` are ignored); the clock carries on and output is appended
  to its existing files. Other settings come from this call, so pass the same ones to continue the run unchanged.
- `warm_start`: name of a run whose latest checkpoint starts this new run (clock at 0, new output files). Its parameter file gives
  the donor grid: on the same grid the state is copied exactly, otherwise T, u, v are interpolated onto this run's grid.
- `equilibrium_tol`, `equilibrium_window`: stop at deep equilibrium, when the Gregory-style projected remaining change of volume-mean T
  over the window is below `equilibrium_tol` (°C); see `equilibrium_check`. `nothing` runs for the full `simulation_time`.
- `output_interval`, `snapshot_interval`, `checkpoint_interval`: model time between 3D snapshots, `_section` (zonal-mean lat × z) and
  `_surface` (top-cell lat × lon) snapshots, and checkpoints (`nothing` for none).
"""
function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, T_day, T_night, simulation_time;
                          resolution = 1.0, max_lat = 80, n_depth = 20,
                          use_GPU = true, float_type = Float32, vertical_stretching = 1.2,
                          κ_horizontal = 1e3, ν_vertical = 1e-4, κ_vertical = 1e-4,
                          wind_field = nothing, salinity = nothing,
                          emulator_wind = false, surface_pressure = nothing, wind_stress_scale = 1.0,
                          cfl = 0.2, max_Δt = nothing,
                          restart = nothing, warm_start = nothing,
                          equilibrium_tol = nothing, equilibrium_window = 10 * 365days,
                          output_interval = 5 * 365days, snapshot_interval = 36.5days, checkpoint_interval = nothing)

    @info "Setting up simulation $(simulation_name)..."
    isnothing(restart) || isnothing(warm_start) || error("pass either restart or warm_start, not both")
    isnothing(restart) || restart == simulation_name || error("restart continues the run named $(restart), so simulation_name must be the same")
    isnothing(warm_start) || warm_start != simulation_name || error("a warm start writes a new run, so simulation_name must differ from $(warm_start)")
    if isnothing(restart)
        0 < max_lat < 90 || error("max_lat must be between 0 and 90 (the grid's walls are at ±max_lat), got $(max_lat)")
        Δlon, Δlat = resolution isa Tuple ? resolution : (resolution, resolution)
        n_lon, n_lat = grid_cells(360, Δlon, "longitude"), grid_cells(2max_lat, Δlat, "latitude")
    else
        g = donor_grid(read_parameters(restart))
        (; n_lon, n_lat, n_depth, ocean_depth, max_lat, vertical_stretching, float_type) = g
        resolution = (360 / n_lon, 2max_lat / n_lat)
    end
    max_Δt = something(max_Δt, default_max_Δt(rotational_period, max_lat))

    # Sets up wind
    wind_file = nothing
    if emulator_wind  # solve before the grid exists, so the Python GPU job has released its memory
        isnothing(surface_pressure) && error("emulator_wind = true needs surface_pressure (bar)")
        isnothing(wind_field) || error("pass either wind_field or emulator_wind, not both")
        rotational_period > 0 || error("emulator_wind needs a rotating, tidally locked planet (rotational_period > 0)")
        wind_file = emulator_wind_file(rotational_period, T_day, T_night, surface_pressure)
    end

    # Sets up model settings
    arch = select_architecture(use_GPU)
    momentum = WENOVectorInvariant(order = 5)
    tracers_adv = WENO(order = 5)
    closure = ocean_closure(; κ_horizontal, ν_vertical, κ_vertical)
    halo = required_halo(momentum, tracers_adv, closure)
    grid = ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo, float_type, r = vertical_stretching, max_lat)

    rotation_rate = rotational_period == 0 ? 0 : omega_Earth / rotational_period
    eos = LinearEquationOfState(thermal_expansion = 2e-4, haline_contraction = 0.0)
    buoyancy = isnothing(salinity) ? SeawaterBuoyancy(equation_of_state = eos, constant_salinity = 35.0) : SeawaterBuoyancy(equation_of_state = eos)

    wind_stress = isnothing(wind_file) ? nothing : emulator_wind_stress(wind_file, grid; P_rot = rotational_period, scale = wind_stress_scale)
    u_bcs, v_bcs = velocity_boundary_conditions(wind_field, wind_stress)
    relaxation = (; T_day, T_night, W = Q_surface / (rho_seawater * cp_seawater))
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(surface_relaxation, discrete_form = true, parameters = relaxation))

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = momentum,
                                        tracer_advection = tracers_adv,
                                        coriolis = HydrostaticSphericalCoriolis(; rotation_rate),
                                        buoyancy,
                                        tracers = isnothing(salinity) ? (:T,) : (:T, :S),
                                        closure,
                                        free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = max_Δt),
                                        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs)
                                        )

    # Warm starts the model if a snapshot is provided    
    donor_parameters = isnothing(warm_start) ? nothing : read_parameters(warm_start)
    this_grid = (; n_lon, n_lat, n_depth, ocean_depth = Float64(ocean_depth), max_lat, vertical_stretching, float_type)
    regrid = !isnothing(warm_start) && !same_grid(donor_grid(donor_parameters), this_grid)
    if regrid
        @info "Warm-starting from $(latest_checkpoint(warm_start)), regridded from $(donor_grid(donor_parameters))"
        regrid_warm_start!(model, latest_checkpoint(warm_start), donor_parameters; planet_radius, T_night)
    elseif isnothing(restart) && isnothing(warm_start)
        set!(model, T = T_night)
    end

    isnothing(salinity) || set!(model, S = salinity)

    simulation = Simulation(model, Δt = 2minutes, stop_time = simulation_time)
    
    if !isnothing(warm_start) && !regrid
        @info "Warm-starting from $(latest_checkpoint(warm_start)) on the same grid"
        restore_checkpoint!(simulation, latest_checkpoint(warm_start); drop_salinity = isnothing(salinity), reset_clock = true)
    end

    if isnothing(restart)
        write_parameters(simulation_name, (
            planet = (; rotational_period, ocean_depth, planet_radius, T_day, T_night),
            grid = (; resolution, n_lon, n_lat, n_depth, ocean_depth, max_lat, vertical_stretching, float_type, halo,
                    z_faces = first(ocean_z_faces(ocean_depth, n_depth; r = vertical_stretching)), use_GPU, architecture = string(arch)),
            physics = (; κ_horizontal, ν_vertical, κ_vertical, heat_exchange_coefficient = Q_surface, C_Bottom_Drag, salinity,
                       wind_field = isnothing(wind_field) ? nothing : "custom function", emulator_wind, surface_pressure,
                       wind_file, wind_stress_scale),
            time = (; simulation_time, cfl, max_Δt, equilibrium_tol, equilibrium_window, output_interval, snapshot_interval, checkpoint_interval),
            warm_start = (; donor = warm_start, regridded = regrid,
                          checkpoint = isnothing(warm_start) ? nothing : latest_checkpoint(warm_start))))
    end

    equilibrium = isnothing(equilibrium_tol) ? nothing : equilibrium_check(equilibrium_tol, equilibrium_window)
    simulation.callbacks[:progress] = Callback(progress_callback(model, surface_heat_flux(model, relaxation), equilibrium), IterationInterval(100))
    simulation.callbacks[:wizard] = Callback(TimeStepWizard(; cfl, max_change = 1.05, max_Δt), IterationInterval(10))
    global_attributes = isnothing(wind_file) ? Dict() : emulator_attributes(wind_file, wind_stress_scale)
    add_output_writers!(simulation, simulation_name; output_interval, snapshot_interval, checkpoint_interval, global_attributes,
                        append = !isnothing(restart))

    if !isnothing(restart)    # after the writers exist, so their schedules are restored too
        checkpoint = latest_checkpoint(restart)
        restore_checkpoint!(simulation, checkpoint; drop_salinity = isnothing(salinity), reset_clock = false)
        simulation.stop_time = simulation.model.clock.time + simulation_time
        @info "Restarting from $(checkpoint) at $(prettytime(simulation.model.clock.time)) for a further $(prettytime(simulation_time))"
        record_restart!(simulation_name, (; checkpoint, from_time = simulation.model.clock.time, simulation_time))
    end

    @info "Simulation setup complete. Starting the run..."
    run!(simulation, checkpoint_at_end = !isnothing(checkpoint_interval))
    write_final_snapshot!(simulation)
    t = simulation.model.clock.time
    record_finished!(simulation_name, Float64(t), t >= simulation.stop_time ? "stop_time" : "equilibrium")
    @info "Simulation finished successfully!"

    return nothing
end
