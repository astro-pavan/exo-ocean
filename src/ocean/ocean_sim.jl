include("constants.jl")
using Oceananigans
using Oceananigans.Units
using Oceananigans.AbstractOperations: @at, Average
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Diagnostics: AdvectiveCFL, hasnan
using Oceananigans: UpdateStateCallsite
using Oceananigans.Fields: interpolate, compute!
using Oceananigans.Grids: λnode, φnode, znode
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ∂zᶜᶜᶠ
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, shear_squaredᶜᶜᶠ
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

# Bulk stress from the 10 m wind relative to α = 1 − s_w of the top-cell current (p from emulator_current_feedback)
@inline function current_feedback_stress_u(i, j, grid, clock, fields, p)
    k = grid.Nz
    Ux = @inbounds p.Ux_u[i, j] - p.α * fields.u[i, j, k]
    Uy = @inbounds p.Uy_u[i, j] - p.α * ℑxyᶠᶜᵃ(i, j, k, grid, fields.v)
    return @inbounds bulk_stress(Ux, Uy, p.ρ_u[i, j], Ux, p.scale)
end

@inline function current_feedback_stress_v(i, j, grid, clock, fields, p)
    k = grid.Nz
    Ux = @inbounds p.Ux_v[i, j] - p.α * ℑxyᶜᶠᵃ(i, j, k, grid, fields.u)
    Uy = @inbounds p.Uy_v[i, j] - p.α * fields.v[i, j, k]
    return @inbounds bulk_stress(Ux, Uy, p.ρ_v[i, j], Uy, p.scale)
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

# wind_stress = (τx, τy): kinematic stress arrays at u- and v-points (e.g. from emulator_wind_stress), used as array-valued top fluxes,
# or the parameters from emulator_current_feedback for a stress that depends on the surface current
function velocity_boundary_conditions(wind_field, wind_stress = nothing)
    u_bottom = FluxBoundaryCondition(bottom_drag_u, discrete_form = true)
    v_bottom = FluxBoundaryCondition(bottom_drag_v, discrete_form = true)
    if wind_stress isa NamedTuple
        return FieldBoundaryConditions(top = FluxBoundaryCondition(current_feedback_stress_u, discrete_form = true, parameters = wind_stress), bottom = u_bottom),
               FieldBoundaryConditions(top = FluxBoundaryCondition(current_feedback_stress_v, discrete_form = true, parameters = wind_stress), bottom = v_bottom)
    elseif !isnothing(wind_stress)
        return FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress[1]), bottom = u_bottom),
               FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress[2]), bottom = v_bottom)
    end
    isnothing(wind_field) && return FieldBoundaryConditions(bottom = u_bottom), FieldBoundaryConditions(bottom = v_bottom)
    p = (; wind_field)
    return FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress_u, parameters = p), bottom = u_bottom),
           FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress_v, parameters = p), bottom = v_bottom)
end

const VERTICAL_MIXING = (:catke, :convective_adjustment)
const HORIZONTAL_MIXING = (:gm_redi, :horizontal)
const THERMAL_EXPANSION = 2e-4    # 1/K

@inline function tidal_dissipation(P_rotation, eccentricty, r_planet, Q) # from Shi et al. (2025) and Barnes et al. (2013)
    return 0.02 * (P_rotation / 6.1) ^ -5 * (eccentricty / 0.005) ^ 2 * (r_planet / 0.92) ^ 3 * (Q / 84) ^ -1 # W / m^2
end

# Parameters of kappa_tidal, fixed for a run. ϵ0 normalises the profile so ∫ρϵ dh = tidal_fraction × D over the column (Polzin 2009, Ocean Modelling 30, 298)
function tidal_parameters(; rotational_period, ocean_depth, planet_radius, eccentricity, Q = 84, fraction = 0.3, z_p = 500, Γ = :constant, prandtl = 1, κ_max = 0.1)
    rotational_period > 0 || error("tidal mixing needs a synchronously rotating planet (rotational_period > 0)")
    Γ in (:constant, :richardson) || error("tidal Γ must be :constant or :richardson, got $(repr(Γ))")
    D = tidal_dissipation(rotational_period, eccentricity, planet_radius / R_Earth, Q)
    ϵ0 = fraction * D * (1 / ocean_depth + 1 / z_p) / rho_seawater
    return (; gα = g_Earth * THERMAL_EXPANSION, H = Float64(ocean_depth), z_p = Float64(z_p), ϵ0, Γ0 = 0.2, Ri_crt = 0.25, Ri_dependent = Γ == :richardson,
            prandtl = Float64(prandtl), κ_max = Float64(κ_max), S²_min = 1e-20, N²_min = 1e-8)
end

# Osborn (1980) κ = Γϵ/N² with a bottom-decaying dissipation; z_p = Inf gives a uniform ϵ. Smooth through N² = 0 (→ κ_max there) so it can't switch on and off
@inline function kappa_tidal(i, j, k, grid, clock, fields, p)
    N² = max(p.gα * ∂zᶜᶜᶠ(i, j, k, grid, fields.T), 0)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, fields)
    Ri = max(N² / (S² + p.S²_min), p.Ri_crt)    # Γ(Ri) held at its peak Γ0 below Ri_crt, else κ would jump from 0 to κ_max as N² → 0
    z = znode(i, j, k, grid, Center(), Center(), Face())
    h = z + p.H    # height above the seafloor

    ϵ = p.ϵ0 / ((1 + h / p.z_p) ^ 2)
    Γ = ifelse(p.Ri_dependent, 2 * p.Γ0 * sqrt(Ri / p.Ri_crt) / (1 + Ri / p.Ri_crt), p.Γ0)

    κ = Γ * ϵ / (N² + p.N²_min)
    return κ * p.κ_max / (κ + p.κ_max)    # harmonic cap, smooth unlike min
end

# The closure reads κ stored before each step: computed on the fly, it would see T and u that the implicit solve overwrites in place as it goes
@inline kappa_tidal_stored(i, j, k, grid, clock, fields, p) = @inbounds p.κ[i, j, k]
@inline viscosity_tidal_stored(i, j, k, grid, clock, fields, p) = @inbounds p.prandtl * p.κ[i, j, k]

# Refresh the stored tidal κ from the current state, before each step's tendencies and implicit solve
function add_tidal_κ_callback!(simulation, tidal, tidal_κ)
    model = simulation.model
    operation = KernelFunctionOperation{Center, Center, Face}(kappa_tidal, model.grid, model.clock, (; model.velocities.u, model.velocities.v, model.tracers.T), tidal)
    κ_computed = Field(operation; data = tidal_κ.data)
    compute!(κ_computed)
    simulation.callbacks[:tidal_κ] = Callback(_ -> compute!(κ_computed), callsite = UpdateStateCallsite())
    return nothing
end

# κ_horizontal = 0 omits the horizontal tracer diffusivity; vertical_mixing is added to the constant background of tracer diffusivity
# κ_vertical and viscosity prandtl × κ_vertical. :gm_redi uses κ_horizontal as both the GM skew and the Redi isopycnal diffusivity,
# tapered where isopycnal slopes exceed 0.01 (Danabasoglu & McWilliams 1995, J. Climate 8, 2967)
function ocean_closure(; κ_horizontal, κ_vertical = 1e-4, prandtl = 10, vertical_mixing = :catke, horizontal_mixing = :gm_redi, tidal = nothing)
    vertical_mixing in VERTICAL_MIXING || error("vertical_mixing must be one of $(VERTICAL_MIXING), got $(repr(vertical_mixing))")
    horizontal_mixing in HORIZONTAL_MIXING || error("horizontal_mixing must be one of $(HORIZONTAL_MIXING), got $(repr(horizontal_mixing))")
    horizontal = iszero(κ_horizontal) ? nothing :
                 horizontal_mixing == :gm_redi ? IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_horizontal, κ_symmetric = κ_horizontal) :    # Gent & McWilliams (1990), Redi (1982)
                 HorizontalScalarDiffusivity(κ = κ_horizontal)
    mixing = vertical_mixing == :catke ? CATKEVerticalDiffusivity() :    # Wagner et al. (2025, JAMES 17, e2024MS004522)
             ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0)
    tidal_mixing = isnothing(tidal) ? nothing : VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = kappa_tidal_stored, ν = viscosity_tidal_stored,
                                                                                         discrete_form = true, loc = (Center, Center, Face), parameters = tidal)
    closures = (VerticalScalarDiffusivity(ν = prandtl * κ_vertical, κ = κ_vertical), tidal_mixing, horizontal, mixing)
    return Tuple(c for c in closures if !isnothing(c))
end

# A run's vertical mixing scheme from its parameter file; older files predate the option, when convective adjustment was the only one
run_vertical_mixing(parameters) = Symbol(get(parameters["physics"], "vertical_mixing", "convective_adjustment"))

# A run's horizontal mixing scheme; older files predate the option, when geopotential (level) diffusion was the only one
run_horizontal_mixing(parameters) = Symbol(get(parameters["physics"], "horizontal_mixing", "horizontal"))

# Whether a run has tidal mixing, which adds a closure to the tuple; older files predate the option
run_tidal(parameters) = get(parameters["physics"], "tidal_eccentricity", "nothing") != "nothing"

# A run's tidal_parameters from its parameter file, or nothing if it has no tidal mixing
function run_tidal_parameters(parameters)
    run_tidal(parameters) || return nothing
    physics, planet = parameters["physics"], parameters["planet"]
    return tidal_parameters(; rotational_period = planet["rotational_period"], ocean_depth = planet["ocean_depth"], planet_radius = planet["planet_radius"],
                            eccentricity = physics["tidal_eccentricity"], Q = physics["tidal_Q"], fraction = physics["tidal_fraction"], z_p = physics["tidal_z_p"],
                            Γ = Symbol(physics["tidal_Γ"]), prandtl = physics["tidal_prandtl"], κ_max = physics["tidal_κ_max"])
end

# A run's background Prandtl number from its parameter file; older files set the viscosity directly
function run_prandtl(parameters)
    p = parameters["physics"]
    return haskey(p, "prandtl") ? Float64(p["prandtl"]) : p["ν_vertical"] / p["κ_vertical"]
end

# Momentum and tracer advection schemes; third-order upwind (Leonard 1979, CMAME 19, 59) adds spurious mixing (Marchesiello et al. 2009, Ocean Modelling 26, 156)
function advection_schemes(advection)
    advection == :weno5 && return WENOVectorInvariant(order = 5), WENO(order = 5)
    advection == :upwind3 || error("advection must be :weno5 or :upwind3, got $(repr(advection))")
    upwind = UpwindBiased(order = 3)
    return VectorInvariant(vorticity_scheme = upwind, vertical_advection_scheme = upwind, divergence_scheme = upwind), upwind
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

# Set T, u, v from the last 3D snapshot of a run on the same grid, for a donor with no checkpoint
function set_from_snapshot!(model, name)
    NCDataset(joinpath(directory, "$(name).nc")) do ds
        n = size(ds["T"], 4)
        set!(model.tracers.T, Array(ds["T"][:, :, :, n]))
        set!(model.velocities.u, Array(ds["u"][:, :, :, n]))
        set!(model.velocities.v, Array(ds["v"][:, :, :, n]))
    end
    return nothing
end

# Remove `key` from every NamedTuple in a loaded checkpoint state (e.g. :S for a model without salinity)
drop_state_key(x, key) = x
drop_state_key(nt::NamedTuple, key) = (; (name => drop_state_key(value, key) for (name, value) in pairs(nt) if name != key)...)

# Restore a same-grid checkpoint (model, Δt and the schedules of output writers already added), without the state keys in `drop`;
# new_closure skips the saved closure fields (for a different vertical mixing scheme); reset_clock starts a fresh run from it
function restore_checkpoint!(simulation, path; drop, new_closure = false, reset_clock)
    state = Oceananigans.OutputWriters.load_checkpoint_state(path)
    state = foldl(drop_state_key, drop; init = state)
    new_closure && (state = merge(state, (; model = merge(state.model, (; closure_fields = nothing)))))    # the model reads it by name
    Oceananigans.restore_prognostic_state!(simulation, state)
    if reset_clock
        simulation.model.clock.time = zero(simulation.model.clock.time)
        simulation.model.clock.iteration = 0
        simulation.Δt = 2minutes
    end
    return nothing
end

# Warm start from a checkpoint or a run's 3D snapshot file on another grid, described by the donor run's parameter file (see `read_parameters`).
# T, u, v are interpolated; below a shallower donor's floor u = v = 0 and T continues the donor's thermocline exponentially.
function regrid_warm_start!(model, source, donor_parameters; planet_radius, T_night = 0.0)
    donor = donor_grid(donor_parameters)
    from_snapshot = endswith(source, ".nc")
    m = from_snapshot ? nothing : Oceananigans.OutputWriters.load_checkpoint_state(source).model
    saved = from_snapshot ? nothing : size(parent(m.tracers.T.data))
    halo = from_snapshot ? something(donor.halo, (4, 4, 3)) : something(donor.halo, (saved .- (donor.n_lon, donor.n_lat, donor.n_depth)) .÷ 2)    # older parameter files have no halo
    all(halo .> 0) || error("the donor grid $(donor) doesn't match the checkpoint (T is $(saved) with halos)")

    src = ocean_grid(CPU(); donor.n_lon, donor.n_lat, donor.n_depth, donor.ocean_depth, planet_radius, halo,
                     float_type = donor.float_type, r = donor.vertical_stretching, max_lat = donor.max_lat)
    dst = on_architecture(CPU(), model.grid)

    T_src, u_src, v_src = CenterField(src), XFaceField(src), YFaceField(src)
    if from_snapshot
        NCDataset(source) do ds
            n = size(ds["T"], 4)
            for (f, name) in ((T_src, "T"), (u_src, "u"), (v_src, "v"))
                data = Array(ds[name][:, :, :, n])
                size(data) == size(interior(f)) || error("the donor grid $(donor) doesn't match $(source) ($(name) is $(size(data)), expected $(size(interior(f))))")
                interior(f) .= data
                fill_halo_regions!(f)
            end
        end
    else
        for (f, saved, name) in ((T_src, m.tracers.T, :T), (u_src, m.velocities.u, :u), (v_src, m.velocities.v, :v))
            data = Array(parent(saved.data))
            size(data) == size(parent(f)) || error("the donor grid $(donor) doesn't match the checkpoint ($(name) is $(size(data)) with halos, expected $(size(parent(f))))")
            copyto!(parent(f), data)
            fill_halo_regions!(f)
        end
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

# A finished run that stopped on a NaN (older files recorded those as "equilibrium" with a NaN model_time)
blew_up(name) = is_finished(name) && (f = read_parameters(name)["finished"]; f["stopped_by"] == "nan" || !isfinite(f["model_time"]))

# Latest checkpoint <name>_iteration<N>.jld2 of a run
function latest_checkpoint(name)
    dir = joinpath(directory, "checkpoints")
    matches = isdir(dir) ? filter(!isnothing, [match(Regex("^\\Q$(name)\\E_iteration(\\d+)\\.jld2\$"), f) for f in readdir(dir)]) : []
    isempty(matches) && error("no checkpoint of $(name) in $(dir)")
    return joinpath(dir, argmax(m -> parse(Int, m[1]), matches).match)
end

has_checkpoint(name) = isdir(joinpath(directory, "checkpoints")) &&
                       any(f -> occursin(Regex("^\\Q$(name)\\E_iteration\\d+\\.jld2\$"), f), readdir(joinpath(directory, "checkpoints")))

# What a run can be warm-started from: its latest checkpoint, or else its 3D snapshot file, whose last frame holds the final state
function warm_start_source(name)
    has_checkpoint(name) && return latest_checkpoint(name)
    path = joinpath(directory, "$(name).nc")
    isfile(path) || error("$(name) has neither a checkpoint nor a 3D snapshot file")
    return path
end

function remove_checkpoints!(name)
    dir = joinpath(directory, "checkpoints")
    isdir(dir) || return nothing
    foreach(f -> occursin(Regex("^\\Q$(name)\\E_iteration\\d+\\.jld2\$"), f) && rm(joinpath(dir, f)), readdir(dir))
    return nothing
end

#####
##### Output compression
#####

# Copy a NetCDF dataset or group, deflating and byte-shuffling its float fields of two or more dimensions
function copy_compressed!(src, dst; deflatelevel)
    foreach(((k, v),) -> dst.attrib[k] = v, src.attrib)
    unlimited = NCDatasets.unlimited(src.dim)
    foreach(((name, len),) -> defDim(dst, name, name in unlimited ? Inf : len), src.dim)
    for name in keys(src)
        v = src[name].var
        dims = NCDatasets.dimnames(v)
        compress = ndims(v) >= 2 && eltype(v) <: AbstractFloat
        by_frame = compress && last(dims) in unlimited
        options = compress ? (; deflatelevel, shuffle = true) : (;)
        by_frame && (options = merge(options, (; chunksizes = (size(v)[1:end-1]..., 1))))
        out = defVar(dst, name, eltype(v), dims; attrib = collect(v.attrib), options...).var
        if by_frame
            for n in 1:size(v, ndims(v))
                out[ntuple(_ -> :, ndims(v) - 1)..., n] = v[ntuple(_ -> :, ndims(v) - 1)..., n]
            end
        elseif ndims(v) == 0
            out[] = v[]
        else
            ranges = map(n -> 1:n, size(v))
            out[ranges...] = v[ranges...]
        end
    end
    for name in keys(src.group)
        copy_compressed!(src.group[name], defGroup(dst, name); deflatelevel)
    end
    return nothing
end

# Replace a finished NetCDF file with a compressed copy, only after checking the copy's last frame of T; a failure leaves the original
function compress_netcdf!(path; deflatelevel = 1)
    copy_path = path * ".compressing"
    try
        NCDataset(path) do src
            NCDataset(copy_path, "c") do dst
                copy_compressed!(src, dst; deflatelevel)
            end
        end
        same = NCDataset(path) do a
            NCDataset(copy_path) do b
                n = size(a["T"], ndims(a["T"]))
                size(a["T"]) == size(b["T"]) && isequal(a["T"].var[ntuple(_ -> :, ndims(a["T"]) - 1)..., n], b["T"].var[ntuple(_ -> :, ndims(b["T"]) - 1)..., n])
            end
        end
        same || error("the compressed copy differs from the original")
        before, after = filesize(path), filesize(copy_path)
        mv(copy_path, path; force = true)
        @info "Compressed $(basename(path)): $(round(before / 1e6, digits = 1)) → $(round(after / 1e6, digits = 1)) MB"
    catch err
        @warn "Couldn't compress $(path); keeping the original" exception = err
        isfile(copy_path) && rm(copy_path)
    end
    return nothing
end

function compress_run_output!(simulation, name)
    for key in (:full_3d, :section, :surface)
        haskey(simulation.output_writers, key) && try close(simulation.output_writers[key]) catch end    # release the file before replacing it
    end
    foreach(suffix -> (path = joinpath(directory, "$(name)$(suffix).nc"); isfile(path) && compress_netcdf!(path)), ("", "_section", "_surface"))
    return nothing
end

# Model time (s) a run is meant to reach: the end of its last restart, or else its original simulation_time
function planned_end(p)
    restarts = get(p, "restarts", Any[])
    isempty(restarts) && return Float64(p["time"]["simulation_time"])
    return parse(Float64, restarts[end]["from_time"]) + parse(Float64, restarts[end]["simulation_time"])
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

# Deep-equilibrium test over the last `window` of model time, once the first window (the fast surface adjustment, which steepens the
# fit and hides the slow deep drift; Rugenstein et al. 2020, GRL 47, e2019GL083898) has passed: stop when both the projected remaining
# change of ⟨T⟩ (if there is drift left to fit) and the change one more window at the window-mean flux would make are below `tol` (°C).
# `heat_capacity` is ρ c_p × depth (J/m²/K), so the second test is H̄ window / heat_capacity < tol.
function equilibrium_check(tol, window, heat_capacity)
    history = NTuple{3, Float64}[]
    function check!(sim, t, T̄, H)
        push!(history, (t, T̄, H))
        filter!(s -> t - s[1] <= window, history)
        (length(history) >= 3 && t >= 1.99 * window && t - history[1][1] >= 0.99 * window) || return NaN
        gap, H̄ = projected_change(getindex.(history, 2), getindex.(history, 3))
        if (!isfinite(gap) || abs(gap) < tol) && abs(H̄) * window / heat_capacity < tol
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
# Output at multiples of `interval` back from the end `t_end`, over the last `window` of model time
last_times(interval, t_end, window) = [t_end - n * interval for n in floor(Int, window / interval):-1:0 if t_end - n * interval > 0]
writer_schedule(interval, save_times) = isnothing(save_times) ? TimeInterval(interval) : SpecifiedTimes(last_times(interval, save_times...))

# save_times = (end of the run, window): keep the 3D and surface output for that window only; the small zonal-mean `_section` file covers the whole run
function add_output_writers!(simulation, name; output_interval, snapshot_interval, checkpoint_interval, global_attributes = Dict(), append = false, save_times = nothing)
    model = simulation.model
    ocean_dir = joinpath(directory)
    mkpath(ocean_dir)
    fields = (; T = model.tracers.T, u = model.velocities.u, v = model.velocities.v, w = model.velocities.w)

    if !isnothing(output_interval)
        simulation.output_writers[:full_3d] = NetCDFWriter(model, fields; filename = joinpath(ocean_dir, "$(name).nc"),
                                                           schedule = writer_schedule(output_interval, save_times), overwrite_existing = !append, global_attributes)
    end

    if !isnothing(snapshot_interval)
        Nz = size(model.grid, 3)
        section = map(f -> Field(Average(f, dims = 1)), fields)    # zonal mean (lat × z)
        simulation.output_writers[:section] = NetCDFWriter(model, section; filename = joinpath(ocean_dir, "$(name)_section.nc"),
                                                           schedule = TimeInterval(snapshot_interval), overwrite_existing = !append, global_attributes)
        # top cell (lat × lon); w on the top cell's lower face. The writer slices, so the file's z coordinate matches the outputs
        simulation.output_writers[:surface] = NetCDFWriter(model, fields; filename = joinpath(ocean_dir, "$(name)_surface.nc"), indices = (:, :, Nz),
                                                           schedule = writer_schedule(snapshot_interval, save_times), overwrite_existing = !append, global_attributes)
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
    schedule = writer.schedule
    last_write = schedule isa SpecifiedTimes ? (schedule.previous_actuation > 0 ? schedule.times[schedule.previous_actuation] : -Inf) :
                                               schedule.first_actuation_time + schedule.actuations * schedule.interval
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
- `κ_horizontal`: horizontal tracer diffusivity (m²/s; 0 for none), or with GM-Redi both its eddy-thickness and isopycnal diffusivities.
- `horizontal_mixing`: `:gm_redi` (default; Gent & McWilliams 1990 eddy advection plus Redi 1982 diffusion along isopycnals, slopes
  above 0.01 tapered) or `:horizontal` (diffusion along level surfaces, which mixes across sloping isotherms). A restart keeps its
  run's scheme (`:horizontal` for runs that predate the option), so leave it as `nothing` or pass the same one.
- `advection`: `:weno5` (default) or `:upwind3`, third-order upwind for momentum and tracers (cheaper and more numerically diffusive; intended for spin-ups).
- `κ_vertical`: background vertical tracer diffusivity (m²/s).
- `prandtl`: background Prandtl number, so the background vertical viscosity is `prandtl × κ_vertical` (default 10, as in the
  Pacanowski & Philander 1981 and KPP backgrounds). A restart keeps its run's value (ν/κ for runs that set ν directly), so leave it as
  `nothing` or pass the same one.
- `vertical_mixing`: `:catke` (default; CATKE, Wagner et al. 2025) or `:convective_adjustment`, added to the background. A restart
  keeps its run's scheme (convective adjustment for runs that predate the option), so leave it as `nothing` or pass the same one.
- `tidal_eccentricity`: adds tidal mixing (`kappa_tidal`) on top of the background and `vertical_mixing`; `nothing` (default) has none. The column
  energy is `tidal_fraction` × `tidal_dissipation(rotational_period, tidal_eccentricity, radius, tidal_Q)`, so it scales the mixing with `tidal_Q`.
- `tidal_z_p`: decay height (m) of the Polzin (2009) dissipation profile above the floor; `Inf` for a uniform ϵ. `tidal_Γ`: `:constant` (0.2) or
  `:richardson` (Γ(Ri)). `tidal_prandtl` sets the viscosity as a multiple of the tidal κ, and `tidal_κ_max` caps κ (m²/s). Pass the same ones on a restart.
- `wind_field`: `(lon, lat, t) -> (u_wind, v_wind)` in m/s for a bulk wind stress.
- `emulator_wind`, `surface_pressure`: force with CLERO-emulated winds of the planet whose emulated (T_day, T_night) match this run's at
  surface pressure `surface_pressure` (bar). The wind file is solved by `src/climate/climate_emulator.py` (Python env `\$EXO_EMULATOR_PYTHON`)
  and cached in `\$EXO_OCEAN_OUTPUT/winds/`; errors if no planet matches. Uses the ExoCAM emulator and the nearby M-dwarf mean [Fe/H];
  `wind_stress_scale` multiplies the stress.
- `wind_current_coupling`: s_w, how far the emulator's wind follows the ocean surface current (Renault et al. 2016, JPO 46, 1685, eq. 19;
  2020, JAMES 12, e2019MS001715). The stress uses the wind relative to (1 − s_w) × the top-cell current: 1 (default) ignores the
  current (absolute wind), 0 uses the full relative wind, and 0.3 is Renault et al. (2020)'s global mean for a forced ocean.
- `salinity`: carry a passive salinity tracer initialised to this value (`nothing`: temperature only).
- `cfl`, `max_Δt`: advective CFL target and timestep cap (inertial stability).
- `free_surface_cfl`: barotropic wave CFL, which sets the number of free-surface substeps.
- `wall_time_limit`: stop (with a checkpoint) after this many seconds of wall time; `finished.stopped_by` is then `"wall_time"`.
- `save_last`: keep the 3D and surface output for only this last stretch of model time (`nothing`: the whole run); `_section` always covers the whole run.
- `keep_checkpoints`: `false` deletes the run's checkpoints once it finishes normally, keeping them after a NaN or wall-time stop; other runs can warm-start from its 3D snapshots.
- `compress_output`: deflate and byte-shuffle the run's NetCDF files once it finishes.
- `restart`: name of a run to continue from its latest checkpoint for a further `simulation_time` (`simulation_name` must be the same).
  The grid comes from its parameter file (grid keywords and `ocean_depth` are ignored); the clock carries on and output is appended
  to its existing files. Other settings come from this call, so pass the same ones to continue the run unchanged.
- `resume`: with `restart`, run on to the run's planned end (see `planned_end`) instead of a further `simulation_time`, which is ignored;
  used to pick up a run cut off by a job's wall time.
- `warm_start`: name of a run whose latest checkpoint starts this new run (clock at 0, new output files). Its parameter file gives
  the donor grid: on the same grid the state is copied exactly, otherwise T, u, v are interpolated onto this run's grid.
- `initial_snapshot`: name of a run on the same grid whose last 3D snapshot gives T, u, v for a new run with no checkpoint to start from (clock at 0); TKE and
  the free surface start fresh and adjust in the first steps.
- `equilibrium_tol`, `equilibrium_window`: stop at deep equilibrium, when, after the first window, both the Gregory-style projected
  remaining change of volume-mean T and its drift over one more window are below `equilibrium_tol` (°C); see `equilibrium_check`. `nothing` runs for the full `simulation_time`.
- `output_interval`, `snapshot_interval`, `checkpoint_interval`: model time between 3D snapshots, `_section` (zonal-mean lat × z) and
  `_surface` (top-cell lat × lon) snapshots, and checkpoints (`nothing` for none).
"""
function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, T_day, T_night, simulation_time;
                          resolution = 1.0, max_lat = 80, n_depth = 20,
                          use_GPU = true, float_type = Float32, vertical_stretching = 1.2,
                          κ_horizontal = 1e3, κ_vertical = 1e-4, prandtl = nothing, vertical_mixing = nothing, horizontal_mixing = nothing, advection = :weno5,
                          tidal_eccentricity = nothing, tidal_Q = 84, tidal_fraction = 0.3, tidal_z_p = 500, tidal_Γ = :constant, tidal_prandtl = 1, tidal_κ_max = 0.1,
                          wind_field = nothing, salinity = nothing,
                          emulator_wind = false, surface_pressure = nothing, wind_stress_scale = 1.0, wind_current_coupling = 1.0,
                          cfl = 0.2, max_Δt = nothing, free_surface_cfl = 0.7,
                          restart = nothing, resume = false, warm_start = nothing, initial_snapshot = nothing,
                          equilibrium_tol = nothing, equilibrium_window = 10 * 365days,
                          output_interval = 5 * 365days, snapshot_interval = 36.5days, checkpoint_interval = nothing, wall_time_limit = Inf,
                          save_last = nothing, keep_checkpoints = true, compress_output = false)

    @info "Setting up simulation $(simulation_name)..."
    isnothing(restart) || isnothing(warm_start) || error("pass either restart or warm_start, not both")
    isnothing(initial_snapshot) || (isnothing(restart) && isnothing(warm_start)) || error("initial_snapshot starts a new run, so it can't go with restart or warm_start")
    isnothing(restart) || restart == simulation_name || error("restart continues the run named $(restart), so simulation_name must be the same")
    !resume || !isnothing(restart) || error("resume continues a run, so it needs restart")
    isnothing(warm_start) || warm_start != simulation_name || error("a warm start writes a new run, so simulation_name must differ from $(warm_start)")
    if isnothing(restart)
        0 < max_lat < 90 || error("max_lat must be between 0 and 90 (the grid's walls are at ±max_lat), got $(max_lat)")
        Δlon, Δlat = resolution isa Tuple ? resolution : (resolution, resolution)
        n_lon, n_lat = grid_cells(360, Δlon, "longitude"), grid_cells(2max_lat, Δlat, "latitude")
    else
        restart_parameters = read_parameters(restart)
        g = donor_grid(restart_parameters)
        (; n_lon, n_lat, n_depth, ocean_depth, max_lat, vertical_stretching, float_type) = g
        resolution = (360 / n_lon, 2max_lat / n_lat)
        run_mixing = run_vertical_mixing(restart_parameters)
        isnothing(vertical_mixing) || Symbol(vertical_mixing) == run_mixing ||
            error("$(restart) ran with vertical_mixing = :$(run_mixing), so it can't continue with :$(vertical_mixing)")
        vertical_mixing = run_mixing
        run_horizontal = run_horizontal_mixing(restart_parameters)
        isnothing(horizontal_mixing) || Symbol(horizontal_mixing) == run_horizontal ||
            error("$(restart) ran with horizontal_mixing = :$(run_horizontal), so it can't continue with :$(horizontal_mixing)")
        horizontal_mixing = run_horizontal
        run_pr = run_prandtl(restart_parameters)
        isnothing(prandtl) || isapprox(prandtl, run_pr; rtol = 1e-6) ||
            error("$(restart) ran with prandtl = $(run_pr), so it can't continue with $(prandtl)")
        prandtl = run_pr
    end
    vertical_mixing = Symbol(something(vertical_mixing, :catke))
    horizontal_mixing = Symbol(something(horizontal_mixing, :gm_redi))
    prandtl = Float64(something(prandtl, 10))
    ν_vertical = prandtl * κ_vertical
    max_Δt = something(max_Δt, default_max_Δt(rotational_period, max_lat))

    # Sets up wind
    wind_file = nothing
    if emulator_wind  # solve before the grid exists, so the Python GPU job has released its memory
        isnothing(surface_pressure) && error("emulator_wind = true needs surface_pressure (bar)")
        isnothing(wind_field) || error("pass either wind_field or emulator_wind, not both")
        rotational_period > 0 || error("emulator_wind needs a rotating, tidally locked planet (rotational_period > 0)")
        wind_file = emulator_wind_file(rotational_period, T_day, T_night, surface_pressure)
    end
    wind_current_coupling == 1 || emulator_wind || error("wind_current_coupling needs emulator_wind = true")

    # Sets up model settings
    arch = select_architecture(use_GPU)
    momentum, tracers_adv = advection_schemes(advection)
    tidal = isnothing(tidal_eccentricity) ? nothing : tidal_parameters(; rotational_period, ocean_depth, planet_radius, eccentricity = tidal_eccentricity, Q = tidal_Q,
                                                                       fraction = tidal_fraction, z_p = tidal_z_p, Γ = Symbol(tidal_Γ), prandtl = tidal_prandtl, κ_max = tidal_κ_max)
    halo = required_halo(momentum, tracers_adv, ocean_closure(; κ_horizontal, κ_vertical, prandtl, vertical_mixing, horizontal_mixing))    # tidal mixing needs no more halo
    grid = ocean_grid(arch; n_lon, n_lat, n_depth, ocean_depth, planet_radius, halo, float_type, r = vertical_stretching, max_lat)
    tidal_κ = isnothing(tidal) ? nothing : Field{Center, Center, Face}(grid)
    closure = ocean_closure(; κ_horizontal, κ_vertical, prandtl, vertical_mixing, horizontal_mixing, tidal = isnothing(tidal) ? nothing : (; κ = tidal_κ, tidal.prandtl))

    rotation_rate = rotational_period == 0 ? 0 : omega_Earth / rotational_period
    eos = LinearEquationOfState(thermal_expansion = THERMAL_EXPANSION, haline_contraction = 0.0)
    buoyancy = isnothing(salinity) ? SeawaterBuoyancy(equation_of_state = eos, constant_salinity = 35.0) : SeawaterBuoyancy(equation_of_state = eos)

    wind_stress = isnothing(wind_file) ? nothing :
                  wind_current_coupling == 1 ? emulator_wind_stress(wind_file, grid; P_rot = rotational_period, scale = wind_stress_scale) :
                  emulator_current_feedback(wind_file, grid, wind_current_coupling; P_rot = rotational_period, scale = wind_stress_scale)
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
                                        free_surface = SplitExplicitFreeSurface(grid; cfl = free_surface_cfl, fixed_Δt = max_Δt),
                                        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs)
                                        )

    # Warm starts the model if a snapshot is provided    
    donor_parameters = isnothing(warm_start) ? nothing : read_parameters(warm_start)
    this_grid = (; n_lon, n_lat, n_depth, ocean_depth = Float64(ocean_depth), max_lat, vertical_stretching, float_type)
    regrid = !isnothing(warm_start) && !same_grid(donor_grid(donor_parameters), this_grid)
    if !isnothing(initial_snapshot)
        @info "Starting from the last 3D snapshot of $(initial_snapshot)"
        set_from_snapshot!(model, initial_snapshot)
    elseif regrid
        @info "Warm-starting from $(warm_start_source(warm_start)), regridded from $(donor_grid(donor_parameters))"
        regrid_warm_start!(model, warm_start_source(warm_start), donor_parameters; planet_radius, T_night)
    elseif !isnothing(warm_start) && !has_checkpoint(warm_start)
        @info "Warm-starting from the last 3D snapshot of $(warm_start) on the same grid"
        set_from_snapshot!(model, warm_start)
    elseif isnothing(restart) && isnothing(warm_start)
        set!(model, T = T_night)
    end

    isnothing(salinity) || set!(model, S = salinity)

    simulation = Simulation(model, Δt = 2minutes, stop_time = simulation_time, wall_time_limit = Float64(wall_time_limit))
    isnothing(tidal) || add_tidal_κ_callback!(simulation, tidal, tidal_κ)
    no_salinity = isnothing(salinity) ? (:S,) : ()

    if !isnothing(warm_start) && !regrid && has_checkpoint(warm_start)
        @info "Warm-starting from $(latest_checkpoint(warm_start)) on the same grid"
        new_vertical = run_vertical_mixing(donor_parameters) != vertical_mixing    # the new scheme starts its own TKE and diffusivities
        new_closure = new_vertical || run_horizontal_mixing(donor_parameters) != horizontal_mixing || run_tidal(donor_parameters) != !isnothing(tidal)
        restore_checkpoint!(simulation, latest_checkpoint(warm_start); drop = (no_salinity..., (new_vertical ? (:e,) : ())...),
                            new_closure, reset_clock = true)
    end

    if isnothing(restart)
        write_parameters(simulation_name, (
            planet = (; rotational_period, ocean_depth, planet_radius, T_day, T_night),
            grid = (; resolution, n_lon, n_lat, n_depth, ocean_depth, max_lat, vertical_stretching, float_type, halo,
                    z_faces = first(ocean_z_faces(ocean_depth, n_depth; r = vertical_stretching)), use_GPU, architecture = string(arch)),
            physics = (; advection = string(advection), κ_horizontal, horizontal_mixing = string(horizontal_mixing), κ_vertical, prandtl, ν_vertical, vertical_mixing = string(vertical_mixing),
                       tidal_eccentricity, tidal_Q, tidal_fraction, tidal_z_p, tidal_Γ = string(tidal_Γ), tidal_prandtl, tidal_κ_max,
                       heat_exchange_coefficient = Q_surface, C_Bottom_Drag, salinity, wind_field = isnothing(wind_field) ? nothing : "custom function", emulator_wind, surface_pressure,
                       wind_file, wind_stress_scale, wind_current_coupling),
            time = (; simulation_time, cfl, max_Δt, free_surface_cfl, equilibrium_tol, equilibrium_window, output_interval, snapshot_interval, checkpoint_interval,
                    save_last, keep_checkpoints, compress_output),
            warm_start = (; donor = warm_start, regridded = regrid, initial_snapshot,
                          checkpoint = isnothing(warm_start) ? nothing : warm_start_source(warm_start))))
    end

    equilibrium = isnothing(equilibrium_tol) ? nothing : equilibrium_check(equilibrium_tol, equilibrium_window, rho_seawater * cp_seawater * ocean_depth)
    simulation.callbacks[:progress] = Callback(progress_callback(model, surface_heat_flux(model, relaxation), equilibrium), IterationInterval(100))
    simulation.callbacks[:wizard] = Callback(TimeStepWizard(; cfl, max_change = 1.05, max_Δt), IterationInterval(10))
    global_attributes = isnothing(wind_file) ? Dict() : emulator_attributes(wind_file, wind_stress_scale, wind_current_coupling)
    run_end = isnothing(restart) ? simulation_time : resume ? planned_end(read_parameters(restart)) : nothing
    !isnothing(save_last) && isnothing(run_end) && @warn "save_last is ignored when extending a finished run; saving all its output"
    save_times = isnothing(save_last) || isnothing(run_end) ? nothing : (run_end, save_last)
    add_output_writers!(simulation, simulation_name; output_interval, snapshot_interval, checkpoint_interval, global_attributes,
                        append = !isnothing(restart), save_times)

    if !isnothing(restart)    # after the writers exist, so their schedules are restored too
        checkpoint = latest_checkpoint(restart)
        restore_checkpoint!(simulation, checkpoint; drop = no_salinity, reset_clock = false)
        t₀ = simulation.model.clock.time
        if resume
            simulation_time = planned_end(read_parameters(restart)) - t₀
            simulation_time > 0 || (record_finished!(simulation_name, Float64(t₀), "stop_time"); return nothing)    # cut off after its last checkpoint
        end
        simulation.stop_time = t₀ + simulation_time
        @info "Restarting from $(checkpoint) at $(prettytime(t₀)) for a further $(prettytime(simulation_time))"
        record_restart!(simulation_name, (; checkpoint, from_time = simulation.model.clock.time, simulation_time))
    end

    @info "Simulation setup complete. Starting the run..."
    run!(simulation)
    t = simulation.model.clock.time
    if !isfinite(t) || hasnan(model.velocities.u) || hasnan(model.tracers.T)    # the NaN checker stopped it; keep the last good checkpoint
        record_finished!(simulation_name, Float64(t), "nan")
        @error "$(simulation_name) blew up (NaN) at $(prettytime(t)); its latest checkpoint is $(has_checkpoint(simulation_name) ? latest_checkpoint(simulation_name) : "none")"
        return nothing
    end
    stopped_by = t >= simulation.stop_time ? "stop_time" : simulation.run_wall_time >= wall_time_limit ? "wall_time" : "equilibrium"
    remove_checkpoints = !keep_checkpoints && stopped_by != "wall_time"
    isnothing(checkpoint_interval) || remove_checkpoints || Oceananigans.OutputWriters.checkpoint(simulation)
    write_final_snapshot!(simulation)
    compress_output && stopped_by != "wall_time" && compress_run_output!(simulation, simulation_name)
    remove_checkpoints && remove_checkpoints!(simulation_name)
    record_finished!(simulation_name, Float64(t), stopped_by)
    @info "Simulation finished successfully!"

    return nothing
end
