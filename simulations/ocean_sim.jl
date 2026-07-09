include("constants.jl")
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.AbstractOperations: @at, Average
using Statistics
const _cuda_available = try
    using CUDA
    CUDA.functional()
catch
    false
end

const _amdgpu_available = !_cuda_available && try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

const directory = "exo_ocean_sims/"

@inline function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, T_day, T_night, simulation_time; n_lat=160, n_lon=360, n_depth=20, use_GPU=true, n_write=1000, wind_field=nothing, initial_T=nothing, τ_relax=30days, τ_biharmonic=10days, κ_horizontal=1e3, pickup=nothing, checkpoint_interval=nothing, instellation=nothing, albedo=0.06)

    @info "Setting up simulation..."

    arch = if !use_GPU
        CPU()
    elseif _cuda_available
        @info "CUDA GPU detected, using CUDA backend."
        GPU()
    elseif _amdgpu_available
        @info "AMD GPU detected, using AMDGPU backend."
        GPU()
    else
        @warn "No functional GPU found, falling back to CPU."
        CPU()
    end

    # Grid — exponential stretching with constant ratio r between adjacent cells.
    # Δz₀ is derived so the faces span exactly ocean_depth.
    r = 1.2
    Δz₀ = ocean_depth * (r - 1) / (r^n_depth - 1)
    z_faces = [-Δz₀ * (r^(n_depth - k) - 1) / (r - 1) for k in 0:n_depth]

    grid = LatitudeLongitudeGrid(arch, Float32,
                                size = (n_lon, n_lat, n_depth),
                                longitude = (-180, 180),
                                latitude = (-80, 80),
                                z = z_faces,
                                radius = planet_radius,
                                topology = (Periodic, Bounded, Bounded))

    # Coriolis Force
    if rotational_period != 0
        coriolis = HydrostaticSphericalCoriolis(rotation_rate = omega_Earth / rotational_period)
    else
        coriolis = HydrostaticSphericalCoriolis(rotation_rate = 0)
    end

    # Linear Equation of State
    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=0.0))

    # Bottom Drag
    @inline function bottom_drag_u(i, j, grid, clock, model_fields)
        u = @inbounds model_fields.u[i, j, 1]
        v = @inbounds model_fields.v[i, j, 1]
        return -C_Bottom_Drag * u * sqrt(u^2 + v^2)
    end

    @inline function bottom_drag_v(i, j, grid, clock, model_fields)
        u = @inbounds model_fields.u[i, j, 1]
        v = @inbounds model_fields.v[i, j, 1]
        return -C_Bottom_Drag * v * sqrt(u^2 + v^2)
    end

    u_bottom_bc = FluxBoundaryCondition(bottom_drag_u, discrete_form=true)
    v_bottom_bc = FluxBoundaryCondition(bottom_drag_v, discrete_form=true)

    # Wind Forcing
    # wind_field(lon, lat, t) returns (u_wind, v_wind) in m/s, the zonal and
    # meridional wind speed at the ocean surface, converted to a wind stress
    # via the standard quadratic bulk formula.
    @inline function wind_stress_u(lon, lat, t, p)
        u_wind, v_wind = p.wind_field(lon, lat, t)
        wind_speed = sqrt(u_wind^2 + v_wind^2)
        return -rho_air * C_D_wind * wind_speed * u_wind / rho_seawater
    end

    @inline function wind_stress_v(lon, lat, t, p)
        u_wind, v_wind = p.wind_field(lon, lat, t)
        wind_speed = sqrt(u_wind^2 + v_wind^2)
        return -rho_air * C_D_wind * wind_speed * v_wind / rho_seawater
    end

    # Boundary Conditions
    if !isnothing(wind_field)
        wind_params = (; wind_field)
        u_top_bc = FluxBoundaryCondition(wind_stress_u, parameters = wind_params)
        v_top_bc = FluxBoundaryCondition(wind_stress_v, parameters = wind_params)
        u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc)
        v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc)
    else
        u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
        v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)
    end

    # Surface Temperature Forcing
    # Default: Newtonian relaxation ("restoring") of sea surface temperature toward a
    # tidally-locked day/night equilibrium temperature, hottest at the substellar point
    # (lon=0°, lat=0°) and coldest at the antistellar point — mirrors the Newtonian
    # cooling scheme used for the atmosphere in atmosphere_sim.jl.
    #
    # This is the standard Haney (1971) restoring formulation: the top boundary's
    # kinematic flux is λ * (T_surface - T_eq), with piston velocity λ = Δz_top / τ_relax
    # (Δz_top being the top grid cell's thickness), which reproduces a relaxation
    # tendency of (T_eq - T_surface) / τ_relax in the surface cell.
    @inline function surface_temp_relaxation(lon, lat, t, T, p)
        lon_rad = lon * π / 180
        lat_rad = lat * π / 180
        T_eq = p.T_night + (p.T_day - p.T_night) * max(zero(lon_rad), cos(lon_rad) * cos(lat_rad))
        return p.Δz_top * (T - T_eq) / p.τ_relax # sign convention in Oceananigans means that heating (T < T_eq) is negative
    end

    # Optional alternative forcing: a prescribed shortwave heat flux (uniform cooling tuned so total absorbed stellar power = total cooling power), used in place of the temperature-restoring scheme above whenever `instellation` is supplied.
    @inline function surface_heat_flux(lon, lat, t, p)
        lon_rad = lon * π / 180
        lat_rad = lat * π / 180
        Q_stellar = p.instellation * (1 - p.albedo) * max(zero(lon_rad), cos(lon_rad) * cos(lat_rad))
        return -(Q_stellar - p.Q_cool) / (rho_seawater * cp_seawater) # sign convention in Oceananigans means that heating is negative
    end

    if !isnothing(instellation)
        # Uniform cooling tuned so total absorbed stellar power = total cooling power.
        # Stellar integral: ∫_{-π/2}^{π/2} cos(lon) dlon × ∫_{-80°}^{80°} cos²(lat) dlat
        # Area integral:    ∫_{-π}^{π} dlon × ∫_{-80°}^{80°} cos(lat) dlat
        lat_min_rad = -80π / 180
        lat_max_rad =  80π / 180
        stellar_integral = 2.0 * (0.5*(lat_max_rad - lat_min_rad) + 0.25*(sin(2*lat_max_rad) - sin(2*lat_min_rad)))
        ocean_area_integral = 2π * (sin(lat_max_rad) - sin(lat_min_rad))
        Q_cool = instellation * (1 - albedo) * stellar_integral / ocean_area_integral

        heat_flux_params = (instellation = instellation, albedo = albedo, Q_cool = Q_cool)
        T_top_bc = FluxBoundaryCondition(surface_heat_flux, parameters = heat_flux_params)
    else
        relax_params = (T_day = T_day, T_night = T_night, τ_relax = τ_relax, Δz_top = Δz₀)
        T_top_bc = FluxBoundaryCondition(surface_temp_relaxation, field_dependencies = :T, parameters = relax_params)
    end
    T_bcs = FieldBoundaryConditions(top = T_top_bc)

    # Diffusivity
    # Horizontal momentum uses a scale-selective biharmonic viscosity (∇⁴): it damps
    # grid-scale noise hard while leaving the resolved (equatorial) current largely
    # unbraked, so the jet speed reflects the thermal/pressure-gradient dynamics rather
    # than the lateral friction.
    #
    # On a LatitudeLongitudeGrid the zonal cell width shrinks like cos(latitude), and
    # biharmonic stability scales as Δx⁴ — so a *constant* ν would be ~1000× too strong
    # near the poles (unstable) or far too weak at the equator. Instead ν is a function
    # of latitude that tracks the local grid spacing, giving a spatially uniform grid-
    # scale damping timescale `τ_biharmonic` (default 10 days):
    #     ν₄(φ) = min(Δx(φ), Δy)⁴ / τ_biharmonic
    # Sweep the friction strength via `τ_biharmonic` (smaller τ ⇒ stronger damping).
    #
    # Horizontal tracers use a small Laplacian diffusivity instead: it preserves a
    # maximum principle (no spurious over/undershoots in SST at the day/night front),
    # and UpwindBiased(order=3) already supplies scale-selective dissipation there.
    R_planet = Float64(planet_radius)
    Δλ_rad = 2π / n_lon
    Δy_grid = deg2rad(160) / n_lat * R_planet    # meridional grid spacing (m); 160° matches latitude=(-80,80)
    @inline function ν_biharmonic(λ, φ, z, t)
        Δx = Δλ_rad * R_planet * cosd(φ)         # zonal grid spacing at latitude φ (m)
        δ = min(Δx, Δy_grid)
        return δ^4 / τ_biharmonic
    end

    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0)
    vertical_diffusivity = VerticalScalarDiffusivity(ν=1e-4, κ=1e-4)
    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν_biharmonic)
    horizontal_diffusivity = HorizontalScalarDiffusivity(κ=κ_horizontal)
    closure = (vertical_diffusivity, horizontal_viscosity, horizontal_diffusivity, convective_adjustment)

    model = HydrostaticFreeSurfaceModel(grid,
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = UpwindBiased(order=3),
                                    coriolis = coriolis,
                                    buoyancy = buoyancy,
                                    tracers = (:T, :S),
                                    closure = closure,
                                    free_surface = ImplicitFreeSurface(),
                                    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs))

    if isnothing(pickup)
        # Cold start: uniform initial temperature (default T_night) and salinity.
        T_initial = isnothing(initial_T) ? T_night : initial_T
        set!(model, T = T_initial, S = 35.0)
    end
    # (Warm start from a checkpoint is applied after the Simulation is built — see below.)

    simulation = Simulation(model, Δt=2minutes, stop_time=simulation_time)

    if !isnothing(pickup)
        # Warm start: transplant a converged state (velocities, tracers, free surface, and
        # AB2 tendencies) from a checkpoint of a neighbouring run, so this run begins
        # already spun-up and only has to adjust to the new forcing. `set!(simulation; …)`
        # is this Oceananigans version's restore API (checkpoints are simulation-rooted).
        # Restoring also brings the donor's clock and Δt, so reset them: this run integrates
        # `simulation_time` from t=0 with a fresh small Δt. Requires the checkpoint to share
        # this model's grid (same n_lat/n_lon/n_depth/ocean_depth/architecture).
        @info "Warm-starting from checkpoint: $(pickup)"
        set!(simulation; checkpoint = pickup)
        simulation.model.clock.time = zero(simulation.model.clock.time)
        simulation.model.clock.iteration = 0
        simulation.Δt = 2minutes
    end

    cfl = AdvectiveCFL(simulation.Δt)

    # Steady-state diagnostics.
    # Volume-weighted domain averages (Oceananigans `Average` uses the grid metrics,
    # so deep/large cells are weighted by their volume rather than by cell count).
    #   ⟨T⟩  — heat-content-weighted mean temperature; its drift → 0 at thermal equilibrium.
    #   ⟨KE⟩ — volume-averaged kinetic energy ½(u²+v²+w²); its drift → 0 once the circulation spins up.
    mean_T_field = Field(Average(model.tracers.T))
    ke_op = @at (Center, Center, Center) (model.velocities.u^2 + model.velocities.v^2 + model.velocities.w^2) / 2
    mean_KE_field = Field(Average(ke_op))

    # Previous-sample state, so we can report rates of change per unit *model* time
    # (robust to the adaptive timestep). NaN sentinels skip the rate on the first call.
    prev_diag = Ref((t = 0.0, T = NaN, KE = NaN))

    @inline function display_progress(sim)
        u_data = interior(sim.model.velocities.u)

        max_u = maximum(abs.(u_data))
        mean_u = mean(abs.(u_data))

        compute!(mean_T_field)
        compute!(mean_KE_field)
        # Copy the (1,1,1) reduction to the host to avoid GPU scalar indexing.
        mean_T = Array(interior(mean_T_field))[1]
        mean_KE = Array(interior(mean_KE_field))[1]

        # Rates of change since the last progress call.
        t = sim.model.clock.time
        Δt_days = (t - prev_diag[].t) / 86400
        if isfinite(prev_diag[].T) && Δt_days > 0
            dT_dt = (mean_T - prev_diag[].T) / Δt_days * 365           # °C per year
            dKE_dt = (mean_KE - prev_diag[].KE) / prev_diag[].KE / Δt_days * 100  # % per day
        else
            dT_dt = NaN
            dKE_dt = NaN
        end
        prev_diag[] = (t = t, T = mean_T, KE = mean_KE)

        current_cfl = cfl(sim.model)

        @info "Time: $(prettytime(sim.model.clock.time)), " *
            "Δt: $(prettytime(sim.Δt)), " *
            "CFL: $(round(current_cfl, digits=3)), " *
            "⟨T⟩: $(round(mean_T, digits=3)) °C " *
            "(d⟨T⟩/dt: $(round(dT_dt, digits=3)) °C/yr), " *
            "⟨KE⟩: $(round(mean_KE, sigdigits=3)) m²/s² " *
            "(d⟨KE⟩/dt: $(round(dKE_dt, digits=2)) %/day)"
    end

    simulation.callbacks[:progress] = Callback(display_progress, IterationInterval(100))

    output_filename = joinpath(directory, "ocean", "$(simulation_name).nc")

    simulation.output_writers[:full_3d] = NetCDFWriter(model,
                                                    (; T=model.tracers.T, u=model.velocities.u, v=model.velocities.v, w=model.velocities.w),
                                                    filename=output_filename,
                                                    schedule=IterationInterval(n_write),
                                                    overwrite_existing=true)

    # Optional checkpointing: serialises the full model state (prognostic fields + AB2
    # tendencies) to a JLD2 file. A later run can warm-start from it via the `pickup`
    # kwarg (a sweep spins up one point, then hands each neighbour the previous
    # checkpoint), or this same run can be resumed with `run!(sim, pickup=true)`.
    # `checkpoint_interval` is a *time* (e.g. 365days, or `simulation_time` for end-only);
    # `checkpoint_at_end=true` below guarantees the final converged state is always saved.
    if !isnothing(checkpoint_interval)
        checkpoint_dir = joinpath(directory, "ocean", "checkpoints")
        mkpath(checkpoint_dir)
        simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                    schedule = TimeInterval(checkpoint_interval),
                                                    dir = checkpoint_dir,
                                                    prefix = simulation_name,
                                                    cleanup = true,
                                                    overwrite_existing = true)
    end


    # Automatically adapt the timestep to keep CFL at a safe value.
    # The wizard only tracks *advective* CFL, but the biharmonic viscosity has its own diffusive stability limit (~τ_biharmonic/32 for the near-isotropic equatorial cells).
    # During quiescent spin-up (near-zero velocity) the advective CFL is tiny, so cap Δt below that biharmonic limit to stop Δt growing into an instability.
    max_Δt = min(1days, τ_biharmonic / 64)
    wizard = TimeStepWizard(cfl=0.1, max_change=1.05, max_Δt=max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    @info "Simulation setup complete. Starting the run..."

    run!(simulation, checkpoint_at_end = !isnothing(checkpoint_interval))

    @info "Simulation finished successfully!"

    return nothing
end