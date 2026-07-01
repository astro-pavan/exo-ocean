include("constants.jl")
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Oceananigans.Diagnostics: AdvectiveCFL
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

const directory = "/home/pt426/data/exo_ocean_sims"

@inline function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, instellation, simulation_time; n_lat=160, n_lon=360, n_depth=20, use_GPU=true, n_write=1000, wind_stress=nothing, initial_T=1.0, albedo=0.06)

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

    # u_top_bc = FluxBoundaryCondition(wind_stress_u, discrete_form=true)
    # v_top_bc = FluxBoundaryCondition(wind_stress_v, discrete_form=true)

    u_bottom_bc = FluxBoundaryCondition(bottom_drag_u, discrete_form=true)
    v_bottom_bc = FluxBoundaryCondition(bottom_drag_v, discrete_form=true)

    # Boundary Conditions
    if !isnothing(wind_stress)
        error("Wind stress boundary conditions are not yet implemented.")
        # u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc)
        # v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc)
    end
    u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
    v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)

    # Heat Flux
    # Uniform cooling tuned so total absorbed stellar power = total cooling power.
    # Stellar integral: ∫_{-π/2}^{π/2} cos(lon) dlon × ∫_{-80°}^{80°} cos²(lat) dlat
    # Area integral:    ∫_{-π}^{π} dlon × ∫_{-80°}^{80°} cos(lat) dlat
    lat_min_rad = -80π / 180
    lat_max_rad =  80π / 180
    stellar_integral = 2.0 * (0.5*(lat_max_rad - lat_min_rad) + 0.25*(sin(2*lat_max_rad) - sin(2*lat_min_rad)))
    ocean_area_integral = 2π * (sin(lat_max_rad) - sin(lat_min_rad))
    Q_cool = instellation * (1 - albedo) * stellar_integral / ocean_area_integral

    @inline function surface_heat_flux(lon, lat, t, p)
        lon_rad = lon * π / 180
        lat_rad = lat * π / 180
        Q_stellar = p.instellation * (1 - p.albedo) * max(zero(lon_rad), cos(lon_rad) * cos(lat_rad))
        return -(Q_stellar - p.Q_cool) / (rho_seawater * cp_seawater) # sign convention in Oceanigans means that heating is negative
    end

    heat_flux_params = (instellation = instellation, albedo = albedo, Q_cool = Q_cool)
    T_top_bc = FluxBoundaryCondition(surface_heat_flux, parameters = heat_flux_params)
    T_bcs = FieldBoundaryConditions(top = T_top_bc)

    # Diffusivity
    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0)
    background_diffusivity = (VerticalScalarDiffusivity(ν=1e-4, κ=1e-4), HorizontalScalarDiffusivity(ν=1e5, κ=1e4))
    closure = (background_diffusivity..., convective_adjustment)

    model = HydrostaticFreeSurfaceModel(grid,
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = UpwindBiased(order=3),
                                    coriolis = coriolis,
                                    buoyancy = buoyancy,
                                    tracers = (:T, :S),
                                    closure = closure,
                                    free_surface = ImplicitFreeSurface(),
                                    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs))

    set!(model, T = initial_T, S = 35.0)

    simulation = Simulation(model, Δt=2minutes, stop_time=simulation_time)

    cfl = AdvectiveCFL(simulation.Δt)

    @inline function display_progress(sim)
        u_data = interior(sim.model.velocities.u)
        T_data = interior(sim.model.tracers.T)
        
        max_u = maximum(abs.(u_data))
        mean_u = mean(abs.(u_data)) 
        
        mean_T = mean(identity.(T_data)) 
        
        current_cfl = cfl(sim.model)
        
        @info "Time: $(prettytime(sim.model.clock.time)), " *
            "Δt: $(prettytime(sim.Δt)), " *
            "CFL: $(round(current_cfl, digits=3)), " *
            "Max u: $(round(max_u, digits=3)) m/s, " *
            "Mean u: $(round(mean_u, digits=4)) m/s, " *
            "Mean T: $(round(mean_T, digits=2)) °C"
    end

    simulation.callbacks[:progress] = Callback(display_progress, IterationInterval(100))

    output_filename = joinpath(directory, "ocean", "$(simulation_name).nc")

    simulation.output_writers[:full_3d] = NetCDFWriter(model, 
                                                    (; T=model.tracers.T, u=model.velocities.u, v=model.velocities.v, w=model.velocities.w), 
                                                    filename=output_filename,
                                                    schedule=IterationInterval(n_write),
                                                    overwrite_existing=true)


    # Automatically adapt the timestep to keep CFL at a safe 0.2
    wizard = TimeStepWizard(cfl=0.1, max_change=1.05, max_Δt=1days)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    @info "Simulation setup complete. Starting the run..."

    run!(simulation)

    @info "Simulation finished successfully!"

    return nothing
end