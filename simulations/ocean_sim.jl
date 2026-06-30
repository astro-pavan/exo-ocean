using Oceananigans
using Oceananigans.Units
using NCDatasets
using Oceananigans.Grids: Center
using Oceananigans.Diagnostics: AdvectiveCFL
using Statistics
using CUDA

const omega_Earth = 7.29e-5 # 1 rotation per day in rad/s
const R_Earth = 6371000 # m

const rho_seawater = 1026 # kg/m^3
const rho_air = 1.2 # kg/m^3
const C_D_wind = 0.002

const C_Bottom_Drag = 2e-3 

@inline function ocean_simulation(simulation_name, rotational_period, ocean_depth, planet_radius, heat_flux, simulation_time, n_lat=80,n_lon=360, n_depth=20, use_GPU=true, n_write=1000)

    # Grid
    grid = LatitudeLongitudeGrid(arch, Float32,
                                size = (n_lon, n_lat, n_depth),
                                longitude = (-180, 180),
                                latitude = (-80, 80),
                                z = (-ocean_depth, 0),
                                topology = (Periodic, Bounded, Bounded))

    # Coriolis Force
    if rotational_period != 0
        coriolis = HydrostaticSphericalCoriolis(rotation_rate = omega_Earth / t_rotation)
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
    if wind
        # u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc)
        # v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc)
    else
        u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
        v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)
    end

    # Diffusivity
    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0)
    background_diffusivity = (VerticalScalarDiffusivity(ν=1e-4, κ=1e-4), HorizontalScalarDiffusivity(ν=1e4, κ=1e4))
    closure = (background_diffusivity..., convective_adjustment)

    model = HydrostaticFreeSurfaceModel(grid,
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = UpwindBiased(order=3),
                                    coriolis = coriolis,
                                    buoyancy = buoyancy,
                                    tracers = (:T, :S),
                                    closure = closure,
                                    free_surface = ImplicitFreeSurface(),
                                    boundary_conditions = (; T = T_bcs, u = u_bcs, v = v_bcs))

    set!(model, T = initial_T, S = 35.0)

    simulation = Simulation(model, Δt=2minutes, stop_time=t_max)

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

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    output_filename = "simulations/runs/$(simulation_name).nc"

    simulation.output_writers[:full_3d] = NetCDFWriter(model, 
                                                    (; T=model.tracers.T, u=model.velocities.u, v=model.velocities.v, w=model.velocities.w), 
                                                    filename=output_filename,
                                                    schedule=TimeInterval(t_write),
                                                    overwrite_existing=true)


    # Automatically adapt the timestep to keep CFL at a safe 0.2
    wizard = TimeStepWizard(cfl=0.2, max_change=1.05, max_Δt=1days)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    @info "Simulation setup complete. Starting the run..."

    run!(simulation)

    @info "Simulation finished successfully!"

    return nothing
end