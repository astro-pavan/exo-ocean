using SpeedyWeather
using Dates

include("constants.jl")
include("floored_vertical_diffusion.jl")

const directory = "exo_ocean_sims/"

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

# Newtonian cooling toward a tidally-locked day/night equilibrium temperature.
# Hottest at the substellar point (lon=0°, lat=0°), coldest at the antistellar point.
struct NewtonianCooling{NF<:AbstractFloat} <: SpeedyWeather.AbstractForcing
    T_day::NF    # equilibrium temperature at substellar point [K]
    T_night::NF  # equilibrium temperature at antistellar point [K]
    τ_rad::NF    # radiative relaxation timescale [s]
end

SpeedyWeather.initialize!(::NewtonianCooling, ::SpeedyWeather.AbstractModel) = nothing

function SpeedyWeather.forcing!(vars::SpeedyWeather.Variables,
                                 F::NewtonianCooling,
                                 lf::Integer,
                                 model::SpeedyWeather.AbstractModel)
    londs  = model.geometry.londs
    latds  = model.geometry.latds
    σ_full = model.geometry.σ_levels_full
    κ      = Float32(model.atmosphere.κ)   # R_dry / c_p ≈ 0.286
    radius = Float32(model.planet.radius)

    temp_grid = vars.grid.temperature
    dTdt      = vars.tendencies.grid.temperature

    for k in 1:size(temp_grid, 2)
        σ = σ_full[k]
        for ij in 1:size(temp_grid, 1)
            lon_rad = londs[ij] * π / 180f0
            lat_rad = latds[ij] * π / 180f0
            # Surface equilibrium: full day/night contrast at σ = 1
            T_eq_surf = F.T_night + (F.T_day - F.T_night) * max(0f0, cos(lon_rad) * cos(lat_rad))
            # Scale with height: dry-adiabatic profile T_eq ∝ σ^κ, floor at 200 K
            # This puts the stellar heating at the surface and keeps the stratosphere cold,
            # so the upper atmosphere does not fight the surface contrast.
            T_eq = max(200f0, T_eq_surf * σ^κ)
            # SpeedyWeather's grid-space forcing tendency must be scaled by the planet
            # radius to match the internal non-dimensionalisation of the temperature
            # equation (confirmed against the built-in HeldSuarez forcing, which does
            # the same: `temp_relax_freq .*= radius`). Without this the relaxation is
            # effectively radius-times too weak and the day/night contrast never forms.
            dTdt[ij, k] -= radius * (temp_grid[ij, k] - T_eq) / F.τ_rad
        end
    end
end

function atmosphere_simulation(simulation_name, rotational_period, surface_pressure, planet_radius, T_day, T_night, simulation_time;
    n_lat=90, n_levels=8, use_GPU=false, output_dt=6,
    gravity=g_Earth, τ_rad=259200.0,  # 3 days; 10-day default was too slow vs ~4-day dynamical timescale
    # SpeedyWeather's built-in default is Minute(40), which scales to 160min at T7.
    # Reduce this if you see NaN warnings — typically needed for no-rotation cases
    # or large day/night temperature contrasts (>80 K) where winds can be very fast.
    Δt_at_T31=Minute(20),
    # Bulk Richardson number above which BulkRichardsonDrag/vertical diffusion cut off
    # mixing (SpeedyWeather default: 10). See floored_vertical_diffusion.jl for why this
    # alone doesn't fix weak-wind boundary layer decoupling — min_wind_speed does.
    critical_Richardson=10.0,
    # Minimum wind speed [m/s] floored into the bulk Richardson number and mixing
    # coefficient in FlooredBulkRichardsonDiffusion (floored_vertical_diffusion.jl),
    # to prevent the surface layer decoupling from the free troposphere under calm
    # conditions. Comparable to CESM's 0.5 m/s floor.
    min_wind_speed=1.0)

    @info "Setting up atmosphere simulation..."

    # SpeedyWeather GPU support via architecture keyword
    architecture = if use_GPU && _cuda_available
        @info "CUDA GPU detected, using CUDA backend."
        GPU()
    elseif use_GPU && _amdgpu_available
        @info "AMD GPU detected, using AMDGPU backend."
        GPU()
    else
        use_GPU && @warn "No functional GPU found, falling back to CPU."
        CPU()
    end

    # Spectral truncation T ≈ n_lat / 3 (e.g. n_lat=90 → T30)
    trunc = round(Int, n_lat / 3)

    spectral_grid = SpectralGrid(trunc=trunc, nlayers=n_levels, NF=Float32,
                                 architecture=architecture)

    # Planet: rotation rate uses the same convention as ocean_sim.jl
    # (rotational_period=1 → Earth-like, 0 → no rotation)
    rotation_rate = rotational_period == 0 ? 0f0 : Float32(omega_Earth / rotational_period)

    planet = Earth(spectral_grid,
                   rotation=rotation_rate,
                   radius=Float32(planet_radius),
                   gravity=Float32(gravity))

    # Reference surface pressure sets the column mass
    atmosphere = EarthDryAtmosphere(spectral_grid, reference_pressure=Float32(surface_pressure))

    # Newtonian cooling toward day/night equilibrium
    forcing = NewtonianCooling{Float32}(Float32(T_day), Float32(T_night), Float32(τ_rad))

    # Boundary layer drag and vertical diffusion share the same critical_Richardson cutoff;
    # override both consistently rather than leaving one at the SpeedyWeather default.
    # Note: BulkRichardsonDrag(spectral_grid; kwargs...) can't forward keyword overrides in
    # this SpeedyWeather version (its SpectralGrid constructor is missing a `;` before
    # `kwargs...`, so keywords land as unsupported positional args) — construct via the
    # type's own keyword constructor instead.
    boundary_layer = BoundaryLayer(spectral_grid;
                                   drag=BulkRichardsonDrag{Float32}(critical_Richardson=Float32(critical_Richardson)))
    vertical_diffusion = FlooredBulkRichardsonDiffusion(spectral_grid;
                                                        critical_Richardson=Float32(critical_Richardson),
                                                        min_wind_speed=Float32(min_wind_speed))

    # Time-based output (SpeedyWeather uses wall-clock periods, not iteration counts)
    output = NetCDFOutput(spectral_grid, PrimitiveDryModel,
                          path=joinpath(directory, "atm"),
                          id=simulation_name,
                          interval=Hour(output_dt))
    add!(output, FlooredBoundaryLayerHeightOutput())

    time_stepping = Leapfrog(spectral_grid, Δt_at_T31=Δt_at_T31)
    actual_Δt = time_stepping.Δt_millisec.value / 60000
    @info "Time step: $(round(actual_Δt, digits=1)) min at T$(trunc)"

    # Disable SpeedyWeather's built-in radiation: JeevanjeeRadiation cools toward 200 K
    # and TransparentShortwave adds solar input — both conflict with our Newtonian cooling,
    # which already encodes the net day/night radiative balance.
    model = PrimitiveDryModel(spectral_grid;
                              planet=planet,
                              atmosphere=atmosphere,
                              forcing=forcing,
                              output=output,
                              time_stepping=time_stepping,
                              initial_conditions=StartFromRest(spectral_grid),
                              orography=NoOrography(spectral_grid),
                              land_sea_mask=AquaPlanetMask(spectral_grid),
                              boundary_layer=boundary_layer,
                              vertical_diffusion=vertical_diffusion,
                              longwave_radiation=nothing,
                              shortwave_radiation=nothing,
                              # verbose=true forces the progress bar on even when running
                              # as a non-interactive script (Feedback's default is isinteractive())
                              feedback=Feedback(verbose=true))

    simulation = initialize!(model)

    # simulation_time is in seconds (consistent with ocean_sim.jl convention)
    period = Second(round(Int, simulation_time))

    @info "Atmosphere simulation setup complete. Starting the run..."
    run!(simulation, period=period, output=true)
    @info "Atmosphere simulation finished successfully!"

    return nothing
end
