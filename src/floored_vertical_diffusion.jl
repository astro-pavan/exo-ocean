using Base: @propagate_inbounds

# ---------------------------------------------------------------------------
# BulkRichardsonDiffusion (Frierson, 2006) diagnoses both the depth of vertical
# mixing (via a bulk Richardson number Ri ~ 1/V^2) and the *magnitude* of the
# mixing coefficient (via K0 ~ surface_speed) directly from the raw near-surface
# wind. Under sustained weak near-surface wind, both collapse toward zero
# together: Ri blows up (cutting mixing off after ~1 layer) *and* K0 vanishes
# (making what little mixing remains negligible). The result is a surface layer
# that never communicates with the free troposphere above it, regardless of
# vertical resolution or integration time — raising critical_Richardson alone
# does not fix this, since it only affects the depth term, not K0 (confirmed:
# runs 0006 vs 0011, identical apart from critical_Richardson=10 vs 1000, no
# difference in outcome).
#
# This is the "stable boundary layer problem" documented in the NWP literature
# for bulk-Richardson-number schemes under weak wind (e.g. Holtslag et al. 2013,
# BAMS; Sandu et al. 2013, JAMES). The standard fix is a minimum wind speed
# floor in the Richardson number and flux calculations (e.g. CESM uses 0.5 m/s).
# This reimplements BulkRichardsonDiffusion with that floor applied everywhere
# V enters — a new component defined here, not a patch to SpeedyWeather itself.
# ---------------------------------------------------------------------------
Base.@kwdef struct FlooredBulkRichardsonDiffusion{NF, VectorType} <: SpeedyWeather.AbstractVerticalDiffusion
    von_Karman::NF = 0.4f0
    roughness_length::NF = 3.21f-5
    critical_Richardson::NF = 10f0
    surface_layer_fraction::NF = 0.1f0
    min_wind_speed::NF = 1f0            # [m/s] floor applied to V in both Ri and K0
    diffuse_static_energy::Bool = true
    diffuse_momentum::Bool = true
    ∇²_above::VectorType
    ∇²_below::VectorType
    σ_full::VectorType     # full-level σ, needed for the σ<->z Jacobian (see below)
end

SpeedyWeather.variables(::FlooredBulkRichardsonDiffusion) = (
    SpeedyWeather.ParameterizationVariable(:boundary_layer_height, SpeedyWeather.Grid2D(),
                                           desc = "Boundary layer height index (floored scheme)", units = "1"),
)

function FlooredBulkRichardsonDiffusion(SG::SpeedyWeather.SpectralGrid; kwargs...)
    ∇²_above = SpeedyWeather.on_architecture(SG.architecture, zeros(SG.NF, SG.nlayers))
    ∇²_below = SpeedyWeather.on_architecture(SG.architecture, zeros(SG.NF, SG.nlayers))
    σ_full   = SpeedyWeather.on_architecture(SG.architecture, zeros(SG.NF, SG.nlayers))
    return FlooredBulkRichardsonDiffusion{SG.NF, SG.VectorType}(; ∇²_above, ∇²_below, σ_full, kwargs...)
end

function SpeedyWeather.initialize!(diffusion::FlooredBulkRichardsonDiffusion, model::SpeedyWeather.AbstractModel)
    (; nlayers) = model.geometry
    nlayers == 1 && return nothing

    σ = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), model.geometry.σ_levels_full)
    σ_half = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), model.geometry.σ_levels_half)
    ∇²_above = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), diffusion.∇²_above)
    ∇²_below = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), diffusion.∇²_below)

    for k in 1:nlayers
        σ₋ = k <= 1 ? -Inf : σ[k - 1]
        σ₊ = k >= nlayers ? Inf : σ[k + 1]
        ∇²_above[k] = inv(2 * (σ[k] - σ₋) * (σ_half[k + 1] - σ_half[k]))
        ∇²_below[k] = inv(2 * (σ₊ - σ[k]) * (σ_half[k + 1] - σ_half[k]))
    end

    arch = model.architecture
    diffusion.∇²_above .= SpeedyWeather.on_architecture(arch, ∇²_above)
    diffusion.∇²_below .= SpeedyWeather.on_architecture(arch, ∇²_below)
    diffusion.σ_full   .= model.geometry.σ_levels_full
    return nothing
end

@propagate_inbounds SpeedyWeather.parameterization!(ij, vars, diffusion::FlooredBulkRichardsonDiffusion, model) =
    floored_vertical_diffusion!(ij, vars, diffusion, model.atmosphere, model.planet, model.orography, model.geopotential)

@propagate_inbounds function floored_vertical_diffusion!(ij, vars, diffusion::FlooredBulkRichardsonDiffusion, atmosphere, planet, orography, geopot)
    (; diffuse_momentum, diffuse_static_energy) = diffusion
    (diffuse_momentum || diffuse_static_energy) || return nothing

    K, kₕ = floored_diffusion_coefficients!(ij, vars, diffusion, atmosphere, planet, orography, geopot)

    u_tend = vars.tendencies.grid.u
    v_tend = vars.tendencies.grid.v
    temp_tend = vars.tendencies.grid.temperature
    u = vars.grid.u
    v = vars.grid.v

    diffuse_momentum && floored_diffuse!(ij, u_tend, u, K, kₕ, diffusion)
    diffuse_momentum && floored_diffuse!(ij, v_tend, v, K, kₕ, diffusion)

    if diffuse_static_energy
        # dry-model only: dry static energy, no virtual temperature/humidity correction
        dry_static_energy = vars.scratch.grid.a
        cₚ = atmosphere.heat_capacity
        T = vars.grid.temperature
        Φ = vars.grid.geopotential
        for k in 1:size(T, 2)
            dry_static_energy[ij, k] = cₚ * T[ij, k] + Φ[ij, k]
            K[ij, k] /= cₚ
        end
        floored_diffuse!(ij, temp_tend, dry_static_energy, K, kₕ, diffusion)
    end
    return nothing
end

@propagate_inbounds function floored_diffusion_coefficients!(ij, vars, diffusion::FlooredBulkRichardsonDiffusion, atmosphere, planet, orog, geopot)
    nlayers = length(diffusion.∇²_above)

    Ri_c = diffusion.critical_Richardson
    fb = diffusion.surface_layer_fraction
    κ = diffusion.von_Karman
    z₀ = diffusion.roughness_length
    V_min = diffusion.min_wind_speed
    gravity⁻¹ = inv(planet.gravity)

    T₀ = atmosphere.reference_temperature
    gravity = planet.gravity
    Δp_geopot_full = geopot.Δp_geopot_full
    Z = T₀ * Δp_geopot_full[nlayers] / gravity
    logZ_z₀ = log(Z / z₀)

    u = vars.grid.u
    v = vars.grid.v
    geopotential = vars.grid.geopotential
    (; orography) = orog
    R_dry = atmosphere.R_dry
    σ_full = diffusion.σ_full
    T = vars.grid.temperature

    Ri = floored_bulk_richardson!(ij, vars, atmosphere, V_min)
    kₕ::Int = nlayers
    while kₕ > 0 && Ri[ij, kₕ] < Ri_c
        kₕ -= 1
    end
    kₕ += 1

    vars.parameterizations.boundary_layer_height[ij] = kₕ

    K = vars.scratch.grid.b
    for k in 1:nlayers
        K[ij, k] = 0
    end

    if kₕ <= nlayers
        h = max(geopotential[ij, kₕ] * gravity⁻¹ - orography[ij], 0)
        Ri_N = clamp(Ri[ij, nlayers], 0, Ri_c)
        sqrtC = (κ / logZ_z₀) * (1 - Ri_N / Ri_c)
        surface_speed = max(sqrt(u[ij, nlayers]^2 + v[ij, nlayers]^2), V_min)
        K0 = κ * surface_speed * sqrtC

        for k in kₕ:nlayers
            z = max(geopotential[ij, k] * gravity⁻¹ - orography[ij], z₀)
            zmin = min(z, fb * h)
            K_k = K0 * zmin
            K_k *= z < fb * h ? one(K0) : floored_zfac(z, h, fb)
            K_k *= Ri[ij, kₕ] <= 0 ? one(K0) : floored_Rifac(Ri[ij, kₕ], Ri_c, logZ_z₀)

            # K_k here is a physical (z-space) eddy diffusivity [m²/s], but the ∇²_above/∇²_below
            # operators are pure σ-space (1/Δσ²). The stencil discretises ∂σ(K̃ ∂σ φ), which needs
            # K̃ in σ-space units [1/s], not m²/s. Convert with the Jacobian (dσ/dz)²:
            #   dσ/dz = ρg/pₛ = σ g/(R_dry T)   (hydrostatic + ideal gas)
            # so K̃ = K_z (dσ/dz)². The upstream SpeedyWeather scheme omits this factor (~1e-8
            # near the surface), which — once the k±1 indexing bug is also fixed — makes the raw
            # tendency ~1e8× too large and blows up on step 1. Fold the Jacobian into K here.
            dσdz = σ_full[k] * gravity / (R_dry * T[ij, k])
            K[ij, k] = K_k * dσdz^2
        end
    end

    return K, kₕ
end

@inline floored_zfac(z, h, fb) = z / (fb * h) * (1 - (z - fb * h) / ((1 - fb) * h))^2

@inline function floored_Rifac(Ri, Ri_c, logz_z₀)
    Ri_Ri_c = Ri / Ri_c
    return inv(1 + Ri_Ri_c * logz_z₀ / (1 - Ri_Ri_c))
end

@propagate_inbounds function floored_diffuse!(ij, tend, var, K, kₕ, diffusion::FlooredBulkRichardsonDiffusion)
    (; ∇²_above, ∇²_below) = diffusion
    nlayers = size(tend, 2)
    for k in kₕ:nlayers
        # Upstream SpeedyWeather.BulkRichardsonDiffusion has this as max(k, 1) / min(k, nlayers)
        # (no +-1 offset), which always evaluates to k itself -- making the whole diffusion
        # tendency exactly zero, always. Fixed here: correct neighbour indices, clamped at
        # the top/surface boundaries to give zero-gradient (no-flux) boundary conditions.
        k₋ = max(k - 1, 1)
        k₊ = min(k + 1, nlayers)
        K_∂var_below = (var[ij, k₊] - var[ij, k]) * (K[ij, k₊] + K[ij, k])
        K_∂var_above = (var[ij, k] - var[ij, k₋]) * (K[ij, k] + K[ij, k₋])
        tend[ij, k] += ∇²_below[k] * K_∂var_below - ∇²_above[k] * K_∂var_above
    end
    return nothing
end

# Following Frierson, 2007, but with a min_wind_speed floor on V^2 in both terms
# to avoid Ri blowing up (and the diagnosed mixing depth collapsing to a single
# layer) under calm near-surface conditions. Dry-model only: uses T directly
# rather than virtual temperature (no humidity in PrimitiveDryModel).
@propagate_inbounds function floored_bulk_richardson!(ij, vars, atmosphere, V_min)
    Ri = vars.scratch.grid.a
    nlayers = size(Ri, 2)
    surface = nlayers
    cₚ = atmosphere.heat_capacity

    u = vars.grid.u_prev
    v = vars.grid.v_prev
    Φ = vars.grid.geopotential
    T = vars.grid.temperature_prev

    V_min² = V_min^2

    V² = max(u[ij, surface]^2 + v[ij, surface]^2, V_min²)
    Θ₀ = cₚ * T[ij, surface]
    Θ₁ = Θ₀ + Φ[ij, surface]
    Ri[ij, surface] = Φ[ij, surface] * (Θ₁ - Θ₀) / (Θ₀ * V²)

    for k in 1:(nlayers - 1)
        V² = max(u[ij, k]^2 + v[ij, k]^2, V_min²)
        virtual_dry_static_energy = cₚ * T[ij, k] + Φ[ij, k]
        Ri[ij, k] = Φ[ij, k] * (virtual_dry_static_energy - Θ₁) / (Θ₁ * V²)
    end

    return Ri
end

# NetCDF output for boundary_layer_height (kh), the diagnosed uppermost layer within
# the mixed boundary layer. A simple field passthrough, mirroring BoundaryLayerDragOutput.
Base.@kwdef mutable struct FlooredBoundaryLayerHeightOutput <: SpeedyWeather.AbstractOutputVariable
    name::String = "bl_kh"
    unit::String = "1"
    long_name::String = "Boundary layer height index (floored scheme)"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 7
end

SpeedyWeather.path(::FlooredBoundaryLayerHeightOutput, simulation) =
    simulation.variables.parameterizations.boundary_layer_height
