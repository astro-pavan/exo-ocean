# Closed-form wind at CLERO's lowest level (σ = 0.925), substellar point at lon = 0: function B of docs/wind_parameterisation.md, section 12.

const PATTERN_WIND_RANGE = (ΔT = (5.0, 80.0), P0 = (0.5, 10.0), P_rot = (4.0, 120.0))  # K, bar, days: the fitted sweep

const GAS_CONSTANT = 8.314462618  # J/(mol K)
const N2_MOLAR_MASS = 28.0134e-3  # kg/mol, the background gas in CLERO (src/climate/climate_emulator.py)

"""
    ideal_gas_air_density(P0, T_day, T_night; molar_mass = N2_MOLAR_MASS)

Planet-wide near-surface air density (kg/m³) from the ideal gas law, with surface pressure `P0` (bar) and the mean of `T_day` and `T_night` (°C).
Pass `molar_mass` (kg/mol) when the composition is known: pure N₂ underestimates the density of CO₂-rich atmospheres.
"""
ideal_gas_air_density(P0, T_day, T_night; molar_mass = N2_MOLAR_MASS) = P0 * 1e5 * molar_mass / (GAS_CONSTANT * ((T_day + T_night) / 2 + 273.15))

struct PatternWind{FT}
    U_c :: FT  # core inflow amplitude, m/s
    a_c :: FT  # core width, rad
    U_a :: FT  # cos 2α inflow amplitude, m/s
    G   :: FT  # sin α tangential amplitude, m/s
    J_l :: FT  # lower-latitude jet amplitude, m/s
    J_h :: FT  # high-latitude jet amplitude, m/s
end

"""
    pattern_wind(ΔT, P0, P_rot; FT = Float64)

Wind function for a day-night contrast `ΔT` (K), surface pressure `P0` (bar) and rotation period `P_rot` (days). The result is callable as
`wind(lon, lat, t) -> (u, v)` in m/s, with `lon` and `lat` in degrees; `t` is ignored. Warns outside the range it was fitted on.
"""
function pattern_wind(ΔT, P0, P_rot; FT = Float64)
    for (name, value) in Base.pairs((; ΔT, P0, P_rot))
        low, high = PATTERN_WIND_RANGE[name]
        low <= value <= high || @warn "pattern_wind: $name = $value is outside the fitted range ($low, $high)"
    end
    L, ℓ, r = log(ΔT / 40), log(P0 / 2), log(P_rot / 15)
    return PatternWind{FT}(exp(3.510292 + 0.226779L),
                           exp(-0.670852 - 0.394476ℓ + 0.174005r),
                           3.099589 - 1.701344ℓ,
                           -11.156391 + 5.94797ℓ,
                           -1.759867 + 1.924174L - 1.635126ℓ + 2.833388r,
                           3.396385 + 2.885027ℓ - 4.17552r)
end

@inline pattern_jet(lat, centre, width) = exp(-(abs(lat) - centre)^2 / (2 * width^2))

@inline function (w::PatternWind{FT})(lon, lat, t = zero(FT)) where FT
    sinλ, cosλ = sincos(deg2rad(FT(lon)))
    sinφ, cosφ = sincos(deg2rad(FT(lat)))
    θ = acos(clamp(cosφ * cosλ, -one(FT), one(FT)))
    α = atan(sinφ, cosφ * sinλ)

    # sin θ in the radial profile cancels the 1 / sin θ of the unit vectors, so the wind stays finite at the substellar point
    profile(a) = 1 / (1 + (θ / a)^2)
    inflow = w.U_c * profile(w.a_c) + w.U_a * profile(FT(2.017085)) * cos(2α)
    tangential = w.G * profile(FT(0.474006)) * sin(α)

    jets = w.J_l * pattern_jet(FT(lat), FT(69.2616), FT(42.0959)) + w.J_h * pattern_jet(FT(lat), FT(75.1794), FT(26.7884))
    u = -inflow * sinλ + tangential * sinφ * cosλ + jets
    v = -inflow * sinφ * cosλ - tangential * sinλ
    return u, v
end

# Equivalent neutral 10 m wind vector from the function's wind at the height z of CLERO's lowest level, as emulator_10m_wind
function pattern_10m_wind(wind, z, λ, φ)
    U, V = wind(λ, φ)
    W = sqrt(U^2 + V^2)
    W > 0 || return zero(W), zero(W)
    U₁₀ = neutral_10m_speed(W, z)
    return U₁₀ * U / W, U₁₀ * V / W
end

# (U₁₀x, U₁₀y, ρ_air) arrays at u-points and at v-points on the CPU, in the form of emulator_10m_wind_arrays
function pattern_10m_wind_arrays(wind, ρ, z, grid)
    cpu_grid = on_architecture(CPU(), grid)
    function at(λs, φs)
        w = [pattern_10m_wind(wind, z, λ, φ) for λ in λs, φ in φs]
        return getindex.(w, 1), getindex.(w, 2), fill(ρ, size(w))
    end
    return at(λnodes(cpu_grid, Face()), φnodes(cpu_grid, Center())), at(λnodes(cpu_grid, Center()), φnodes(cpu_grid, Face()))
end

"""
    pattern_wind_stress(T_day, T_night, P0, P_rot, grid; scale = 1, coupling = 1, molar_mass = N2_MOLAR_MASS)

Surface forcing from the function wind for `ocean_simulation(...; function_wind = true)`, with `T_day`, `T_night` in °C, `P0` in bar and `P_rot` in days.
`molar_mass` (kg/mol) sets the air density. Returns kinematic stress arrays as `emulator_wind_stress`, or for `coupling != 1` the current-feedback parameters as `emulator_current_feedback`.
"""
function pattern_wind_stress(T_day, T_night, P0, P_rot, grid; scale = 1, coupling = 1, molar_mass = N2_MOLAR_MASS)
    wind = pattern_wind(T_day - T_night, P0, P_rot)
    ρ = ideal_gas_air_density(P0, T_day, T_night; molar_mass)
    z = lowest_level_height(P0, ρ, g_Earth)
    (Ux_u, Uy_u, ρ_u), (Ux_v, Uy_v, ρ_v) = pattern_10m_wind_arrays(wind, ρ, z, grid)
    τx, τy = bulk_stress.(Ux_u, Uy_u, ρ_u, Ux_u, scale), bulk_stress.(Ux_v, Uy_v, ρ_v, Uy_v, scale)
    @info "Function wind stress (P_rot = $(P_rot) d, P0 = $(P0) bar, ΔT = $(T_day - T_night) K, ρ_air = $(round(ρ, sigdigits=3)) kg/m³): " *
          "max |τ| = $(round(rho_seawater * maximum(hypot.(τx, τy[:, 1:end-1])), sigdigits=3)) N/m² at rest, from winds at $(round(Int, z)) m reduced to 10 m"
    FT = eltype(grid)
    on(a) = on_architecture(architecture(grid), FT.(a))
    coupling == 1 && return on(τx), on(τy)
    @info "Current feedback: stress from the wind relative to (1 − s_w) × the surface current, s_w = $(coupling)"
    return (; Ux_u = on(Ux_u), Uy_u = on(Uy_u), ρ_u = on(ρ_u), Ux_v = on(Ux_v), Uy_v = on(Uy_v), ρ_v = on(ρ_v), α = FT(1 - coupling), scale = FT(scale))
end

# Function-wind provenance for the ocean output's global attributes
pattern_wind_attributes(P0, T_day, T_night, scale, coupling; molar_mass = N2_MOLAR_MASS) =
    Dict("wind_source" => "function B, docs/wind_parameterisation.md section 12", "surface_pressure_bar" => Float64(P0),
         "air_molar_mass_kg_mol" => Float64(molar_mass), "air_density_kg_m3" => Float64(ideal_gas_air_density(P0, T_day, T_night; molar_mass)),
         "wind_stress_scale" => Float64(scale), "wind_current_coupling" => Float64(coupling))
