# Wind stress from a CLERO emulator wind file (src/climate/climate_emulator.py solve); include after ocean_sim.jl (needs constants.jl).
using NCDatasets
using Oceananigans
using Oceananigans.Architectures: architecture

# Bilinear interpolation of f[lon, lat] at (λ, φ) in degrees: periodic in longitude (uniform spacing), clamped in latitude.
function bilinear_periodic(f, lon, lat, λ, φ)
    nλ = length(lon)
    x = mod(λ - lon[1], 360) / (lon[2] - lon[1])
    i0 = floor(Int, x)
    s = x - i0
    i1, i0 = mod(i0 + 1, nλ) + 1, mod(i0, nλ) + 1
    φc = clamp(φ, lat[1], lat[end])
    j1 = clamp(searchsortedfirst(lat, φc), 2, length(lat))
    j0 = j1 - 1
    t = (φc - lat[j0]) / (lat[j1] - lat[j0])
    return (1 - s) * (1 - t) * f[i0, j0] + s * (1 - t) * f[i1, j0] + (1 - s) * t * f[i0, j1] + s * t * f[i1, j1]
end

# Neutral 10 m drag coefficient of Large & Yeager (2004, NCAR TN-460; 2009, Clim. Dyn. 33, 341), with CORE's 0.5 m/s wind-speed floor.
large_yeager_CD(U) = 2.7e-3 / max(U, 0.5) + 1.42e-4 + 7.64e-5 * max(U, 0.5)

# Kinematic surface stress (τx, τy) / ρ₀ at u- and v-points for `ocean_simulation(...; wind_stress)`: τ = ρ_air C_D(|U|) |U| U from the
# planet's lowest-level wind and near-surface air density, signed as wind_stress_u/v in ocean_sim.jl. Pass `P_rot` (days) to check the file.
function emulator_wind_stress(path, grid; P_rot = nothing)
    lon, lat, u, v, ρa, attrs = NCDataset(path) do ds
        haskey(ds, "rho_air") && ndims(ds["u_0"]) == 2 || error("$(basename(path)) is not a single-planet wind file; regenerate it with src/climate/climate_emulator.py")
        Float64.(ds["lon"][:]), Float64.(ds["lat"][:]), Float64.(ds["u_0"][:, :]), Float64.(ds["v_0"][:, :]), Float64.(ds["rho_air"][:, :]),
        Dict(ds.attrib)
    end                                                                          # fields are (lon, lat)
    isnothing(P_rot) || isapprox(P_rot, attrs["P_rot_days"]; rtol=1e-3) || @warn "$(basename(path)) is for P_rot = $(attrs["P_rot_days"]) d, not $(P_rot) d"

    function stress(λ, φ, component)
        U = bilinear_periodic(u, lon, lat, λ, φ)
        V = bilinear_periodic(v, lon, lat, λ, φ)
        W = sqrt(U^2 + V^2)
        return -bilinear_periodic(ρa, lon, lat, λ, φ) * large_yeager_CD(W) * W * (component == :x ? U : V) / rho_seawater
    end

    cpu_grid = on_architecture(CPU(), grid)
    τx = [stress(λ, φ, :x) for λ in λnodes(cpu_grid, Face()), φ in φnodes(cpu_grid, Center())]    # (Nx, Ny) at u-points
    τy = [stress(λ, φ, :y) for λ in λnodes(cpu_grid, Center()), φ in φnodes(cpu_grid, Face())]    # (Nx, Ny+1) at v-points
    @info "Emulator wind stress from $(basename(path)) (P_rot = $(attrs["P_rot_days"]) d, P0 = $(attrs["P0_bar"]) bar, " *
          "T_star = $(round(Int, attrs["T_star_K"])) K, F_star = $(round(Int, attrs["F_star_W_m2"])) W/m², CO2 = $(round(attrs["CO2"], sigdigits=3))): " *
          "max |τ| = $(round(rho_seawater * maximum(hypot.(τx, τy[:, 1:end-1])), sigdigits=3)) N/m²"

    FT = eltype(grid)
    arch = architecture(grid)
    return on_architecture(arch, FT.(τx)), on_architecture(arch, FT.(τy))
end
