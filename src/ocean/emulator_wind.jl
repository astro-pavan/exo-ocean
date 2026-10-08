# CLERO emulator winds for ocean_simulation(...; emulator_wind = true): solve the planet with src/climate/climate_emulator.py (run as a
# subprocess in the Python env that has `clero`), cache the wind file, and turn it into bulk surface stress. Included by ocean_sim.jl.
using Oceananigans.Architectures: architecture

const EMULATOR_PYTHON = get(ENV, "EXO_EMULATOR_PYTHON", "/data/pt426/ocean-venv/bin/python")
const EMULATOR_SCRIPT = joinpath(REPO_ROOT, "src", "climate", "climate_emulator.py")

_tag(x) = replace(string(round(Float64(x), digits = 3)), r"\.0$" => "")

"""
    emulator_wind_file(P_rot, T_day, T_night, P0; gcm = "exocam", feh = nothing, tol = 0.1, device = nothing, overwrite = false)

Path of the CLERO wind file for this (P_rot [d], T_day, T_night [°C], P0 [bar]) under `\$EXO_OCEAN_OUTPUT/winds/`. Reuses a cached file
whose attributes match, otherwise runs `climate_emulator.py solve` (with its [Fe/H] fallback). Errors with the emulator's
NoSolutionError message (closest planet and misfit) if no planet reproduces the targets.
"""
function emulator_wind_file(P_rot, T_day, T_night, P0; gcm = "exocam", feh = nothing, tol = 0.1, device = nothing, overwrite = false)
    name = "P$(_tag(P_rot))_Td$(_tag(T_day))_Tn$(_tag(T_night))_P$(_tag(P0))_$(gcm)" * (isnothing(feh) ? "" : "_FeH$(_tag(feh))")
    path = joinpath(directory, "winds", name * ".nc")
    if isfile(path) && !overwrite
        matches = NCDataset(path) do ds
            a = ds.attrib
            haskey(a, "T_day_target_C") && isapprox(a["P_rot_days"], P_rot) && isapprox(a["P0_bar"], P0) &&
                isapprox(a["T_day_target_C"], T_day) && isapprox(a["T_night_target_C"], T_night) && a["GCM"] == gcm
        end
        if matches
            @info "Emulator wind: reusing $(path)"
            return path
        end
        @warn "Emulator wind: $(path) exists but doesn't match the requested planet; re-solving"
    end
    mkpath(dirname(path))
    args = ["solve", "--P_rot=$(P_rot)", "--T_day=$(T_day)", "--T_night=$(T_night)", "--P0=$(P0)", "--out=$(path)", "--gcm=$(gcm)", "--tol=$(tol)"]
    isnothing(feh) || push!(args, "--feh=$(feh)")
    isnothing(device) || push!(args, "--device=$(device)")
    @info "Emulator wind: solving the CLERO planet for P_rot = $(P_rot) d, (T_day, T_night) = ($(T_day), $(T_night)) °C, P0 = $(P0) bar..."
    log = IOBuffer()
    proc = run(pipeline(ignorestatus(Cmd([EMULATOR_PYTHON, "-W", "ignore", EMULATOR_SCRIPT, args...])); stdout = log, stderr = log))
    lines = filter(l -> !isempty(l) && !occursin("CUDACachingAllocator", l), split(String(take!(log)), '\n'))
    if proc.exitcode != 0 || !isfile(path)
        reason = something(findlast(l -> occursin("NoSolutionError", l), lines), lastindex(lines))
        error("Emulator wind failed (exit code $(proc.exitcode)) for P_rot = $(P_rot) d, (T_day, T_night) = ($(T_day), $(T_night)) °C, " *
              "P0 = $(P0) bar:\n" * join(lines[max(reason - 2, 1):reason], "\n"))
    end
    foreach(l -> @info("Emulator: " * strip(l)), filter(l -> occursin(r"M_star|T_star|rho_air", l), lines))
    return path
end

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
# planet's lowest-level wind and near-surface air density, signed as wind_stress_u/v in ocean_sim.jl. Pass `P_rot` (days) to check the file;
# `scale` multiplies the stress (e.g. 0.5 to bracket the 10 m vs lowest-level wind height bias).
function emulator_wind_stress(path, grid; P_rot = nothing, scale = 1)
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
        return -scale * bilinear_periodic(ρa, lon, lat, λ, φ) * large_yeager_CD(W) * W * (component == :x ? U : V) / rho_seawater
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

# Emulator provenance for the ocean output's global attributes
function emulator_attributes(path, scale)
    a = NCDataset(ds -> Dict(ds.attrib), path)
    keys_out = ("P0_bar", "M_star_Msun", "T_star_K", "F_star_W_m2", "CO2", "FeH", "GCM", "T_day_emu_C", "T_night_emu_C", "emulator")
    return merge(Dict("emulator_wind_file" => path, "wind_stress_scale" => Float64(scale)),
                 Dict("emulator_" * k => a[k] for k in keys_out if haskey(a, k)))
end
