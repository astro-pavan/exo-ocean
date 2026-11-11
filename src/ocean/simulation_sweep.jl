include("ocean_sim.jl")
using Oceananigans.Units
using Printf

const RUN_FIELDS = (:name, :rotational_period, :ocean_depth, :T_day, :T_night)
const NOT_KEYWORDS = (RUN_FIELDS..., :simulation_time, :planet_radius)

# Default two-stage run: a 2° spin-up to deep equilibrium (capped at 300 years), then 10 years on the run's own grid with no stopping test
const SPIN_UP = (resolution = 2.0, simulation_time = 300 * 365days, checkpoint_interval = 10 * 365days, snapshot_interval = 365days, save_last = nothing)
const SWEEP_STORAGE = (keep_checkpoints = false, compress_output = true)    # a sweep's runs warm-start from each other's 3D snapshots
const HIGH_RES_SAVE_LAST = 3 * 365days    # 3D and surface output kept for grids finer than 1°
const MAIN_TIME = 10 * 365days
const MAIN_CHECKPOINT = 365days    # so a run cut off by the wall time loses at most a year when resumed
const EQUILIBRIUM_TOL = 0.002    # default spin-up equilibrium_tol as a fraction of T_day − T_night

"""
    simulation(params, low_res_start = SPIN_UP, warm_start = nothing; spin_up_name = "<name>_spinup", stages = (:spin_up, :main))

Run one `ocean_simulation` from `params`, a NamedTuple with `name`, `rotational_period`, `ocean_depth`, `T_day`, `T_night`, optionally
`simulation_time` (default 10 years) and `planet_radius` (default `R_Earth`), and any `ocean_simulation` keywords (e.g. `resolution`).

- `low_res_start`: settings that override `params` for a first, coarse stage run as `spin_up_name`; the main run then warm-starts
  from it, regridded onto its own grid. By default a 2° spin-up stopped at deep equilibrium (`equilibrium_tol` = 0.2% of
  T_day − T_night unless given) or after 300 years; `nothing` runs the main stage alone.
- `warm_start`: name of an existing run to start from (the low-res stage if there is one, otherwise the main run).
- `stages`: which of `:spin_up` and `:main` to run, so two workers can share a sweep (a main stage needs its spin-up finished).

A stage whose parameter file marks it finished is skipped, and an unfinished one with a checkpoint is resumed from it to its planned
end (it must have the same grid), so rerunning picks up where a sweep stopped; one with no checkpoint reruns from the start. The main
stage checkpoints every year unless `checkpoint_interval` is given.
"""
function simulation(params, low_res_start = SPIN_UP, warm_start = nothing; spin_up_name = "$(params.name)_spinup", stages = (:spin_up, :main))
    missing_fields = filter(k -> !haskey(params, k), RUN_FIELDS)
    isempty(missing_fields) || error("params is missing $(missing_fields)")
    planet_radius = get(params, :planet_radius, R_Earth)

    function launch(name, p, from)
        is_finished(name) && return blew_up(name) ? @error("Skipping $(name): it blew up (NaN)") : @info("Skipping $(name): already finished")
        !isnothing(from) && blew_up(from) && return @error("Skipping $(name): its warm start $(from) blew up (NaN)")
        resume = has_checkpoint(name) && isfile(parameter_file(name))
        resume && (check_same_grid(name, p); @info("Resuming $(name) from its latest checkpoint"))
        ocean_simulation(name, p.rotational_period, p.ocean_depth, planet_radius, p.T_day, p.T_night, get(p, :simulation_time, MAIN_TIME);
                         Base.structdiff(p, NamedTuple{NOT_KEYWORDS})..., (resume ? (; restart = name, resume) : (; warm_start = from))...)
    end

    main = merge((; checkpoint_interval = MAIN_CHECKPOINT), params)
    if isnothing(low_res_start)
        launch(params.name, main, warm_start)
    else
        spin_up = merge((; equilibrium_tol = EQUILIBRIUM_TOL * (params.T_day - params.T_night)), params, low_res_start)
        spin_up = merge(spin_up, (; checkpoint_interval = something(get(spin_up, :checkpoint_interval, nothing), spin_up.simulation_time)))    # the main run needs its checkpoint
        :spin_up in stages && launch(spin_up_name, spin_up, warm_start)
        :main in stages && launch(params.name, main, spin_up_name)
    end
    return nothing
end

# A resumed run takes its grid from its parameter file, so refuse one whose grid settings in `p` have changed since it started
function check_same_grid(name, p)
    g = read_parameters(name)["grid"]
    for k in (:resolution, :n_depth, :ocean_depth, :max_lat, :vertical_stretching, :float_type)
        haskey(p, k) || continue
        want = p[k] isa Tuple ? collect(p[k]) : p[k] isa Type ? string(p[k]) : p[k]
        g[String(k)] == want || error("can't resume $(name): $(k) is $(g[String(k)]) in its parameter file but $(p[k]) now; delete its checkpoint to rerun it")
    end
end

# First-baroclinic speed (m/s) from the res1km runs, see docs/LLM/resolution_timestep_requirements.md
baroclinic_speed(ΔT, resolution) = (resolution >= 2 ? 0.75 : 0.85) * sqrt(abs(ΔT) / 30)

const RESOLUTION_LADDER = (2.0, 1.0, 0.5, 0.25)

"""
    required_resolution(params; bounds = (0.5, 1.0), zonal_coarsening = 2)

Coarsest latitude spacing (degrees) on `RESOLUTION_LADDER`, within `bounds`, at or below a fifth of the equatorial deformation radius √(c/β). The jet's
width follows that radius (Zeng & Yang 2021; Showman & Polvani 2011, ApJ 738, 71), and measured half-widths are 0.64–1.14 of it. Spacings below 1°
are returned as `(Δlon, Δlat)` with `Δlon = zonal_coarsening × Δlat`: at P = 3 d a 1° × 0.5° grid matched 0.5° within 4% on jet speed and width
(`experiments/tests/main_stage_grid_test.jl`) at 0.3 of the cost. A `resolution` in `params` is kept.
"""
function required_resolution(params; bounds = (0.5, 1.0), zonal_coarsening = 2)
    haskey(params, :resolution) && return params.resolution
    choices = filter(r -> bounds[1] <= r <= bounds[2], RESOLUTION_LADDER)
    isempty(choices) && error("no resolution on $(RESOLUTION_LADDER) lies within $(bounds)")
    params.rotational_period == 0 && return first(choices)
    radius = get(params, :planet_radius, R_Earth)
    β = 2 * omega_Earth / params.rotational_period / radius
    L_eq = rad2deg(sqrt(baroclinic_speed(params.T_day - params.T_night, last(choices)) / β) / radius)
    i = findfirst(r -> r <= L_eq / 5, choices)
    Δφ = isnothing(i) ? last(choices) : choices[i]
    return Δφ < 1 ? (zonal_coarsening * Δφ, Δφ) : Δφ    # the jet is zonal, so fine grids only need the latitude spacing
end

"""
    timestep_cap(params, resolution)

`max_Δt` (s) for a run at `resolution`: min(0.5/ω, 0.30/f_wall, 4 h) from docs/LLM/resolution_timestep_requirements.md, and the explicit background
diffusion limit 0.25 Δz²/max(κ, ν) in the top cell unless `implicit_vertical_diffusion` is set. c is raised by (κ/1e-4)^(1/6) above the calibrated κ, since thermocline depth goes as κ^(1/3) (Lai & Yang 2022).
"""
function timestep_cap(params, resolution)
    P, max_lat, radius = params.rotational_period, get(params, :max_lat, 80), get(params, :planet_radius, R_Earth)
    Δλ, Δφ = resolution isa Tuple ? resolution : (resolution, resolution)
    κ = get(params, :κ_vertical, 1e-4)
    ν = something(get(params, :prandtl, nothing), 1) * κ
    c = baroclinic_speed(params.T_day - params.T_night, max(Δλ, Δφ)) * max(1, (κ / 1e-4)^(1 / 6))
    row = max_lat - Δφ / 2
    Δy = deg2rad(Δφ) * radius
    Δx = deg2rad(Δλ) * radius * cosd(row)
    f_scale = P == 0 ? 0.0 : 2 * omega_Earth / P
    ω = sqrt((f_scale * sind(row))^2 + 4c^2 * (1 / Δx^2 + 1 / Δy^2))
    Δz_top = last(ocean_z_faces(params.ocean_depth, get(params, :n_depth, 20); r = get(params, :vertical_stretching, 1.2)))
    diffusion = get(params, :implicit_vertical_diffusion, false) ? Inf : 0.25 * Δz_top^2 / max(κ, ν)
    return min(0.5 / ω, 0.30 / (f_scale * sind(max_lat)), 4hours, diffusion)
end

"""
    run_numerics(params; spin_up_resolution = 2.0, resolution_bounds = (0.5, 1.0))

Resolution and time-step caps of a run from its `params`, as `(; main = (; resolution, max_Δt), spin_up = (; resolution, max_Δt))`.
"""
function run_numerics(params; spin_up_resolution = 2.0, resolution_bounds = (0.5, 1.0))
    resolution = required_resolution(params; bounds = resolution_bounds)
    return (; main = (; resolution, max_Δt = timestep_cap(params, resolution)),
            spin_up = (; resolution = spin_up_resolution, max_Δt = timestep_cap(params, spin_up_resolution)))
end

# How a swept value appears in a run name, as (before, after) the value: P10, nu1e-5, 0.5deg, wind3bar
const NAME_TAGS = Dict(:rotational_period => ("P", ""), :T_day => ("Tday", ""), :T_night => ("Tnight", ""), :ocean_depth => ("D", "m"),
                       :resolution => ("", "deg"), :n_depth => ("nz", ""), :prandtl => ("Pr", ""), :κ_vertical => ("kv", ""),
                       :κ_horizontal => ("kh", ""), :surface_pressure => ("wind", "bar"), :vertical_mixing => ("", ""),
                       :horizontal_mixing => ("", ""), :wind_current_coupling => ("sw", ""))
compact(x::Real) = isinteger(x) ? string(Int(x)) : abs(x) < 0.01 ? replace(@sprintf("%.0e", x), "e-0" => "e-") : string(x)
compact(x) = string(x)
name_tag(key, value) = ((before, after) = get(NAME_TAGS, key, (String(key), "")); before * compact(value) * after)
run_name(prefix, point) = join([prefix; [name_tag(k, v) for (k, v) in pairs(point)]], "_")

"""
    sweep(base; low_res_start = SPIN_UP, warm_start = nothing, prefix = "sweep", swept...)

Run `simulation` for every combination of the swept values, e.g.
`sweep((; ocean_depth = 1000.0, T_night = 0.0, T_day = 30.0); prefix = "test", rotational_period = [3, 10], resolution = [1.0, 0.5])`
runs test_P3_1deg, test_P3_0.5deg, test_P10_1deg and test_P10_0.5deg. `base` holds the fixed parameters. Runs that differ only in
`resolution` (or `simulation_time`) share one spin-up, named without them (here test_P3_spinup and test_P10_spinup), since the spin-up
overrides both; give sweeps with different `base` settings different prefixes so they don't share spin-ups by mistake. `warm_start`
is a run name or a function `params -> run name`. A failed run is logged and the sweep continues; returns each run's name and status.
"""
function sweep(base; low_res_start = SPIN_UP, warm_start = nothing, prefix = "sweep", swept...)
    statuses = Pair{String, String}[]
    for values in Iterators.product(Base.values(swept)...)
        point = NamedTuple{Tuple(keys(swept))}(values)
        name = run_name(prefix, point)
        spin_up_name = run_name(prefix, Base.structdiff(point, NamedTuple{(:resolution, :simulation_time)})) * "_spinup"
        params = merge(SWEEP_STORAGE, base, point, (; name))
        minimum(get(params, :resolution, 1.0)) < 1 && !haskey(params, :save_last) && (params = merge(params, (; save_last = HIGH_RES_SAVE_LAST)))
        try
            simulation(params, low_res_start, warm_start isa Function ? warm_start(params) : warm_start; spin_up_name)
            push!(statuses, name => "ok")
        catch err
            @error "Run $(name) failed" exception = (err, catch_backtrace())
            push!(statuses, name => "error")
        end
    end
    foreach(s -> @info("$(s.first): $(s.second)"), statuses)
    return statuses
end
