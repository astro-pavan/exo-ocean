include("ocean_sim.jl")
using Oceananigans.Units

const RUN_FIELDS = (:name, :rotational_period, :ocean_depth, :T_day, :T_night)
const NOT_KEYWORDS = (RUN_FIELDS..., :simulation_time, :planet_radius)

# Default two-stage run: a 2° spin-up to deep equilibrium (capped at 300 years), then 10 years on the run's own grid with no stopping test
const SPIN_UP = (resolution = 2.0, simulation_time = 300 * 365days, checkpoint_interval = 10 * 365days)
const MAIN_TIME = 10 * 365days
const EQUILIBRIUM_TOL = 0.002    # default spin-up equilibrium_tol as a fraction of T_day − T_night

"""
    simulation(params, low_res_start = SPIN_UP, warm_start = nothing)

Run one `ocean_simulation` from `params`, a NamedTuple with `name`, `rotational_period`, `ocean_depth`, `T_day`, `T_night`, optionally
`simulation_time` (default 10 years) and `planet_radius` (default `R_Earth`), and any `ocean_simulation` keywords (e.g. `resolution`).

- `low_res_start`: settings that override `params` for a first, coarse stage run as `<name>_lowres`; the main run then warm-starts
  from it, regridded onto its own grid. By default a 2° spin-up stopped at deep equilibrium (`equilibrium_tol` = 0.2% of
  T_day − T_night unless given) or after 300 years; `nothing` runs the main stage alone.
- `warm_start`: name of an existing run to start from (the low-res stage if there is one, otherwise the main run).

A stage whose parameter file marks it finished is skipped, so rerunning picks up where a sweep stopped; an unfinished one reruns
from the start.
"""
function simulation(params, low_res_start = SPIN_UP, warm_start = nothing)
    missing_fields = filter(k -> !haskey(params, k), RUN_FIELDS)
    isempty(missing_fields) || error("params is missing $(missing_fields)")
    planet_radius = get(params, :planet_radius, R_Earth)

    function launch(name, p, from)
        is_finished(name) && return @info("Skipping $(name): already finished")
        ocean_simulation(name, p.rotational_period, p.ocean_depth, planet_radius, p.T_day, p.T_night, get(p, :simulation_time, MAIN_TIME);
                         Base.structdiff(p, NamedTuple{NOT_KEYWORDS})..., warm_start = from)
    end

    if isnothing(low_res_start)
        launch(params.name, params, warm_start)
    else
        spin_up = merge((; equilibrium_tol = EQUILIBRIUM_TOL * (params.T_day - params.T_night)), params, low_res_start)
        spin_up = merge(spin_up, (; checkpoint_interval = something(get(spin_up, :checkpoint_interval, nothing), spin_up.simulation_time)))    # the main run needs its checkpoint
        launch("$(params.name)_lowres", spin_up, warm_start)
        launch(params.name, params, "$(params.name)_lowres")
    end
    return nothing
end

"""
    sweep(base; low_res_start = SPIN_UP, warm_start = nothing, prefix = "sweep", name = nothing, swept...)

Run `simulation` for every combination of the swept values, e.g.
`sweep((ocean_depth = 1kilometer, T_night = 0.0, resolution = 0.5); rotational_period = [3, 10, 30], T_day = [10, 30])`.
`base` holds the fixed parameters; each run is named `<prefix>_<key><value>_...` unless `name(params)` is given. `warm_start` is a run
name or a function `params -> run name`. A failed run is logged and the sweep continues; returns each run's name and status.
"""
function sweep(base; low_res_start = SPIN_UP, warm_start = nothing, prefix = "sweep", name = nothing, swept...)
    keys_swept = Tuple(keys(swept))
    statuses = Pair{String, String}[]
    for values in Iterators.product(Base.values(swept)...)
        point = NamedTuple{keys_swept}(values)
        params = merge(base, point)
        run_name = isnothing(name) ? prefix * join("_$(k)$(v)" for (k, v) in pairs(point)) : name(params)
        params = merge(params, (; name = run_name))
        donor = warm_start isa Function ? warm_start(params) : warm_start
        try
            simulation(params, low_res_start, donor)
            push!(statuses, run_name => "ok")
        catch err
            @error "Run $(run_name) failed" exception = (err, catch_backtrace())
            push!(statuses, run_name => "error")
        end
    end
    foreach(s -> @info("$(s.first): $(s.second)"), statuses)
    return statuses
end
