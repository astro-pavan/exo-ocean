# Why do the P = 3 and 5 d runs of base_case_1km_resolution.jl lose heat at 2° and blow up on fine grids? In priority order (1 km, ΔT = 30 K, no
# wind unless named, warm-started from the res1km_*_spinup runs, so run on the machine that has their checkpoints):
#   heat_*    2°, 10 yr: does the budget close with a shorter time-step cap, a varying Δt, another closure or a ±75° wall? P10 is the closed control
#   blow_*    1°: reproduce each blow-up with a step-by-step recorder (healthy P10 control), and try the same changes as possible fixes
#   wind_*    the same questions with emulator wind, which the sweep needs
#   half_*    0.5° at P3 with a cap scaled from the 1° test; it runs last and uses whatever time is left
# Every run writes to $EXO_OCEAN_OUTPUT/fastdiag/: <run>.csv (extremes and CFL per step), <run>_budget.csv (heat budget every 10 steps) and, for the
# 1° and finer runs, <run>_<event>.nc (the fields around the first CFL > 1 and the blow-up, every 5 steps then every step) with <run>_<event>.csv
# (largest values and where they are, per frame). Nothing in src/ocean changes: the recorder attaches through a more specific add_output_writers! method.
# Estimated ~19 h on the RTX 2000 alone (±50%), plus the half_ run. Rerunning skips finished runs and resumes cut-off ones.
#   experiments/tests/fast_rotation_diagnostics.sh               launches everything in the background with a 23.5 h limit
#   julia --project=. -t 4 experiments/tests/fast_rotation_diagnostics.jl heat_P3 blow_P5    runs only tests whose names contain these
#   python analysis/fast_rotation_diagnostics.py                 summarises the results
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

const DIAG_DIR = joinpath(directory, "fastdiag")
const RING_FRAMES = 16    # frames kept before an event, one every RING_INTERVAL steps
const RING_INTERVAL = 5
const POST_FRAMES = Dict("spike" => 5, "blowup" => 15)    # frames after the event, one per step
const ALARMS = Ref((; cfl = 1.0, speed = 4.0, w = 0.02))    # no-wind flows here peak near 1.5 m/s and 3e-3 m/s
const RECORDER = Ref{Any}(nothing)

#####
##### Recorder
#####

# (largest |x|, index) ignoring non-finite values, which an alarm reports separately
function largest(A)
    value, index = findmax(x -> isfinite(x) ? abs(x) : -one(x), A)
    return value, Tuple(index)
end

# One line per frame: time-step numbers and where each field peaks, enough to diagnose an event without downloading its fields
function frame_summary(frame, grid_info)
    (; λ, φ, z, z_faces, Δx, Δz, f_max) = grid_info
    Nx, Ny, Nz = size(frame.T)
    where_is(i, j, k, zs) = "$(round(λ[i], digits=1)),$(round(φ[min(j, Ny)], digits=1)),$(round(zs[k], digits=1))"
    u, iu = largest(frame.u)
    v, iv = largest(frame.v)
    w, iw = largest(frame.w)
    ΔT = diff(frame.T, dims = 3)
    inversion, ii = findmin(x -> isfinite(x) ? x : Inf32, ΔT ./ reshape(diff(z), 1, 1, :))    # most negative dT/dz: statically unstable
    u_cfl = maximum(x -> isfinite(x[2]) ? abs(x[2]) * frame.Δt / Δx[x[1][2]] : 0.0, pairs(frame.u))
    w_cfl = maximum(x -> isfinite(x[2]) ? abs(x[2]) * frame.Δt / Δz[min(x[1][3], Nz)] : 0.0, pairs(frame.w))
    drag = C_Bottom_Drag * maximum(x -> isfinite(x) ? abs(x) : 0.0, view(frame.u, :, :, 1)) * frame.Δt / Δz[1]
    η, iη = hasproperty(frame, :η) ? largest(frame.η) : (NaN32, (1, 1, 1))
    e, ie = hasproperty(frame, :e) ? largest(frame.e) : (NaN32, (1, 1, 1))
    return join((frame.iteration, frame.time, frame.Δt, u, where_is(iu..., z), v, where_is(iv..., z), w, where_is(iw..., z_faces), inversion,
                 where_is(Tuple(ii)..., z), u_cfl, w_cfl, drag, f_max * frame.Δt, η, e, where_is(ie..., z)), ";")
end

# Writes the frames as NetCDF, with `lon`, `lat`, `z` and the per-frame summary alongside
function write_event(name, event, frames, grid_info)
    path = joinpath(DIAG_DIR, "$(name)_$(event)")
    open(path * ".csv", "w") do io
        println(io, "iteration;time;dt;u;u_at;v;v_at;w;w_at;dTdz_min;dTdz_at;u_cfl;w_cfl;drag_number;f_dt;eta;e;e_at")
        foreach(frame -> println(io, frame_summary(frame, grid_info)), frames)
    end
    NCDataset(path * ".nc", "c") do ds
        defDim(ds, "frame", length(frames))
        for key in (:iteration, :time, :Δt)
            defVar(ds, String(key), [getproperty(frame, key) for frame in frames], ("frame",))
        end
        defVar(ds, "lon", grid_info.λ, (defDim(ds, "lon", length(grid_info.λ)); ("lon",)))
        defVar(ds, "lat", grid_info.φ, (defDim(ds, "lat", length(grid_info.φ)); ("lat",)))
        defVar(ds, "z", grid_info.z, (defDim(ds, "z", length(grid_info.z)); ("z",)))
        for key in filter(k -> first(frames)[k] isa Array, keys(first(frames)))
            A = first(frames)[key]
            dims = ntuple(d -> "$(key)_$(d)", ndims(A))
            foreach((d, n) -> defDim(ds, d, n), dims, size(A))
            var = defVar(ds, String(key), Float32, (dims..., "frame"); deflatelevel = 1)
            for (n, frame) in enumerate(frames)
                selectdim(var, ndims(A) + 1, n)[:] = frame[key]
            end
        end
    end
    @info "$(name): wrote the $(event) fields ($(length(frames)) frames) at iteration $(last(frames).iteration)"
end

function install_recorder!(sim, name, config)
    model, grid = sim.model, sim.model.grid
    mkpath(DIAG_DIR)
    record_every, budget_every = get(config, :record_every, 1), get(config, :budget_every, 10)    # budget_every must be a multiple of record_every
    Nx, Ny, Nz = size(grid)
    T, (u, v, w) = model.tracers.T, (model.velocities.u, model.velocities.v, model.velocities.w)

    λ, φ, z, z_faces = Array(λnodes(grid, Center())), Array(φnodes(grid, Center())), Array(znodes(grid, Center())), Array(znodes(grid, Face()))
    φ_faces = Array(φnodes(grid, Face()))
    Δz = diff(z_faces)
    Δx = [deg2rad(λ[2] - λ[1]) * grid.radius * cosd(p) for p in φ]
    f_max = 2 * omega_Earth / config.rotational_period * sind(maximum(abs, φ))
    grid_info = (; λ, φ, z, z_faces, Δx, Δz, f_max)
    polar_south, polar_north = 1:count(<=(-70), φ), Ny - count(>=(70), φ) + 1:Ny

    weights = [a * d for _ in 1:Nx, a in diff(sind.(φ_faces)), d in Δz]
    weights ./= sum(weights)
    heat_capacity = rho_seawater * cp_seawater * (z_faces[end] - z_faces[1])
    relaxation = (; T_day = config.T_day, T_night = config.T_night, W = Q_surface / (rho_seawater * cp_seawater))
    heat_flux = surface_heat_flux(model, relaxation)

    steps_path, budget_path = joinpath(DIAG_DIR, "$(name).csv"), joinpath(DIAG_DIR, "$(name)_budget.csv")
    isfile(steps_path) || write(steps_path, "iteration,time,dt,wall,cfl,u,v,w,T_min,T_max,u_polar,w_polar\n")
    isfile(budget_path) || write(budget_path, "iteration,time,dt,T_mean,flux,explicit_flux,heat_capacity\n")
    steps, budget = IOBuffer(), IOBuffer()
    flush_rows!() = (open(io -> write(io, take!(steps)), steps_path, "a"); open(io -> write(io, take!(budget)), budget_path, "a"))
    wall₀ = time()

    has_e = :e in keys(model.tracers)
    has_η = hasproperty(model.free_surface, :displacement)
    function take_frame()
        fields = (; T = Array{Float32}(interior(T)), u = Array{Float32}(interior(u)), v = Array{Float32}(interior(v)), w = Array{Float32}(interior(w)),
                  (has_η ? (; η = Array{Float32}(interior(model.free_surface.displacement))) : (;))...,
                  (has_e ? (; e = Array{Float32}(interior(model.tracers.e))) : (;))...)
        return (; iteration = model.clock.iteration, time = model.clock.time, Δt = Float64(sim.Δt), fields...)
    end

    ring, event, left, done = Any[], nothing, 0, Set{String}()
    function finish_event!()
        write_event(name, event, copy(ring), grid_info)
        push!(done, event)
        event = nothing
        empty!(ring)
        flush_rows!()
    end

    function record(sim)
        i, t = model.clock.iteration, model.clock.time
        U, V, W = maximum(abs, interior(u)), maximum(abs, interior(v)), maximum(abs, interior(w))
        T_min, T_max = minimum(interior(T)), maximum(interior(T))
        cfl = AdvectiveCFL(sim.Δt)(model)
        u_polar = max(maximum(abs, view(interior(u), :, polar_south, :)), maximum(abs, view(interior(u), :, polar_north, :)))
        w_polar = max(maximum(abs, view(interior(w), :, polar_south, :)), maximum(abs, view(interior(w), :, polar_north, :)))
        println(steps, join((i, t, sim.Δt, round(time() - wall₀, digits = 2), cfl, U, V, W, T_min, T_max, u_polar, w_polar), ","))

        if i % budget_every == 0
            explicit = sum(weights .* Array(interior(model.timestepper.Gⁿ.T))) * heat_capacity    # tendencies before the implicit vertical solve
            println(budget, join((i, t, sim.Δt, sum(weights .* Array(interior(T))), heat_flux(), explicit, heat_capacity), ","))
        end
        i % (record_every * (100 ÷ record_every)) == 0 && flush_rows!()
        config.frames || return nothing

        broken = !(isfinite(U) && isfinite(T_min) && isfinite(T_max))
        if isnothing(event)
            if "blowup" ∉ done && (broken || max(U, V) > ALARMS[].speed || W > ALARMS[].w)
                event, left = "blowup", POST_FRAMES["blowup"]
            elseif "spike" ∉ done && cfl > ALARMS[].cfl
                event, left = "spike", POST_FRAMES["spike"]
            end
        end
        if !isnothing(event)
            push!(ring, take_frame())    # every step during an event
            left -= 1
            (left <= 0 || broken) && finish_event!()
        elseif i % RING_INTERVAL == 0
            push!(ring, take_frame())
            length(ring) > RING_FRAMES && popfirst!(ring)
        end
        if broken
            flush_rows!()
            sim.running = false    # ocean_simulation then records the blow-up
        end
        return nothing
    end

    sim.callbacks[:recorder] = Callback(record, IterationInterval(record_every))
    haskey(config, :setup) && config.setup(sim, name)    # test-specific changes to the built simulation
    return nothing
end

function add_output_writers!(simulation::Simulation, name::AbstractString; kwargs...)
    isnothing(RECORDER[]) || install_recorder!(simulation, name, RECORDER[])
    return invoke(add_output_writers!, Tuple{Any, Any}, simulation, name; kwargs...)
end

#####
##### Tests
#####

base = (; ocean_depth = 1000.0, T_day = 30.0, T_night = 0.0, n_depth = 16, float_type = Float64, use_GPU = true, vertical_mixing = :catke,
        horizontal_mixing = :gm_redi, κ_vertical = 1e-4, prandtl = 10, emulator_wind = false)
coarse = (; resolution = 2.0, simulation_time = 10 * 365days, output_interval = 5 * 365days, checkpoint_interval = 5 * 365days)
fine(years) = (; resolution = 1.0, simulation_time = years * 365days, output_interval = 365days, checkpoint_interval = 365days)
wind = (; emulator_wind = true, surface_pressure = 2)
cap(P, factor = 1) = factor * min(4hours, default_max_Δt(P))    # as in base_case_1km_resolution.jl; ±75° keeps it so only the wall changes
horizontal = (; horizontal_mixing = :horizontal)
convective = (; vertical_mixing = :convective_adjustment)
lat75(P) = (; max_lat = 75, max_Δt = cap(P))

# (test name, P, stage, changes from the base case); f·Δt at the wall is 0.4 at the default cap for P ≤ 5, 0.2 for P10
const TESTS = [
    ("heat_P3_control", 3, coarse, (;)),
    ("heat_P3_cap0.5", 3, coarse, (; max_Δt = cap(3, 0.5))),
    ("heat_P3_cap0.75", 3, coarse, (; max_Δt = cap(3, 0.75))),
    ("heat_P3_cfl0.1", 3, coarse, (; cfl = 0.1)),                       # Δt shorter like cap0.5 but varying
    ("heat_P3_horizontal", 3, coarse, horizontal),
    ("heat_P3_convective", 3, coarse, convective),
    ("heat_P3_lat75", 3, coarse, lat75(3)),
    ("heat_P5_control", 5, coarse, (;)),
    ("heat_P5_cap0.5", 5, coarse, (; max_Δt = cap(5, 0.5))),
    ("heat_P10_control", 10, merge(coarse, (; simulation_time = 5 * 365days)), (;)),
    ("blow_P10_control", 10, fine(1), (;)),
    ("blow_P3_control", 3, fine(3), (;)),
    ("blow_P3_cap1h", 3, fine(4), (; max_Δt = 1hour)),
    ("blow_P3_cfl0.1", 3, fine(4), (; cfl = 0.1)),
    ("blow_P3_lat75", 3, fine(4), lat75(3)),
    ("blow_P3_horizontal", 3, fine(4), horizontal),
    ("blow_P3_convective", 3, fine(4), convective),
    ("blow_P5_control", 5, fine(5), (;)),
    ("blow_P5_cap1h", 5, fine(5), (; max_Δt = 1hour)),
    ("blow_P5_lat75", 5, fine(5), lat75(5)),
    ("blow_P5_convective", 5, fine(5), convective),
    ("wind_P3_2deg", 3, coarse, wind),
    ("wind_P3_1deg", 3, fine(2), wind),
    ("half_P3_cap30min", 3, merge(fine(2), (; resolution = 0.5)), (; max_Δt = 30minutes)),
]

donor_of(P) = "res1km_P$(P)_dT30_spinup"

function run_test(test_name, P, stage, changes; donor = donor_of(P), common = base, recorder = (;))
    name = "fastdiag_$(test_name)"
    if !isnothing(donor) && !(is_finished(donor) && !blew_up(donor) && has_checkpoint(donor))
        return @error "Skipping $(name): $(donor) isn't a finished run with a checkpoint here"
    end
    params = merge(common, (; max_Δt = cap(P)), stage, changes, (; name, rotational_period = P))
    RECORDER[] = merge((; rotational_period = P, params.T_day, params.T_night, frames = params.resolution < 2), recorder)
    try
        simulation(params, nothing, donor)
    catch err
        @error "$(name) failed; continuing" exception = (err, catch_backtrace())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    chosen = isempty(ARGS) ? TESTS : filter(t -> any(occursin(a, t[1]) for a in ARGS), TESTS)
    @info "Running $(length(chosen)) tests: $(join(first.(chosen), ", "))"
    foreach(t -> run_test(t...), chosen)
    @info "All tests done"
end
