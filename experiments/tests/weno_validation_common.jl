# Shared code for weno_validation_h100.jl and weno_validation_a400.jl: run definitions, per-run logs, and the scheduler.
# Optional flags for both scripts: --dry-run (print the plan and estimates), --smoke (every run 4× coarser for 20 model days),
# --force (rerun runs already marked ok in the summary; otherwise an interrupted session resumes where it stopped).
using Distributed, Logging, Printf, Dates
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))

const VALIDATION_COMMON = @__FILE__
const DONOR_1KM = "deepeq_P_10_dT_30.0_D_1"    # 1° equilibrium that the warm-started 1 km runs regrid from

tidy(x) = isinteger(x) ? Int(x) : x
levels_for(H) = findfirst(n -> last(ocean_z_faces(H, n)) <= 11.5, 1:60)    # as in deep_equilibrium.jl

# One run with WENO5 momentum and UpwindBiased(3) tracers (T_night = 0, so T_day = ΔT); `cost` = expected minutes per model year
function case(group, res, depth, P, dT; κ = 1e3, years, cost, cpu = false, warm = false)
    name = "wenoval_$(tidy(res))deg_D$(tidy(depth))_P$(tidy(P))_dT$(tidy(dT))" * (κ == 1e3 ? "" : "_kh$(tidy(κ))") * (warm ? "_warm" : "")
    return (; group, name, res, depth, P, dT, κ, years, cost, cpu, warm)
end

# Shorter Δt for faster rotation (inertial cap) and stronger flow (CFL)
estimate_minutes(c) = c.cost * (c.P <= 3 ? 1.3 : 1.0) * (c.dT >= 50 ? 1.5 : 1.0) * c.years + 1

# MB per million cells, from the jetres files: 3D file (start + end) 38, zonal file 3.6 + 0.024 per record per (1°, 9 levels), checkpoint ~60
megacells(c) = (360 / c.res) * (160 / c.res) * levels_for(c.depth) / 1e6
output_mb(c) = 41.6 * megacells(c) + 0.024 * (levels_for(c.depth) / 9) / c.res * (10 * c.years + 1)
checkpoint_mb(c) = 60 * megacells(c)

function latest_checkpoint(dir, name)
    files = isdir(dir) ? filter(f -> occursin(Regex("^$(name)_iteration\\d+\\.jld2\$"), f), readdir(dir)) : String[]
    isempty(files) && return nothing
    return joinpath(dir, files[argmax([parse(Int, match(r"_iteration(\d+)", f)[1]) for f in files])])
end

# Print the plan and estimates; returns the runs still to do (longest first), the 1 km donor checkpoint, and the summary file
function plan(cases, label; n_workers)
    smoke = "--smoke" in ARGS
    smoke && (cases = [merge(c, (; name = c.name * "_smoke", res = 4c.res, years = 20 / 365)) for c in cases])
    ocean_dir = joinpath(directory, "ocean")
    summary_path = joinpath(ocean_dir, "wenoval_$(label)$(smoke ? "_smoke" : "")_summary.txt")

    done = Set{String}()
    if isfile(summary_path) && !("--force" in ARGS)
        for l in eachline(summary_path)
            m = match(r"^\S+ \S+\s+(\S+)\s+ok\b", l)
            isnothing(m) || push!(done, m[1])
        end
    end
    todo = sort(filter(c -> !(c.name in done), cases), by = estimate_minutes, rev = true)

    donor = any(c -> c.warm, todo) ? latest_checkpoint(joinpath(ocean_dir, "checkpoints"), DONOR_1KM) : nothing
    any(c -> c.warm, todo) && isnothing(donor) && error("the warm-started runs need a checkpoint of $(DONOR_1KM) in $(ocean_dir)/checkpoints")

    println("WENO5 validation ($(label)): $(length(todo)) runs to do, $(length(cases) - length(todo)) already done, $(n_workers) at a time. Output: $(ocean_dir)")
    for c in todo
        @printf("  %-40s %-8s %5.2f°  %4d m  P %2d d  ΔT %2d K  %5.1f yr  %s  ~%5.1f h  ~%4.0f MB\n", c.name, c.group, c.res, c.depth,
                c.P, c.dT, c.years, c.cpu ? "CPU" : "GPU", estimate_minutes(c) / 60, output_mb(c) + checkpoint_mb(c))
    end
    serial = sum(estimate_minutes, todo; init = 0.0) / 60
    @printf("Estimate: %.1f h one at a time (the longest run alone: %.1f h); %.1f GB on disk, %.1f GB without checkpoints\n",
            serial, maximum(estimate_minutes, todo; init = 0.0) / 60,
            sum(c -> output_mb(c) + checkpoint_mb(c), todo; init = 0.0) / 1e3, sum(output_mb, todo; init = 0.0) / 1e3)
    "--dry-run" in ARGS && exit()
    return todo, donor, summary_path
end

# Worker processes compile once each; they get LD_LIBRARY_PATH without CUDA paths so CUDA.jl loads its own libraries
function start_workers(n; threads)
    cleaned = join(filter(p -> !occursin(r"cuda"i, p), split(get(ENV, "LD_LIBRARY_PATH", ""), ":", keepempty = false)), ":")
    if n > 1
        addprocs(n; exeflags = `--project=$(Base.active_project()) -t $(threads)`, env = ["LD_LIBRARY_PATH" => cleaned])
    elseif cleaned != get(ENV, "LD_LIBRARY_PATH", "")
        @warn "LD_LIBRARY_PATH contains CUDA paths, so CUDA.jl may load mismatched system libraries; remove it before starting"
    end
    return nothing
end

# Writes each log line to the run's file with a Unix-time prefix (as analysis/jet_resolution.py expects) and flags NaN stops
struct RunLogger <: AbstractLogger
    io::IO
    name::String
    nan::Base.RefValue{Bool}
end
Logging.min_enabled_level(::RunLogger) = Logging.Info
Logging.shouldlog(::RunLogger, args...) = true
Logging.catch_exceptions(::RunLogger) = true
function Logging.handle_message(l::RunLogger, level, message, _module, group, id, file, line; kwargs...)
    msg = string(message)
    occursin("NaN found", msg) && (l.nan[] = true)
    for (k, v) in kwargs
        msg *= k == :exception ? "\n" * sprint(showerror, v...) : "\n$(k) = $(v)"
    end
    prefix = level >= Logging.Error ? "┌ Error: " : level >= Logging.Warn ? "┌ Warning: " : "[ Info: "
    stamp = string(round(Int, time()))
    for (i, s) in enumerate(split(msg, '\n'))
        println(l.io, stamp, " ", i == 1 ? prefix : "│ ", s)
    end
    flush(l.io)
    startswith(msg, "Time:") || println("[$(l.name)] ", first(split(msg, '\n')))    # progress lines go to the log only
end

function run_case(c, donor)
    ocean_dir = joinpath(directory, "ocean")
    mkpath(ocean_dir)
    duration = c.years * 365days
    pickup, pickup_grid = c.warm ? (donor, (n_lon = 360, n_lat = 160, n_depth = levels_for(1000), ocean_depth = 1000)) : (nothing, nothing)
    nan, t0 = Ref(false), time()
    status = open(joinpath(ocean_dir, "$(c.name).log"), "w") do io
        with_logger(RunLogger(io, c.name, nan)) do
            try
                ocean_simulation(c.name, c.P, Float64(c.depth), R_Earth, Float64(c.dT), 0.0, duration;
                                 n_lon = round(Int, 360 / c.res), n_lat = round(Int, 160 / c.res), n_depth = levels_for(c.depth),
                                 use_GPU = !c.cpu, momentum_advection = :weno5, τ_biharmonic = nothing, κ_horizontal = c.κ,
                                 max_Δt = min(3hours * c.res, default_max_Δt(nothing, c.P)),    # ≈ 2× the CFL Δt; inertially stable
                                 pickup, pickup_grid,
                                 output_interval = duration, zonal_mean_interval = min(36.5days, duration / 10),
                                 checkpoint_interval = min(10 * 365days, duration))
                nan[] ? "nan" : "ok"
            catch err
                @error "Run failed" exception = (err, catch_backtrace())
                "error"
            end
        end
    end
    # The checkpointer also saves the initial state; keep only the latest
    dir = joinpath(ocean_dir, "checkpoints")
    initial = "$(c.name)_iteration0.jld2"
    isdir(dir) && isfile(joinpath(dir, initial)) &&
        any(f -> startswith(f, c.name * "_iteration") && f != initial, readdir(dir)) && rm(joinpath(dir, initial))
    GC.gc(true)
    CUDA_AVAILABLE && CUDA.reclaim()
    return (; c.name, status, minutes = (time() - t0) / 60)
end

# Each worker takes the longest remaining run; none starts after the budget (running ones finish)
function run_all(todo, donor, summary_path; budget_hours = Inf)
    started = time()
    queue = copy(todo)
    @sync for w in workers()    # [1] when no workers were added
        @async while !isempty(queue) && (time() - started) / 3600 < budget_hours
            c = popfirst!(queue)
            r = remotecall_fetch(run_case, w, c, donor)
            line = @sprintf("%s  %-40s %-5s %7.1f min", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), r.name, r.status, r.minutes)
            println(line)
            open(io -> println(io, line), summary_path, "a")
        end
    end
    isempty(queue) || println("Budget reached; not started: ", join((c.name for c in queue), ", "))
    @printf("Done in %.1f h. Summary: %s\n", (time() - started) / 3600, summary_path)
end
