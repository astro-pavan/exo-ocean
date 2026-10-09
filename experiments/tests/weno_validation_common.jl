# Shared code for the weno_validation_*.jl and h100_test_*.jl scripts: run definitions, per-run logs, the scheduler and the H100 budget.
# Optional (local checks only): --dry-run or H100_TESTS_DRYRUN=1 prints the plan and estimates; --smoke or H100_TESTS_SMOKE=1 runs
# every case 4× coarser (at most 8°) for 20 model days; --force reruns cases already marked ok in the summary (otherwise an
# interrupted session resumes where it stopped).
using Distributed, Logging, Printf, Dates
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))

const VALIDATION_COMMON = @__FILE__
const SCRIPT_START = time()
const DONOR_1KM = "deepeq_P_10_dT_30.0_D_1"    # 1° biharmonic equilibrium that the weno_validation_a400.jl runs regrid from
const H100_TOTAL_HOURS = 8.5                    # pod hours for all h100_test_*.jl scripts together (~$34 at $3.5/h with setup)
const H100_RATE = 3.5                           # $ per hour

dry_run() = "--dry-run" in ARGS || get(ENV, "H100_TESTS_DRYRUN", "0") == "1"
smoke_mode() = "--smoke" in ARGS || get(ENV, "H100_TESTS_SMOKE", "0") == "1"

tidy(x) = isinteger(x) ? Int(x) : x
sci(x) = replace(@sprintf("%.0e", x), "e-0" => "e-", "e+0" => "e")
levels_for(H) = findfirst(n -> last(ocean_z_faces(H, n)) <= 11.5, 1:60)    # as in deep_equilibrium.jl

"""
One run with WENO5 momentum (T_night = 0, so T_day = ΔT); `cost` = expected minutes per model year on the target machine.
`pickup = (; name, res, depth)` warm-starts from the latest checkpoint of that run (regridded); `drift = (; tol, window)` (°C/yr, years)
stops at deep equilibrium, with `years` as the cap and `est_years` the expected length; `P0` (bar) adds CLERO emulator wind stress.
"""
function case(group, res, depth, P, dT; κ = 1e3, years, cost, cpu = false, warm = false, FT = Float32, prefix = "wenoval",
              ν_v = 1e-4, κ_v = 1e-4, nz_factor = 1, tracer = :upwind3, P0 = nothing, wind_scale = 1.0, pickup = nothing, drift = nothing,
              est_years = years)
    warm && (pickup = (name = DONOR_1KM, res = 1.0, depth = 1000))
    name = "$(prefix)_$(tidy(res))deg_D$(tidy(depth))_P$(tidy(P))_dT$(tidy(dT))" *
           (κ == 1e3 ? "" : "_kh$(tidy(κ))") *
           (tracer == :upwind3 ? "" : "_tr$(tracer)") *
           (ν_v == 1e-4 && κ_v == 1e-4 ? "" : "_nu$(sci(ν_v))_kv$(sci(κ_v))") *
           (nz_factor == 1 ? "" : "_nz$(nz_factor * levels_for(depth))") *
           (isnothing(P0) ? "" : "_windP$(tidy(P0))" * (wind_scale == 1 ? "" : "x$(tidy(wind_scale))")) *
           (warm ? "_warm" : isnothing(pickup) ? "" : "_from$(tidy(pickup.res))deg") *
           (FT == Float64 ? "_f64" : "")
    return (; group, name, res, depth, P, dT, κ, years, cost, cpu, FT, ν_v, κ_v, nz_factor, tracer, P0, wind_scale, pickup, drift, est_years)
end

# Shorter Δt for faster rotation (inertial cap) and stronger flow (CFL); Float64 costs ~1.3× on the H100
estimate_minutes(c) = c.cost * c.nz_factor * (c.P <= 3 ? 1.3 : 1.0) * (c.dT >= 50 ? 1.5 : 1.0) * (c.FT == Float64 ? 1.3 : 1.0) * c.est_years + 1

# MB per million cells, from the jetres files: 3D file (start + end) 38, zonal file 3.6 + 0.024 per record per (1°, 9 levels), checkpoint ~60
megacells(c) = (360 / c.res) * (160 / c.res) * c.nz_factor * levels_for(c.depth) / 1e6
output_mb(c) = 41.6 * megacells(c) + 0.024 * (c.nz_factor * levels_for(c.depth) / 9) / c.res * (10 * c.est_years + 1)
checkpoint_mb(c) = 60 * megacells(c) * (c.FT == Float64 ? 2 : 1)

function latest_checkpoint(dir, name)
    files = isdir(dir) ? filter(f -> occursin(Regex("^\\Q$(name)\\E_iteration\\d+\\.jld2\$"), f), readdir(dir)) : String[]
    isempty(files) && return nothing
    return joinpath(dir, files[argmax([parse(Int, match(r"_iteration(\d+)", f)[1]) for f in files])])
end

wind_file_name(c) = "P$(_tag(c.P))_Td$(_tag(c.dT))_Tn0_P$(_tag(c.P0))_exocam.nc"    # as cached by emulator_wind_file (exocam, default [Fe/H])

function smoke_case(c, names)
    pickup = isnothing(c.pickup) || !(c.pickup.name in names) ? c.pickup :
             (name = c.pickup.name * "_smoke", res = min(4c.pickup.res, 8.0), depth = c.pickup.depth)
    return merge(c, (; name = c.name * "_smoke", res = min(4c.res, 8.0), years = 20 / 365, est_years = 20 / 365, pickup,
                     drift = nothing))
end

# Print the plan and estimates; returns the runs still to do (longest first), nothing (kept for older scripts), and the summary file
function plan(cases, label; n_workers, prefix = "wenoval")
    smoke = smoke_mode()
    names = Set(c.name for c in cases)
    smoke && (cases = [smoke_case(c, names) for c in cases])
    ocean_dir = joinpath(directory, "ocean")
    summary_path = joinpath(ocean_dir, "$(prefix)_$(label)$(smoke ? "_smoke" : "")_summary.txt")

    done = Set{String}()
    if isfile(summary_path) && !("--force" in ARGS)
        for l in eachline(summary_path)
            m = match(r"^\S+ \S+\s+(\S+)\s+ok\b", l)
            isnothing(m) || push!(done, m[1])
        end
    end
    todo = sort(filter(c -> !(c.name in done), cases), by = estimate_minutes, rev = true)

    # Inputs that must already be on disk: emulator wind files (the pod has no Python) and checkpoints of runs from other scripts
    needed = String[]
    for c in todo
        !isnothing(c.P0) && !isfile(joinpath(directory, "winds", wind_file_name(c))) && push!(needed, "winds/" * wind_file_name(c))
        in_script = !isnothing(c.pickup) && any(d -> d.name == c.pickup.name, cases)
        !isnothing(c.pickup) && !in_script && isnothing(latest_checkpoint(joinpath(ocean_dir, "checkpoints"), c.pickup.name)) &&
            push!(needed, "ocean/checkpoints/$(c.pickup.name)_iteration*.jld2")
    end

    println("$(label): $(length(todo)) runs to do, $(length(cases) - length(todo)) already done, $(n_workers) at a time. Output: $(ocean_dir)")
    for c in todo
        @printf("  %-52s %5.2f°  %4d m  P %2d d  ΔT %2d K  %5.1f yr  %s  ~%5.2f h  ~%4.0f MB%s\n", c.name, c.res, c.depth,
                c.P, c.dT, c.est_years, c.cpu ? "CPU" : "GPU", estimate_minutes(c) / 60, output_mb(c) + checkpoint_mb(c),
                isnothing(c.pickup) ? "" : "  (from $(c.pickup.name))")
    end
    @printf("Estimate: %.1f h one at a time (longest run %.1f h); %.1f GB on disk, %.1f GB without checkpoints\n",
            sum(estimate_minutes, todo; init = 0.0) / 60, maximum(estimate_minutes, todo; init = 0.0) / 60,
            sum(c -> output_mb(c) + checkpoint_mb(c), todo; init = 0.0) / 1e3, sum(output_mb, todo; init = 0.0) / 1e3)
    isempty(needed) || println("Missing inputs (copy them into $(directory) first):\n  " * join(unique(needed), "\n  "))
    dry_run() && exit()
    isempty(needed) || error("missing inputs; see above")
    return todo, nothing, summary_path
end

# Hours this script may still start runs in: its own cap, limited by what the shared ledger says is left of H100_TOTAL_HOURS.
# The session's hours (including compilation) are added to the ledger when Julia exits, also after Ctrl-C.
function h100_budget(label, cap)
    ledger = joinpath(directory, "ocean", "h100_budget.txt")
    used = isfile(ledger) ? sum(parse(Float64, split(l)[end - 1]) for l in eachline(ledger) if !isempty(strip(l)); init = 0.0) : 0.0
    left = H100_TOTAL_HOURS - used
    @printf("H100 budget: %.2f h used by earlier scripts (\$%.0f), %.2f h left of %.1f; this script may start runs for %.2f h\n",
            used, used * H100_RATE, left, H100_TOTAL_HOURS, max(min(cap, left), 0))
    left <= 0 && (println("Budget exhausted; nothing to do. Ledger: $(ledger)"); exit())
    if !dry_run() && !smoke_mode()
        mkpath(dirname(ledger))
        atexit(() -> open(io -> @printf(io, "%s  %-28s %.3f h\n", Dates.format(now(), "yyyy-mm-dd HH:MM"), label,
                                       (time() - SCRIPT_START) / 3600), ledger, "a"))
    end
    return min(cap, left) - (time() - SCRIPT_START) / 3600
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

function run_case(c, _ = nothing)
    ocean_dir = joinpath(directory, "ocean")
    mkpath(ocean_dir)
    duration = c.years * 365days
    nan, t0 = Ref(false), time()
    status = open(joinpath(ocean_dir, "$(c.name).log"), "w") do io
        with_logger(RunLogger(io, c.name, nan)) do
            try
                pickup, pickup_grid = nothing, nothing
                if !isnothing(c.pickup)
                    pickup = latest_checkpoint(joinpath(ocean_dir, "checkpoints"), c.pickup.name)
                    isnothing(pickup) && error("no checkpoint of $(c.pickup.name) to warm-start from")
                    pickup_grid = (n_lon = round(Int, 360 / c.pickup.res), n_lat = round(Int, 160 / c.pickup.res),
                                   n_depth = levels_for(c.pickup.depth), ocean_depth = Float64(c.pickup.depth))
                end
                ocean_simulation(c.name, c.P, Float64(c.depth), R_Earth, Float64(c.dT), 0.0, duration;
                                 n_lon = round(Int, 360 / c.res), n_lat = round(Int, 160 / c.res), n_depth = c.nz_factor * levels_for(c.depth),
                                 vertical_stretching = 1.2^(1 / c.nz_factor),    # splits each default cell into nz_factor cells
                                 use_GPU = !c.cpu, float_type = c.FT, momentum_advection = :weno5, τ_biharmonic = nothing,
                                 κ_horizontal = c.κ, tracer_advection = c.tracer, ν_vertical = c.ν_v, κ_vertical = c.κ_v,
                                 emulator_wind = !isnothing(c.P0), surface_pressure = c.P0, wind_stress_scale = c.wind_scale,
                                 max_Δt = min(3hours * c.res, default_max_Δt(nothing, c.P)),    # ≈ 2× the CFL Δt; inertially stable
                                 pickup, pickup_grid,
                                 T_drift_tol = isnothing(c.drift) ? nothing : c.drift.tol,
                                 T_drift_window = isnothing(c.drift) ? 10 * 365days : c.drift.window * 365days,
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

# Each worker takes the longest remaining run whose warm-start donor (if it is in this script) has finished ok; runs whose donor failed
# are skipped. No run starts after the budget (running ones finish).
function run_all(todo, _, summary_path; budget_hours = Inf)
    started = time()
    queue = copy(todo)
    names = Set(c.name for c in todo)
    state = Dict{String, String}()
    donor_state(c) = isnothing(c.pickup) || !(c.pickup.name in names) ? "ok" : get(state, c.pickup.name, "pending")
    record(name, status, minutes) = begin
        line = @sprintf("%s  %-52s %-7s %7.1f min", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), name, status, minutes)
        println(line)
        open(io -> println(io, line), summary_path, "a")
    end
    @sync for w in workers()    # [1] when no workers were added
        @async while !isempty(queue) && (time() - started) / 3600 < budget_hours
            for c in filter(c -> donor_state(c) in ("nan", "error", "skipped"), queue)
                state[c.name] = "skipped"
                record(c.name, "skipped", 0.0)
                filter!(d -> d.name != c.name, queue)
            end
            i = findfirst(c -> donor_state(c) == "ok", queue)
            if isnothing(i)
                isempty(queue) && break
                sleep(20)
                continue
            end
            c = popat!(queue, i)
            state[c.name] = "running"
            r = remotecall_fetch(run_case, w, c)
            state[c.name] = r.status
            record(r.name, r.status, r.minutes)
        end
    end
    isempty(queue) || println("Budget reached; not started: ", join((c.name for c in queue), ", "))
    @printf("Done in %.1f h (session %.1f h, ≈ \$%.0f at \$%.1f/h). Summary: %s\n", (time() - started) / 3600,
            (time() - SCRIPT_START) / 3600, (time() - SCRIPT_START) / 3600 * H100_RATE, H100_RATE, summary_path)
end
