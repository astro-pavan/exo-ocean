# Step cost of the sweep's speed-up options on the RTX 2000: 1 km, 16 levels, P = 10 d, ΔT = 30 K, 2 bar function wind, GM-Redi + CATKE, each case
# run from rest for RUN_MINUTES of wall time. Steps/s come from the timestamped progress lines (every 100 steps), so the log must be timestamped.
#   solo   one run at a time on the GPU, then the 2° cases on the CPU
#   share  a 1° run on the GPU with a 2° run beside it, first on the GPU, then on the CPU
# About 4 h in all (~12 min per case).
#   scripts/launch.sh experiments/tests/step_cost_benchmark.jl                  runs everything
#   julia --project=. -t auto experiments/tests/step_cost_benchmark.jl f32 1deg   runs only cases whose names contain all of these
#   python analysis/step_cost_benchmark.py                                      tabulates steps/s and min per model year
#launch limit 6h
#launch gpu RTX 2000
#launch timestamp
#launch threads auto
#launch conflicts single_depth_sweep_[a-z_]*|speedup_validation
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

const RUN_MINUTES = 10    # wall time per case, including the first step's compilation
const ZONAL = (1.0, 0.5)
const LOG_DIR = joinpath(REPO_ROOT, "logs")

physics = (; rotational_period = 10, T_day = 30.0, T_night = 0.0, ocean_depth = 1000.0, n_depth = 16, κ_vertical = 1e-4, prandtl = 1,
           vertical_mixing = :catke, horizontal_mixing = :gm_redi, function_wind = true, surface_pressure = 2)
storage = (; output_interval = nothing, snapshot_interval = nothing, checkpoint_interval = nothing)
off = (; tke_advection = false, tke_gm_redi = false)

# (case name, resolution, options); every case's name starts with bench_
const CASES = [
    ("1deg_f64", 1.0, (;)),
    ("1deg_f64_tkeadv", 1.0, (; tke_advection = false)),
    ("1deg_f64_tkegm", 1.0, (; tke_gm_redi = false)),
    ("1deg_f64_tkeoff", 1.0, off),
    ("1deg_f64_levelmix", 1.0, (; horizontal_mixing = :horizontal)),
    ("1deg_f32_tkeoff", 1.0, merge((; float_type = Float32), off)),
    ("zonal_f64", ZONAL, (;)),
    ("zonal_f64_tkeoff", ZONAL, off),
    ("zonal_f32_tkeoff", ZONAL, merge((; float_type = Float32), off)),
    ("2deg_f64", 2.0, (;)),
    ("2deg_f64_upwind3", 2.0, (; advection = :upwind3)),
    ("2deg_f64_up3mom", 2.0, (; advection = :upwind3_momentum)),
    ("2deg_f64_up3mom_tkeoff", 2.0, merge((; advection = :upwind3_momentum), off)),
    ("2deg_f64_cap2h", 2.0, (; max_Δt = 2hours)),
    ("2deg_f64_cpu", 2.0, (; use_GPU = false)),
    ("2deg_f64_up3mom_tkeoff_cpu", 2.0, merge((; advection = :upwind3_momentum, use_GPU = false), off)),
]

function run_case(tag, resolution, options; run_minutes = RUN_MINUTES)
    p = merge(physics, (; resolution, float_type = Float64, use_GPU = true), options)
    max_Δt = get(p, :max_Δt, timestep_cap(p, resolution))
    name = "bench_" * tag
    @info "Benchmark $(name): $(resolution)°, Δt cap $(round(max_Δt / hour, digits = 2)) h, $(p.use_GPU ? "GPU" : "CPU with $(Threads.nthreads()) threads"), $(run_minutes) min"
    try
        ocean_simulation(name, p.rotational_period, p.ocean_depth, R_Earth, p.T_day, p.T_night, 5 * 365days;
                         Base.structdiff(p, NamedTuple{(:rotational_period, :ocean_depth, :T_day, :T_night, :max_Δt)})..., storage..., max_Δt, wall_time_limit = 60 * run_minutes)
    catch err
        @error "Benchmark $(name) failed" exception = (err, catch_backtrace())
    end
    @info "Benchmark $(name) done"
end


# A second Julia process running one case for RUN_MINUTES + 3, logging with timestamps to logs/step_cost_benchmark_<tag>.log
function side_process(tag, threads)
    log = joinpath(LOG_DIR, "step_cost_benchmark_$(tag).log")
    julia = joinpath(Sys.BINDIR, "julia")
    script = "cd $(REPO_ROOT) && $(julia) --project=. -t $(threads) $(@__FILE__) --side $(tag) 2>&1 | gawk '{ print systime(), \$0; fflush() }' >> $(log)"
    return run(`sh -c $script`; wait = false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "--side"
        tag = ARGS[2]
        resolution, options = only(c[2:3] for c in CASES if c[1] == replace(tag, r"^share_" => ""))
        run_case(tag, resolution, options; run_minutes = RUN_MINUTES + 3)    # outlasts the main run, so it shares the machine throughout
    else
        @info "Machine: $(Sys.CPU_NAME), $(Sys.CPU_THREADS) CPU threads, this process has $(Threads.nthreads())"
        chosen = filter(c -> all(occursin(a, c[1]) for a in ARGS), CASES)
        foreach(c -> run_case(c...), chosen)
        if isempty(ARGS) || "share" in ARGS
            for (side, threads) in (("share_2deg_f64", 2), ("share_2deg_f64_cpu", max(1, Sys.CPU_THREADS - 2)))
                process = side_process(side, threads)
                sleep(120)    # the side run compiles first, so the main run's timed steps overlap its stepping
                run_case("1deg_f64_beside_$(replace(side, "share_" => ""))", 1.0, (;))
                wait(process)
            end
        end
    end
end
