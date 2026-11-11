# Do the sweep's speed-ups change the answer? 1 km, 16 levels, 4 bar function wind (the sweep's strongest), GM-Redi + CATKE, κ = 1e-4 unless stated.
# Each test warm-starts from a 2° WENO spin-up made here (P = 10 d and P = 3 d, ΔT = 30 K), as a sweep main stage starts from its own spin-up:
#   main_P10   10 yr at 1°: control; twin (CFL 0.199, which sets the run-to-run noise); TKE transport off; + CFL 0.3
#   main_P3    5 yr at 1° × 0.5°, the fastest-rotating case: control; TKE transport off
#   spinup     20 yr at 2° from the P = 10 d spin-up, as a tree edge: κ = 1e-5 with WENO, upwind-3 momentum, + TKE transport off;
#              κ = 1e-2 with explicit (Δt cap 0.91 h) and implicit (4 h) background vertical diffusion
#   float32    Float32 main stages with the WENO fix (float32_weno_fix.jl, simulation_speedups.md §12.6) against the Float64 runs with the same settings,
#              plus Float32 runs from rest and at the sweep's corners (stability only), and a Float32 2° spin-up (does the deep drift stall?)
# Estimated on the RTX 2000: spin-ups ~1–3 h each, main_P10 ~9–10 h, main_P3 ~4–5 h, spinup ~5 h; ~22 h in all. Rerunning skips finished runs.
#   scripts/launch.sh experiments/tests/speedup_validation.jl                 runs everything
#   scripts/launch.sh experiments/tests/speedup_validation.jl float32         runs FLOAT32_SET in its order (~17–20 h; P = 10 d answers within ~8 h)
#   julia --project=. -t 4 experiments/tests/speedup_validation.jl main_P10    runs only tests whose names contain all of these
#   python analysis/speedup_validation.py                                    compares each test with its control
#launch limit 30h
#launch gpu RTX 2000
#launch timestamp
#launch size ~5 GB
#launch conflicts single_depth_sweep_[a-z_]*|step_cost_benchmark
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))
include(joinpath(@__DIR__, "float32_weno_fix.jl"))    # rescaled WENO divergence smoothness for Float32 models

physics = (; T_day = 30.0, T_night = 0.0, ocean_depth = 1000.0, n_depth = 16, κ_vertical = 1e-4, prandtl = 1, float_type = Float64, use_GPU = true,
           vertical_mixing = :catke, horizontal_mixing = :gm_redi, function_wind = true, surface_pressure = 4)
output = (; output_interval = 365days, snapshot_interval = 36.5days, save_last = 3 * 365days, checkpoint_interval = 365days)
off = (; tke_advection = false, tke_gm_redi = false)
f32 = (; float_type = Float32)

donor(P) = "spdval_P$(P)_spinup"

# (test name, rotational period, resolution, years, options, donor, wall-time limit in hours)
const TESTS = [
    ("main_P10_control", 10, 1.0, 10, (;), donor(10), 6),
    ("main_P10_twin", 10, 1.0, 10, (; cfl = 0.199), donor(10), 6),
    ("main_P10_tkeoff", 10, 1.0, 10, off, donor(10), 6),
    ("main_P10_tkeoff_cfl0.3", 10, 1.0, 10, merge(off, (; cfl = 0.3)), donor(10), 6),
    ("spinup_kv1e-5_weno", 10, 2.0, 20, (; κ_vertical = 1e-5), donor(10), 3),
    ("spinup_kv1e-5_up3mom", 10, 2.0, 20, (; κ_vertical = 1e-5, advection = :upwind3_momentum), donor(10), 3),
    ("spinup_kv1e-5_up3mom_tkeoff", 10, 2.0, 20, merge(off, (; κ_vertical = 1e-5, advection = :upwind3_momentum)), donor(10), 3),
    ("spinup_kv1e-2_explicit", 10, 2.0, 20, (; κ_vertical = 1e-2), donor(10), 4),
    ("spinup_kv1e-2_implicit", 10, 2.0, 20, (; κ_vertical = 1e-2, implicit_vertical_diffusion = true), donor(10), 3),
    ("main_P3_control", 3, (1.0, 0.5), 5, (;), donor(3), 6),
    ("main_P3_tkeoff", 3, (1.0, 0.5), 5, off, donor(3), 6),
    ("main_P10_f32", 10, 1.0, 10, f32, donor(10), 2),
    ("main_P10_tkeoff_f32", 10, 1.0, 10, merge(off, f32), donor(10), 2),
    ("main_P3_f32", 3, (1.0, 0.5), 5, f32, donor(3), 2),
    ("main_P3_tkeoff_f32", 3, (1.0, 0.5), 5, merge(off, f32), donor(3), 2),
    ("rest_1deg_f32", 10, 1.0, 10, merge(off, f32), nothing, 2),
    ("rest_zonal_f32", 10, (1.0, 0.5), 10, merge(off, f32), nothing, 2),
    ("corner_kv1e-5_f32", 10, 1.0, 5, merge(off, f32, (; κ_vertical = 1e-5)), donor(10), 2),
    ("corner_kv1e-2_f32", 10, 1.0, 5, merge(off, f32, (; κ_vertical = 1e-2, implicit_vertical_diffusion = true)), donor(10), 2),
    ("corner_dT1_rest_f32", 10, 1.0, 5, merge(off, f32, (; T_day = 1.0)), nothing, 2),
    ("corner_P100_rest_f32", 100, 1.0, 5, merge(off, f32), nothing, 2),
    ("corner_P3_rest_f32", 3, (1.0, 0.5), 5, merge(off, f32), nothing, 2),
    ("spinup_kv1e-5_weno_f32", 10, 2.0, 20, merge(f32, (; κ_vertical = 1e-5)), donor(10), 3),
]

# The Float32 set in run order: from rest first (no donor), each Float32 test before its Float64 reference, the twin before P = 3 d (it sets the tolerances)
const FLOAT32_SET = ["rest_1deg_f32", "rest_zonal_f32", "main_P10_f32", "main_P10_tkeoff_f32", "main_P10_control", "main_P10_tkeoff", "main_P10_twin",
                     "main_P3_f32", "main_P3_tkeoff_f32", "main_P3_tkeoff", "main_P3_control",
                     "corner_kv1e-5_f32", "corner_kv1e-2_f32", "corner_dT1_rest_f32", "corner_P100_rest_f32", "corner_P3_rest_f32",
                     "spinup_kv1e-5_weno", "spinup_kv1e-5_weno_f32"]

kw(p) = Base.structdiff(p, NamedTuple{(:rotational_period, :ocean_depth, :T_day, :T_night)})

# 2° WENO spin-up from rest to the sweep's equilibrium test (20-yr window), capped at 4 h of wall time; a capped one still leaves a checkpoint to start from
function make_donor(P)
    name = donor(P)
    is_finished(name) && !blew_up(name) && return name
    p = merge(physics, (; rotational_period = P))
    settings = merge(p, (; resolution = 2.0, max_Δt = timestep_cap(p, 2.0), equilibrium_tol = 0.001 * p.T_day, equilibrium_window = 20 * 365days,
                         output_interval = 10 * 365days, snapshot_interval = 365days, checkpoint_interval = 10 * 365days, wall_time_limit = 4 * 3600))
    @info "Spinning up $(name)"
    ocean_simulation(name, P, p.ocean_depth, R_Earth, p.T_day, p.T_night, 150 * 365days; kw(settings)...)
    return name
end

function run_test(test, P, resolution, years, options, from, limit_hours)
    name = "spdval_" * test
    is_finished(name) && return @info("Skipping $(name): already finished")
    !isnothing(from) && blew_up(from) && return @error("Skipping $(name): $(from) blew up")
    p = merge(physics, (; rotational_period = P, resolution), options)
    max_Δt = timestep_cap(p, resolution)
    @info "Validation $(name): $(resolution)°, $(p.float_type), $(years) yr, Δt cap $(round(max_Δt / hour, digits = 2)) h, from $(something(from, "rest"))"
    ocean_simulation(name, P, p.ocean_depth, R_Earth, p.T_day, p.T_night, years * 365days; kw(p)..., output..., max_Δt,
                     (isnothing(from) ? (;) : (; warm_start = from))..., wall_time_limit = limit_hours * 3600)
end

if abspath(PROGRAM_FILE) == @__FILE__
    chosen = ARGS == ["float32"] ? [only(filter(t -> t[1] == n, TESTS)) for n in FLOAT32_SET] : filter(t -> all(occursin(a, t[1]) for a in ARGS), TESTS)
    @info "Running $(length(chosen)) tests: $(join(first.(chosen), ", "))"
    for t in chosen
        try
            if !isnothing(t[6])    # each donor is made just before the first test that needs it
                try
                    make_donor(t[2])
                catch err
                    @error "Spin-up for P = $(t[2]) d failed" exception = (err, catch_backtrace())
                end
                has_checkpoint(t[6]) || error("$(t[6]) has no checkpoint to start from")
            end
            run_test(t...)
        catch err
            @error "Test $(t[1]) failed; moving on" exception = (err, catch_backtrace())
        end
    end
    @info "Speed-up validation finished"
end
