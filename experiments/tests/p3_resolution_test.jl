# What resolution does the sweep's fastest planet need as ΔT changes, and does the calculated Δt cap hold? P = 3 d, 1 km, no wind, ΔT = 1, 10 and 30 K.
# See docs/LLM/resolution_timestep_requirements.md for the cap. In priority order, each warm-started from its own 2° state:
#   spinup   2° to deep equilibrium for ΔT = 1 and 10 K; ΔT = 30 K uses the leak-free fastdiag_heat_P3_cap0.5, or spins up too if its checkpoint is missing
#   1deg     10 yr at ΔT = 1, 10, 30 K: the first test of the cap at 1.42–1.74 h
#   0.5deg   10 yr at ΔT = 1, 10, 30 K: convergence against 1°, and the first run of P3 past 1.45 yr at this resolution
#   0.25deg  5 yr at ΔT = 1 K, the case most likely to need it
# Estimated on the RTX 2000 alone: spin-ups 3–6 h each, 1° ~7.5 h in all, 0.5° ~43 h in all, 0.25° ~23 h. Rerunning skips finished runs and resumes cut-off ones.
#   experiments/tests/p3_resolution_test.sh                      launches everything in the background
#   julia --project=. -t 4 experiments/tests/p3_resolution_test.jl dT1_ 1.0deg    runs only tests whose names contain all of these
#   python analysis/p3_resolution_test.py                        summarises the results
include(joinpath(@__DIR__, "fast_rotation_diagnostics.jl"))

# min(0.5/ω, 0.30/f_wall, 4 h), with ω² = f² + 4c²(1/Δx² + 1/Δy²) at the row next to the wall and c ∝ √ΔT from the res1km runs
function Δt_cap(P, ΔT, resolution; max_lat = 80)
    c = (resolution >= 2 ? 0.75 : 0.85) * sqrt(ΔT / 30)
    row = max_lat - resolution / 2
    Δy = deg2rad(resolution) * R_Earth
    Δx = Δy * cosd(row)
    f = 2 * omega_Earth / P * sind(row)
    ω = sqrt(f^2 + 4c^2 * (1 / Δx^2 + 1 / Δy^2))
    return min(0.5 / ω, 0.30 / (2 * omega_Earth / P * sind(max_lat)), 4hours)
end

const P = 3
const EXISTING_SEEDS = Dict(30.0 => "fastdiag_heat_P3_cap0.5")

spin_up(ΔT) = (; resolution = 2.0, simulation_time = 100 * 365days, equilibrium_tol = 0.001ΔT, equilibrium_window = 20 * 365days, checkpoint_interval = 10 * 365days)
stage(resolution, years) = merge(fine(years), (; resolution))

sparse = (; frames = false, record_every = 70, budget_every = 70)    # the spin-ups only need the heat budget
routine = (; frames = false, record_every = 7, budget_every = 7)     # enough to see bursts, at little cost; a blow-up can be replayed from its yearly checkpoint

label(ΔT) = "dT$(Int(ΔT))"
common(ΔT) = merge(base, (; T_day = ΔT))
usable_seed(name) = is_finished(name) && !blew_up(name) && has_checkpoint(name)

# The 2° state to start from: an existing one if it's usable, otherwise a new spin-up from rest, otherwise rest itself, so that no test is skipped
function seed_for(ΔT)
    existing = get(EXISTING_SEEDS, ΔT, nothing)
    !isnothing(existing) && usable_seed(existing) && return existing
    isnothing(existing) || @warn "$(existing) has no usable checkpoint here, so spinning up ΔT = $(ΔT) K from rest"
    test_name = "p3res_$(label(ΔT))_spinup"
    run_test(test_name, P, spin_up(ΔT), (; max_Δt = Δt_cap(P, ΔT, 2.0)); donor = nothing, common = common(ΔT), recorder = sparse)
    seed = "fastdiag_$(test_name)"
    usable_seed(seed) && return seed
    @warn "$(seed) has no usable checkpoint, so the ΔT = $(ΔT) K tests start from rest and won't reach equilibrium"
    return nothing
end

# (ΔT, resolution, years)
const P3_TESTS = [(1.0, 1.0, 10), (10.0, 1.0, 10), (30.0, 1.0, 10), (1.0, 0.5, 10), (10.0, 0.5, 10), (30.0, 0.5, 10), (1.0, 0.25, 5)]
test_name(ΔT, resolution) = "p3res_$(label(ΔT))_$(resolution)deg"

if abspath(PROGRAM_FILE) == @__FILE__
    chosen = isempty(ARGS) ? P3_TESTS : filter(t -> all(occursin(a, test_name(t[1], t[2])) for a in ARGS), P3_TESTS)
    @info "Running $(length(chosen)) tests: $(join((test_name(t[1], t[2]) for t in chosen), ", "))"
    seeds = Dict(ΔT => seed_for(ΔT) for ΔT in unique(first.(chosen)))
    for (ΔT, resolution, years) in chosen
        max_Δt = Δt_cap(P, ΔT, resolution)
        @info "$(test_name(ΔT, resolution)): Δt cap $(round(max_Δt / hour, digits = 2)) h, starting from $(something(seeds[ΔT], "rest"))"
        run_test(test_name(ΔT, resolution), P, stage(resolution, years), (; max_Δt); donor = seeds[ΔT], common = common(ΔT), recorder = routine)
    end
    @info "All tests done"
end
