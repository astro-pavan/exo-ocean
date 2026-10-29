# Which horizontal resolution does each (P, ΔT) need at 1 km without wind? Per planet, a 2° spin-up to deep equilibrium (CATKE, GM-Redi,
# κ = 1e-4, Pr = 10), then 10 years at 1° and at 0.5° warm-started from it, with yearly 3D output so years 5–10 can be compared. The
# planets bracket the jet width: fast rotation (P 3 and 5 d, ΔT 30 K; equatorial deformation radius ~2–3°), weak forcing (P 10 d,
# ΔT 1 K, ~1.7°) and the base case (P 10 d, ΔT 30 K, ~4°), which uses the finished gmredi_nowind_kv1e-4_Pr10 as its 2° stage.
#   experiments/tests/base_case_1km_resolution.sh [ref] [nz32]    launches two chains of two planets in the background, or run one:
#   julia --project=. -t 4 experiments/tests/base_case_1km_resolution.jl P10_dT30 P10_dT1 P5_dT30 P3_dT30    # all four, RTX 2000
#   ... P3_dT30 ref    adds 5 years at 0.25°;    ... P3_dT30 nz32    adds 5 years at 0.5° with 32 levels
# Estimated on the RTX 2000 alone (±50%; 1° and 0.5° rates not measured yet, GM-Redi ×1.2–1.6): spin-up 1–3 h, 1° ~1–2.5 h and 0.5°
# ~6–14 h per planet, so ~30–70 h for all four (split them over two jobs to be safe); 0.25° ~30–65 h and nz32 ~20–40 h each.
# Rerunning skips finished runs and resumes cut-off ones from their yearly checkpoints, so a job that hits its wall time continues.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

const PLANETS = Dict("P10_dT30" => (10, 30.0, "gmredi_nowind_kv1e-4_Pr10", false),    # (P, T_day, donor, spin up from the donor?)
                     "P10_dT1" => (10, 1.0, nothing, true),                            # from rest: a ΔT = 30 K state is far off
                     "P5_dT30" => (5, 30.0, "gmredi_nowind_kv1e-4_Pr10", true),
                     "P3_dT30" => (3, 30.0, "gmredi_nowind_kv1e-4_Pr10", true))
planets = filter(in(keys(PLANETS)), ARGS)
options = setdiff(ARGS, planets)
isempty(planets) && error("name one or more planets: $(join(sort(collect(keys(PLANETS))), ", "))")
options ⊆ ("ref", "nz32") || error("unknown options $(setdiff(options, ("ref", "nz32"))); the options are ref and nz32")

base = (; ocean_depth = 1000.0, T_night = 0.0, n_depth = 16, float_type = Float64, use_GPU = true, vertical_mixing = :catke,
        horizontal_mixing = :gm_redi, κ_vertical = 1e-4, prandtl = 10, emulator_wind = false)
spin_up = (; resolution = 2.0, simulation_time = 100 * 365days, equilibrium_tol = 0.03, equilibrium_window = 20 * 365days,
           checkpoint_interval = 10 * 365days)
main = (; simulation_time = 10 * 365days, output_interval = 365days)
extra = (; simulation_time = 5 * 365days, output_interval = 365days)
name(tag, stage) = "res1km_$(tag)_$(stage)"

for tag in planets
    P, T_day, donor, spin = PLANETS[tag]
    planet = merge(base, (; rotational_period = P, T_day, max_Δt = min(4hours, default_max_Δt(P))))    # 4 h as in gm_redi_a400.jl
    usable = !isnothing(donor) && is_finished(donor) && !blew_up(donor) && has_checkpoint(donor)
    !isnothing(donor) && !usable && @warn "$(tag): donor $(donor) isn't a finished run with a checkpoint here; spinning up from rest"
    from = usable ? donor : nothing
    start = spin || isnothing(from) ? name(tag, "spinup") : from
    stages = [(merge(main, (; resolution = res)), "$(res)deg") for res in (1.0, 0.5)]
    "ref" in options && push!(stages, (merge(extra, (; resolution = 0.25)), "0.25deg"))
    "nz32" in options && push!(stages, (merge(extra, (; resolution = 0.5, n_depth = 32, vertical_stretching = sqrt(1.2))), "0.5deg_nz32"))
    try
        start == from || simulation(merge(planet, spin_up, (; name = start)), nothing, from)
        blew_up(start) && error("the spin-up $(start) blew up")
        for (p, stage) in stages
            try
                simulation(merge(planet, p, (; name = name(tag, stage))), nothing, start)
            catch err
                @error "$(name(tag, stage)) failed; continuing" exception = (err, catch_backtrace())
            end
        end
    catch err
        @error "$(tag) failed; continuing with the next planet" exception = (err, catch_backtrace())
    end
end
