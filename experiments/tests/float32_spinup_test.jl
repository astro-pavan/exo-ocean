# Do Float32 spin-ups reach the same equilibrium? Reruns finished Float64 2° spin-ups with identical settings in Float32 (WENO fix, TKE transport off as in
# the sweep), plus 1° and 1° × 0.5° Float32 main stages from the Float32 P = 10 d and P = 3 d spin-ups (compared with speedup_validation.jl's
# main_P10_tkeoff_f32 and main_P3_tkeoff_f32, which started from the Float64 ones), a Float32 rerun with the TKE transport on, a Float64 rerun with
# today's code (how much any rerun differs), and a new Float64/Float32 pair at κ = 1e-5 with the sweep's stop rule.
# The references' settings are frozen in float32_spinup_refs.toml (from their parameter files, written with --freeze), so the references' output needn't be here.
#   scripts/launch.sh experiments/tests/float32_spinup_test.jl                 runs everything (~3–4 h on the RTX 2000)
#   julia --project=. experiments/tests/float32_spinup_test.jl --freeze        rewrites float32_spinup_refs.toml from the references' parameter files
#   python analysis/float32_spinup_test.py                                    compares each rerun with its reference
#launch limit 12h
#launch gpu RTX 2000
#launch timestamp
#launch size ~3 GB
#launch conflicts single_depth_sweep_[a-z_]*|step_cost_benchmark|speedup_validation
include(joinpath(@__DIR__, "speedup_validation.jl"))    # its physics, run_test and the Float32 WENO fix
using TOML

const REFS_FILE = joinpath(@__DIR__, "float32_spinup_refs.toml")
const REFERENCES = ["spdval_P10_spinup", "spdval_P3_spinup", "fastdiag_p3res_dT1_spinup", "res1km_P10_dT1_spinup", "fastdiag_p3res_dT10_spinup"]
const F32 = (; float_type = Float32, tke_advection = false, tke_gm_redi = false)

# (run name, reference whose settings it copies, overrides); "kv1e-5" isn't a finished run: it is spdval_P10_spinup at κ = 1e-5 with the sweep's stop rule
const KV5 = (; κ_vertical = 1e-5, equilibrium_tol = 0.002 * 30, equilibrium_window = 20 * 365days, simulation_time = 150 * 365days)
const SPIN_UPS = [
    ("f32spin_spdval_P10_spinup", "spdval_P10_spinup", F32),
    ("f32spin_spdval_P3_spinup", "spdval_P3_spinup", F32),
    ("f32spin_fastdiag_p3res_dT1_spinup", "fastdiag_p3res_dT1_spinup", F32),
    ("f32spin_res1km_P10_dT1_spinup", "res1km_P10_dT1_spinup", F32),
    ("f32spin_fastdiag_p3res_dT10_spinup", "fastdiag_p3res_dT10_spinup", F32),
    ("f32spin_tkeon_spdval_P10_spinup", "spdval_P10_spinup", (; float_type = Float32)),
    ("f32spin_kv1e-5_f32", "spdval_P10_spinup", merge(KV5, F32)),
    ("f32spin_kv1e-5_float64", "spdval_P10_spinup", merge(KV5, (; tke_advection = false, tke_gm_redi = false))),    # differs from _f32 only in precision
    ("f32spin_float64_rerun_spdval_P10_spinup", "spdval_P10_spinup", (;)),
]

# Main stages from the Float32 spin-ups, as speedup_validation.jl's (test, P, resolution, years, options, donor, wall hours)
const MAINS = [("f32spin_main_P10", 10, 1.0, 10, merge(off, f32), "f32spin_spdval_P10_spinup", 2),
               ("f32spin_main_P3", 3, (1.0, 0.5), 5, merge(off, f32), "f32spin_spdval_P3_spinup", 2)]

# Run order: the sweep's own case first, each main stage right after its spin-up, the Float64 runs last
const ORDER = ["f32spin_spdval_P10_spinup", "f32spin_main_P10", "f32spin_spdval_P3_spinup", "f32spin_main_P3", "f32spin_fastdiag_p3res_dT1_spinup",
               "f32spin_res1km_P10_dT1_spinup", "f32spin_fastdiag_p3res_dT10_spinup", "f32spin_tkeon_spdval_P10_spinup", "f32spin_kv1e-5_f32",
               "f32spin_kv1e-5_float64", "f32spin_float64_rerun_spdval_P10_spinup"]
# f32spin_kv1e-5_f64 and f32spin_f64rerun_spdval_P10_spinup (12 Nov) ran in Float32 by mistake (float_type wasn't passed); the _float64 names replace them

freeze() = open(io -> TOML.print(io, Dict(r => read_parameters(r) for r in REFERENCES); sorted = true), REFS_FILE, "w")

nothing_or(x) = x == "nothing" ? nothing : x

# ocean_simulation arguments that reproduce a reference; errors on any setting it doesn't know, so nothing is silently dropped
function settings(ref)
    p = TOML.parsefile(REFS_FILE)[ref]
    planet, grid, physics, time = p["planet"], p["grid"], p["physics"], p["time"]
    isnothing(nothing_or(get(p["warm_start"], "donor", "nothing"))) || error("$(ref) was warm-started, so it can't be rerun from rest")
    for (k, want) in (("heat_exchange_coefficient", Q_surface), ("C_Bottom_Drag", C_Bottom_Drag), ("salinity", "nothing"), ("emulator_wind", false),
                      ("wind_field", "nothing"), ("wind_file", "nothing"), ("tidal_eccentricity", "nothing"), ("emulator_climate", "nothing"))
        get(physics, k, want) == want || error("$(ref): $(k) = $(physics[k]) isn't supported here")
    end
    isapprox(physics["ν_vertical"], physics["prandtl"] * physics["κ_vertical"]; rtol = 1e-6) || error("$(ref): ν_vertical isn't prandtl × κ_vertical")
    used = Set(["heat_exchange_coefficient", "C_Bottom_Drag", "salinity", "emulator_wind", "wind_field", "wind_file", "tidal_eccentricity", "emulator_climate",
                "ν_vertical", "tidal_Q", "tidal_fraction", "tidal_z_p", "tidal_Γ", "tidal_κ_max", "tidal_prandtl"])
    kw = Dict{Symbol, Any}(:resolution => grid["resolution"] isa Vector ? Tuple(grid["resolution"]) : grid["resolution"], :n_depth => grid["n_depth"],
                           :max_lat => grid["max_lat"], :vertical_stretching => grid["vertical_stretching"], :use_GPU => true,
                           :float_type => Dict("Float32" => Float32, "Float64" => Float64)[grid["float_type"]])    # ocean_simulation's default is Float32
    for k in ("κ_horizontal", "κ_vertical", "prandtl", "tke_advection", "tke_gm_redi", "implicit_vertical_diffusion", "function_wind", "wind_stress_scale",
              "wind_current_coupling")
        haskey(physics, k) && (kw[Symbol(k)] = physics[k]; push!(used, k))
    end
    for k in ("advection", "vertical_mixing", "horizontal_mixing")
        haskey(physics, k) && (kw[Symbol(k)] = Symbol(physics[k]); push!(used, k))
    end
    haskey(kw, :horizontal_mixing) || (kw[:horizontal_mixing] = :horizontal)    # files from before the option used level diffusion
    if get(physics, "function_wind", false)
        kw[:surface_pressure] = physics["surface_pressure"]
        kw[:air_molar_mass] = physics["air_molar_mass"]
    end
    push!(used, "surface_pressure", "air_molar_mass")
    unknown = setdiff(keys(physics), used)
    isempty(unknown) || error("$(ref): physics settings not handled: $(collect(unknown))")
    for k in ("cfl", "max_Δt", "free_surface_cfl", "equilibrium_window", "output_interval", "snapshot_interval", "checkpoint_interval")
        haskey(time, k) && (kw[Symbol(k)] = nothing_or(time[k]))
    end
    kw[:equilibrium_tol] = nothing_or(time["equilibrium_tol"])
    unknown = setdiff(keys(time), ["cfl", "max_Δt", "free_surface_cfl", "equilibrium_window", "output_interval", "snapshot_interval", "checkpoint_interval",
                                   "equilibrium_tol", "simulation_time", "save_last", "keep_checkpoints", "compress_output"])
    isempty(unknown) || error("$(ref): time settings not handled: $(collect(unknown))")
    kw[:checkpoint_interval] = something(get(kw, :checkpoint_interval, nothing), 10 * 365days)    # main stages warm-start from the last checkpoint
    return (; P = planet["rotational_period"], D = planet["ocean_depth"], R = planet["planet_radius"], T_day = planet["T_day"], T_night = planet["T_night"],
            simulation_time = time["simulation_time"]), (; kw...)
end

function run_spin_up(name, ref, overrides)
    is_finished(name) && return @info("Skipping $(name): already finished")
    planet, kw = settings(ref)
    simulation_time = get(overrides, :simulation_time, planet.simulation_time)
    kw = merge(kw, Base.structdiff(overrides, NamedTuple{(:simulation_time,)}))
    @info "Validation $(name): $(kw.resolution)°, $(kw.float_type), as $(ref) with $(overrides), up to $(round(simulation_time / 365days)) yr"
    ocean_simulation(name, planet.P, planet.D, planet.R, planet.T_day, planet.T_night, simulation_time; kw..., wall_time_limit = 4 * 3600)
    recorded = read_parameters(name)["grid"]["float_type"]
    recorded == string(kw.float_type) || error("$(name) ran in $(recorded), not $(kw.float_type)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if ARGS == ["--freeze"]
        freeze()
        @info "Wrote $(REFS_FILE)"
    else
        foreach(s -> settings(s[2]), SPIN_UPS)    # fail now, not hours in, if a reference can't be reproduced
        runs = Dict([s[1] => () -> run_spin_up(s...) for s in SPIN_UPS]..., [m[1] => () -> run_test(m...) for m in MAINS]...)
        for name in ORDER
            try
                runs[name]()
            catch err
                @error "$(name) failed; moving on" exception = (err, catch_backtrace())
            end
        end
        @info "Float32 spin-up test finished"
    end
end
