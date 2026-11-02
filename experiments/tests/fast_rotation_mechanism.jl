# Follow-up to fast_rotation_diagnostics.jl, with the same recorder and output folder ($EXO_OCEAN_OUTPUT/fastdiag/). In priority order:
#   blow_P3_control    checks event recording and 1° speed on a run expected to blow up near 1 yr
#   probe_*            η at fixed points and ρcp⟨w T⟩ at the top face every step: the surface wave's frequency and direction, and whether it carries the lost heat
#   heat_P3_chi*       moves the Adams-Bashforth limit in ωΔt (0.50 at the default χ = 0.1; 0.60 at χ = 0.3; 0.27 at χ = 0.02)
#   heat_P3_fscfl0.35  doubles the barotropic substeps
#   heat_*_cap*, f0.4  narrows the f·Δt threshold at P3 and P5, and tests whether it holds at P10
#   wind_*             repeats the control and the likely cap with the 2 bar emulator
#   blow_*             1° runs: a noise-free donor, P5, and a shorter Δt, ±75° wall and convective adjustment as possible fixes
# Estimated ~16.5 h on the RTX 2000 alone, with 1° rates uncertain. Rerunning skips finished runs and resumes cut-off ones.
#   experiments/tests/fast_rotation_mechanism.sh                 launches everything in the background with a 17.75 h limit
#   julia --project=. -t 4 experiments/tests/fast_rotation_mechanism.jl probe chi    runs only tests whose names contain these
#   python analysis/fast_rotation_diagnostics.py                 summarises the results
include(joinpath(@__DIR__, "fast_rotation_diagnostics.jl"))

const PROBE_LATS = (-75, -45, -15, 15, 45, 75)
const PROBE_LONS = 9    # equally spaced, so Nx must be divisible by this

#####
##### Setup hooks
#####

function set_χ(χ)
    return function (sim, name)
        sim.model.timestepper.χ = χ
        @info "$(name): Adams-Bashforth χ = $(sim.model.timestepper.χ)"
    end
end

function install_probe!(sim, name)
    model, grid = sim.model, sim.model.grid
    Nx, Ny, Nz = size(grid)
    Nx % PROBE_LONS == 0 || error("$(name): $(Nx) longitudes aren't divisible by $(PROBE_LONS)")
    λ, φ, φ_faces = Array(λnodes(grid, Center())), Array(φnodes(grid, Center())), Array(φnodes(grid, Face()))
    is, js = 1:(Nx ÷ PROBE_LONS):Nx, [argmin(abs.(φ .- lat)) for lat in PROBE_LATS]
    area = reshape(diff(sind.(φ_faces)), 1, :)
    area = area ./ (Nx * sum(area))

    path = joinpath(DIAG_DIR, "$(name)_probe.csv")
    columns = ["eta_$(round(φ[j], digits = 1))_$(round(λ[i], digits = 1))" for j in js for i in is]
    isfile(path) || write(path, join(["iteration", "time", "dt", "top_flux", "eta_rms", columns...], ",") * "\n")
    rows = IOBuffer()

    function probe(sim)
        η = Array(interior(model.free_surface.displacement))[:, :, 1]
        top_flux = rho_seawater * cp_seawater * sum(area .* Array(interior(model.velocities.w))[:, :, Nz + 1] .* Array(interior(model.tracers.T))[:, :, Nz])
        points = (η[i, j] for j in js for i in is)
        println(rows, join((model.clock.iteration, model.clock.time, sim.Δt, top_flux, sqrt(sum(area .* η .^ 2)), points...), ","))
        model.clock.iteration % 100 == 0 && open(io -> write(io, take!(rows)), path, "a")
        return nothing
    end

    sim.callbacks[:probe] = Callback(probe, IterationInterval(1))
    return nothing
end

#####
##### Tests
#####

years(n) = merge(coarse, (; simulation_time = n * 365days, output_interval = n * 365days, checkpoint_interval = n * 365days))
thin = (; record_every = 7, budget_every = 7)    # 7 steps doesn't lock onto the wizard's 10 and cuts the recorder's cost at 2°
fine_budget = (; budget_every = 7)
QUIET_DONOR = "fastdiag_heat_P3_cap0.5"

# (test name, P, stage, changes from the base case, recorder settings, donor)
const MECHANISM_TESTS = [
    ("blow_P3_control", 3, fine(3), (;), fine_budget, donor_of(3)),
    ("probe_P3_control", 3, years(1.5), (;), merge(thin, (; setup = install_probe!)), donor_of(3)),
    ("probe_P3_cap0.75", 3, years(1), (; max_Δt = cap(3, 0.75)), merge(thin, (; setup = install_probe!)), donor_of(3)),
    ("heat_P3_chi0.3", 3, years(3), (;), merge(thin, (; setup = set_χ(0.3))), donor_of(3)),
    ("heat_P3_chi0.02_cap0.75", 3, years(3), (; max_Δt = cap(3, 0.75)), merge(thin, (; setup = set_χ(0.02))), donor_of(3)),
    ("heat_P3_fscfl0.35", 3, years(3), (; free_surface_cfl = 0.35), thin, donor_of(3)),
    ("heat_P3_cap0.875", 3, years(3), (; max_Δt = cap(3, 0.875)), thin, donor_of(3)),
    ("heat_P5_cap0.75", 5, years(5), (; max_Δt = cap(5, 0.75)), thin, donor_of(5)),
    ("heat_P5_cap0.875", 5, years(5), (; max_Δt = cap(5, 0.875)), thin, donor_of(5)),
    ("heat_P10_f0.4", 10, years(5), (; max_Δt = default_max_Δt(10)), thin, donor_of(10)),
    ("wind_P3_control", 3, years(3), wind, thin, donor_of(3)),
    ("wind_P3_cap0.75", 3, years(3), merge(wind, (; max_Δt = cap(3, 0.75))), thin, donor_of(3)),
    ("blow_P3_quietdonor", 3, fine(3), (;), fine_budget, QUIET_DONOR),
    ("blow_P5_control", 5, fine(5), (;), fine_budget, donor_of(5)),
    ("blow_P3_cap1h", 3, fine(3), (; max_Δt = 1hour), fine_budget, donor_of(3)),
    ("blow_P3_lat75", 3, fine(3), lat75(3), fine_budget, donor_of(3)),
    ("blow_P3_convective", 3, fine(3), convective, fine_budget, donor_of(3)),
]

if abspath(PROGRAM_FILE) == @__FILE__
    chosen = isempty(ARGS) ? MECHANISM_TESTS : filter(t -> any(occursin(a, t[1]) for a in ARGS), MECHANISM_TESTS)
    @info "Running $(length(chosen)) tests: $(join(first.(chosen), ", "))"
    foreach(t -> run_test(t[1:4]...; recorder = t[5], donor = t[6]), chosen)
    @info "All tests done"
end
