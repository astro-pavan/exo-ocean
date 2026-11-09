# Can the sweep's 10-year main stages run on a cheaper grid? P = 3 d, 1 km, no wind, warm-started from the 2° fastdiag_p3res_*_spinup runs and compared with the 0.5° and 1° runs of p3_resolution_test.jl:
#   zonal    10 yr at 1° × 0.5° (Δlon × Δlat): halves the cells and relaxes the wall-row Δt limit
#   ladder   7 yr at 1°, then 3 yr at 0.5° from its final state
# In priority order: ΔT = 30 K zonal, ΔT = 30 K ladder, ΔT = 1 K zonal, ΔT = 1 K ladder. Δt caps come from timestep_cap, as for the sweep.
# Estimated on the RTX 2000 alone (±30%): ΔT = 30 K zonal 2.9 h, ladder 3.8 h; ΔT = 1 K zonal 2.3 h, ladder 2.1 h; about 11.5 h in all with setup.
#   experiments/tests/main_stage_grid_test.sh                       launches everything on the RTX 2000 with a 12 h limit
#   julia --project=. -t 4 experiments/tests/main_stage_grid_test.jl dT1 ladder    runs only tests whose names contain all of these
#   python analysis/main_stage_grid_test.py                         compares the results with the reference runs
include(joinpath(@__DIR__, "p3_resolution_test.jl"))

const ZONAL_GRID = (1.0, 0.5)    # Δlon, Δlat in degrees

seed_of(ΔT) = "fastdiag_p3res_$(label(ΔT))_spinup"

function run_stage(test_name, ΔT, resolution, years, donor; settings = common(ΔT))
    max_Δt = timestep_cap(merge(settings, (; rotational_period = P)), resolution)
    @info "$(test_name): $(resolution)° for $(years) yr, Δt cap $(round(max_Δt / hour, digits = 2)) h, from $(donor)"
    run_test(test_name, P, stage(resolution, years), (; max_Δt); donor, common = settings, recorder = routine)
end

zonal_test(ΔT; seed = seed_of(ΔT), settings = common(ΔT), years = 10, grid = ZONAL_GRID) = run_stage("gridtest_$(label(ΔT))_1x0.5deg", ΔT, grid, years, seed; settings)

# The second stage warm-starts from the first, and run_test skips it if the first didn't finish
function ladder_test(ΔT; seed = seed_of(ΔT), settings = common(ΔT), years = (7, 3), grids = (1.0, 0.5))
    first_stage = "ladder_$(label(ΔT))_1deg"
    run_stage(first_stage, ΔT, grids[1], years[1], seed; settings)
    run_stage("ladder_$(label(ΔT))_0.5deg", ΔT, grids[2], years[2], "fastdiag_$(first_stage)"; settings)
end

const GRID_TESTS = [("zonal_dT30", () -> zonal_test(30.0)), ("ladder_dT30", () -> ladder_test(30.0)),
                    ("zonal_dT1", () -> zonal_test(1.0)), ("ladder_dT1", () -> ladder_test(1.0))]

if abspath(PROGRAM_FILE) == @__FILE__
    chosen = isempty(ARGS) ? GRID_TESTS : filter(t -> all(occursin(a, t[1]) for a in ARGS), GRID_TESTS)
    @info "Running $(length(chosen)) tests: $(join(first.(chosen), ", "))"
    foreach(t -> t[2](), chosen)
    @info "All tests done"
end
