# GPU smoke test of the new run machinery before the base-case tests: small, short runs through every code path, with pass/fail checks.
#   julia --project=. experiments/tests/smoke_test_a100.jl
# Covers: emulator wind solve, default two-stage run (Gregory equilibrium stop + regridded warm start), restart appending to the
# output, same-grid warm start, max_lat ≠ 80 with Float32 and a (Δlon, Δlat) resolution, an emulator-wind run, sweep spin-up sharing and skipping,
# and throughput at 2°, 1°, 0.5° and 0.25° (1 km, 16 levels, Float64) for cost estimates. ~15–20 min on an A100 including compilation.
# Deletes earlier smoke_* output first. Output: smoke_*.
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))
using Printf

results = Pair{String, Bool}[]
check(label, ok) = (push!(results, label => ok); println(ok ? "PASS  " : "FAIL  ", label); ok)
function step(f, label)
    try
        f()
    catch err
        @error "$(label) failed" exception = (err, catch_backtrace())
        check("$(label) ran", false)
    end
end
nc_times(name, kind) = isfile(joinpath(directory, "$(name)_$(kind).nc")) ? NCDataset(ds -> Array(ds["time"][:]), joinpath(directory, "$(name)_$(kind).nc")) : Float64[]
last_T_finite(name) = NCDataset(ds -> all(x -> !ismissing(x) && isfinite(x), ds["T"][:, :, end]), joinpath(directory, "$(name)_section.nc"))
finished(name) = is_finished(name) ? read_parameters(name)["finished"] : Dict{String, Any}()

for dir in (directory, joinpath(directory, "checkpoints")), f in (isdir(dir) ? readdir(dir) : String[])
    startswith(f, "smoke_") && rm(joinpath(dir, f))
end

base = (; rotational_period = 10, ocean_depth = 500.0, T_day = 30.0, T_night = 0.0, n_depth = 10, float_type = Float64,
          resolution = 2.0, output_interval = 365days, snapshot_interval = 30days, checkpoint_interval = 365days)

# 1. Emulator wind solve (needs the emulator's Python env)
step("emulator wind solve") do
    path = emulator_wind_file(10, 30.0, 0.0, 3; overwrite = true)
    check("emulator wrote $(basename(path))", isfile(path))
end

# 2. Default two-stage run: 4° spin-up stopped by the Gregory test (loose tolerance), then 1 yr at 2° regridded from it
step("two-stage run") do
    spin_up = merge(SPIN_UP, (; resolution = 4.0, simulation_time = 20 * 365days, equilibrium_tol = 1.0, equilibrium_window = 2 * 365days))
    simulation(merge(base, (; name = "smoke_twostage", simulation_time = 365days)), spin_up)
    s, m = finished("smoke_twostage_spinup"), finished("smoke_twostage")
    check("spin-up stopped by the equilibrium test (got $(get(s, "stopped_by", "unfinished")))", get(s, "stopped_by", "") == "equilibrium")
    check("main stage finished at 1 yr", isapprox(get(m, "model_time", 0.0), 365days; rtol = 1e-3))
    check("main stage regridded from the spin-up", read_parameters("smoke_twostage")["warm_start"]["regridded"] == true)
    check("main stage output finite", last_T_finite("smoke_twostage"))
end

# 3. Restart: 6 more months, appended to the same files with the clock carrying on
step("restart") do
    before = nc_times("smoke_twostage", "section")
    ocean_simulation("smoke_twostage", base.rotational_period, base.ocean_depth, R_Earth, base.T_day, base.T_night, 182.5days;
                     restart = "smoke_twostage", output_interval = base.output_interval, snapshot_interval = base.snapshot_interval,
                     checkpoint_interval = base.checkpoint_interval)
    after = nc_times("smoke_twostage", "section")
    check("restart appended snapshots ($(length(before)) → $(length(after)))", length(after) > length(before) && after[1:length(before)] == before)
    check("snapshot times evenly spaced across the restart", all(isapprox.(diff(after), 30days; rtol = 1e-3)))
    check("restart recorded and run finished at 1.5 yr", haskey(read_parameters("smoke_twostage"), "restarts") &&
                                                         isapprox(get(finished("smoke_twostage"), "model_time", 0.0), 547.5days; rtol = 1e-3))
end

# 4. Same-grid warm start (exact copy, fresh clock and files)
step("same-grid warm start") do
    simulation(merge(base, (; name = "smoke_samegrid", simulation_time = 30days)), nothing, "smoke_twostage")
    check("same-grid warm start not regridded", read_parameters("smoke_samegrid")["warm_start"]["regridded"] == false)
    check("same-grid warm start output finite", last_T_finite("smoke_samegrid"))
end

# 5. max_lat = 70, Float32, (Δlon, Δlat) = (2°, 1°)
step("max_lat / Float32 run") do
    simulation(merge(base, (; name = "smoke_maxlat70", max_lat = 70, float_type = Float32, resolution = (2.0, 1.0), simulation_time = 30days)), nothing)
    lat, nlat = NCDataset(ds -> (Array(ds["φ_aca"][:]), length(ds["φ_aca"])), joinpath(directory, "smoke_maxlat70_surface.nc"))
    check("max_lat 70 grid: 140 latitudes within ±70°", nlat == 140 && maximum(abs, lat) < 70)
    check("Float32 output finite", last_T_finite("smoke_maxlat70"))
end

# 6. Emulator-wind run through ocean_simulation (reuses the file from step 1)
step("emulator wind run") do
    simulation(merge(base, (; name = "smoke_wind3bar", emulator_wind = true, surface_pressure = 3, simulation_time = 30days)), nothing)
    attrs = NCDataset(ds -> Dict(ds.attrib), joinpath(directory, "smoke_wind3bar_section.nc"))
    check("wind run records its emulator file", haskey(attrs, "emulator_wind_file"))
    check("wind run output finite", last_T_finite("smoke_wind3bar"))
end

# 7. Sweep: two resolutions share one 8° spin-up, and a second identical sweep skips everything
step("sweep") do
    tiny = merge(base, (; simulation_time = 30days))
    spin_up = merge(SPIN_UP, (; resolution = 8.0, simulation_time = 365days))
    sweep(tiny; low_res_start = spin_up, prefix = "smoke_sweep", resolution = [4.0, 2.0])
    spin_ups = filter(f -> startswith(f, "smoke_sweep") && endswith(f, "_spinup_parameters.toml"), readdir(directory))
    check("both resolutions used one spin-up (found $(spin_ups))", spin_ups == ["smoke_sweep_spinup_parameters.toml"] &&
                                                                   is_finished("smoke_sweep_4deg") && is_finished("smoke_sweep_2deg"))
    t = @elapsed statuses = sweep(tiny; low_res_start = spin_up, prefix = "smoke_sweep", resolution = [4.0, 2.0])
    check("second sweep skipped finished runs ($(round(t, digits = 1)) s)", all(last.(statuses) .== "ok") && t < 10)
end

# 8. Throughput at 1 km, 16 levels, Float64 (includes setup; the kernels are already compiled)
throughput = Dict{Float64, Float64}()
for (res, days) in ((2.0, 365), (1.0, 120), (0.5, 60), (0.25, 30))
    step("throughput $(res)°") do
        params = merge(base, (; name = "smoke_speed_$(_tag(res))deg", ocean_depth = 1000.0, n_depth = 16, resolution = res,
                                simulation_time = days * 1days, output_interval = nothing, checkpoint_interval = nothing))
        throughput[res] = @elapsed(simulation(params, nothing)) / 60 / (days / 365)
        check("$(res)° run finished", is_finished(params.name))
    end
end

println("\nA100 minutes per model year at 1 km, 16 levels, Float64, one run on the GPU:")
foreach(res -> @printf("  %5.2f°  %7.2f\n", res, throughput[res]), sort(collect(keys(throughput)), rev = true))
println("\n$(count(last.(results))) of $(length(results)) checks passed")
foreach(r -> println("  FAIL  ", r.first), filter(r -> !r.second, results))
