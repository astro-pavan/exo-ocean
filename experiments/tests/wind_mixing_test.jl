# Does weak vertical momentum mixing cause the ~7 m/s equatorial jet of wind1km_shared_wind2bar_1deg? Three 3-year, 2° runs of the
# 2 bar emulator-wind case, warm-started from base1km_spinup (same grid, no regrid), differing only in the vertical closure:
#   control: ν = κ = 1e-4 m²/s plus convective adjustment (the default before CATKE)
#   nu1e-2:  ν = 1e-2 m²/s, about twice the wind-mixed maximum of Pacanowski & Philander (1981, JPO 11, 1443), κ unchanged
#   catke:   CATKE (Wagner et al. 2025, JAMES) on top of the ν = κ = 1e-4 background, replacing convective adjustment
#   nowind_control, nowind_catke: control and catke without wind, to isolate how much the wind speeds up the jet
# The jet saturated within ~2 years in the 1° run, so 3 years is enough to compare.
#   nice -n 19 julia --project=. -t 6 experiments/tests/wind_mixing_test.jl control|nu1e-2|catke|nowind_control|nowind_catke
include(joinpath(@__DIR__, "..", "..", "src", "ocean", "simulation_sweep.jl"))

case = only(ARGS)
case in ("control", "nu1e-2", "catke", "nowind_control", "nowind_catke") || error("unknown case $(case)")
wind = !startswith(case, "nowind_")

if case == "nu1e-2"    # implicit, as explicit ν = 1e-2 is unstable in the 11 m top cell once Δt > ~1 h; replaces ocean_closure here only
    function ocean_closure(; κ_horizontal, κ_vertical = 1e-4, prandtl, vertical_mixing)
        horizontal = iszero(κ_horizontal) ? nothing : HorizontalScalarDiffusivity(κ = κ_horizontal)
        closures = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = prandtl * κ_vertical, κ = κ_vertical), horizontal,
                    ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0))
        return Tuple(c for c in closures if !isnothing(c))
    end
end

use_GPU = get(ENV, "USE_GPU", "false") == "true"
ocean_simulation(wind ? "mixtest_2bar_$(case)" : "mixtest_$(case)", 10, 1000.0, R_Earth, 30.0, 0.0, 3 * 365days;
                 resolution = 2.0, n_depth = 16, float_type = Float64, use_GPU, emulator_wind = wind, surface_pressure = wind ? 2 : nothing,
                 prandtl = case == "nu1e-2" ? 100 : 1, vertical_mixing = endswith(case, "catke") ? :catke : :convective_adjustment,
                 warm_start = "base1km_spinup")
