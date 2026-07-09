include("../src/atmosphere_sim.jl")

const days = 86400  # seconds


# 8 layers is stable, but at min_wind_speed=1 the surface bulk Richardson number
# (~98/V²) stayed ~10x above critical_Richardson=10, so the scheme diagnosed a
# zero-depth boundary layer and applied no mixing (bl_kh=9=nlayers+1 throughout).
# At 8 layers the lowest level sits ~544 m up, making Φ_surface (hence Ri) large;
# need V > ~3.1 m/s to get Ri below the cutoff. Raise the floor to 4 m/s -- high vs
# typical (0.5-1) but justified by the 544 m-thick surface layer here. Now that the
# tendency bug is fixed, this knob actually does something.
atmosphere_simulation("atm_test_hot", 10, 1e5, R_Earth, 400.0, 270.0, 2000*days,
                       n_lat=12, n_levels=8, use_GPU=false, min_wind_speed=4.0,
                       Δt_at_T31=Minute(1))
