include("../src/ocean_sim.jl")
using Oceananigans.Units

ocean_simulation("test_coriolis_6", 10, 1kilometer, R_Earth, 10.0, 0.0, 2*365days, n_lat=160, n_lon=360, n_depth=16, use_GPU=false, checkpoint_interval=365days)