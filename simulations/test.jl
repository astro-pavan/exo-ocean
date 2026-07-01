include("ocean_sim.jl")
include("constants.jl")
using Oceananigans.Units

const solar_constant = 1361.0  # W/m²

ocean_simulation("test_coriolis_3", 3, 1kilometer, R_Earth, solar_constant, 10*365days, n_lat=40, n_lon=90, n_depth=20, use_GPU=false)

