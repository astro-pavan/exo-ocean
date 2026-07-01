include("atmosphere_sim.jl")

const days = 86400  # seconds

atmosphere_simulation("atm_coriolis_4", 30, 1e5, R_Earth, 300.0, 270.0, 10000*days, n_lat=30, n_levels=16, use_GPU=false)
