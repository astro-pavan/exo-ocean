include("atmosphere_sim.jl")

const days = 86400  # seconds

atmosphere_simulation("atm_test", 0, 1e5, R_Earth, 300.0, 270.0, 10*365*days,
                      n_lat=20, n_levels=8, use_GPU=false)
