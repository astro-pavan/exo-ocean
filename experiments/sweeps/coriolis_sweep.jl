include(joinpath(@__DIR__, "..", "..", "src", "ocean", "ocean_sim.jl"))
using Oceananigans.Units

T_night_default = 0
delta_T = [0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50]
rotation_period = [3, 5, 10, 30, 50]

depth = 1 # km

for P in rotation_period
    for dT in delta_T
        ocean_simulation("coriolis_P_$(P)_dT_$(dT)_D_$(depth)", P, depth * kilometer, R_Earth, T_night_default + dT, T_night_default, 1000days, n_lat=160, n_lon=360, n_depth=16, use_GPU=false, checkpoint_interval=356days)
    end
end
