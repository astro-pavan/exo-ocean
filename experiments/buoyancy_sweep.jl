include("../src/ocean_sim.jl")
using Oceananigans.Units

T_day_default = 30
T_night_default = 0
T_day = [1, 3, 5, 10, 30, 50]
delta_T = [0.1, 0.3, 1, 3, 10, 30]
depth = 1 # km

# for T in T_day
#     ocean_simulation("buoyancy_T_$(T)_D_$(depth)", 0, depth * kilometer, R_Earth, T, T_night_default, 10*365days, n_lat=10, n_lon=30, n_depth=10, use_GPU=false)
# end

for T in T_day
    for dT in delta_T
        ocean_simulation("buoyancy_T_$(T)_dT_$(dT)_D_$(depth)", 0, depth * kilometer, R_Earth, T, T - dT, 10*365days, n_lat=10, n_lon=30, n_depth=10, use_GPU=false)
    end
end

depth = [0.1, 0.3, 1, 3, 10, 30]

# for d in depth
#     ocean_simulation("buoyancy_T_$(T_day_default)_D_$(d)", 0, d * kilometer, R_Earth, T_day_default, T_night, 10*365days, n_lat=10, n_lon=30, n_depth=10, use_GPU=false)
# end

