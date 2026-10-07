const omega_Earth = 7.29e-5 # rad/s
const R_Earth = 6371000 # m
const solar_constant = 1361 # W/m^2
const g_Earth = 9.81 # m/s^2

# Ocean
const rho_seawater = 1026 # kg/m^3
const cp_seawater = 3994.0 # J/(kg·K)
const C_Bottom_Drag = 1e-2
const C_D_wind = 0.002

# Atmosphere
const rho_air = 1.2 # kg/m^3
const cp_air = 1004.0 # J/(kg·K)
const R_specific_air = 287.0 # J/(kg·K), dry air
const C_surface_drag = 1e-3