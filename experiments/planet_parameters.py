import numpy as np

R_Earth = 6371000
M_Earth = 5.9722e24
R_sun = 6.957e8 # IAU nominal
M_sun = 1.98847e30
AU = 149597870700
day = 86400
sigma = 5.670374419e-8
G = 6.6743e-11

# SI units throughout; periods in s
class planet_parameters:

    def __init__(self, Rp, Mp, Rs, Ms, Ts, Po, Pr=None):

        self.R_planet = Rp
        self.M_planet = Mp
        self.R_star = Rs
        self.M_star = Ms
        self.T_star = Ts
        self.P_orbit = Po
        self.P_rotation = Pr if Pr is not None else Po # tidally locked by default

        self.semi_major_axis = (G * (Ms + Mp) * Po**2 / (4 * np.pi**2))**(1 / 3) # Kepler's third law

        self.F_star = sigma * Ts**4 * (Rs / self.semi_major_axis)**2 # stellar flux at the planet

# 55 Cancri e: radius from Zhao et al. (2023), mass and star from Bourrier et al. (2018)
cnc_55_e = planet_parameters(1.856 * R_Earth, 7.99 * M_Earth, 0.943 * R_sun, 0.905 * M_sun, 5172, 0.7365474 * day)

# K2-141 b: Bonomo et al. (2023)
k2_141_b = planet_parameters(1.51 * R_Earth, 4.97 * M_Earth, 0.681 * R_sun, 0.708 * M_sun, 4570, 0.2803244 * day)

# TOI-561 b: planet and star from Piotto et al. (2024), stellar T_eff from Lacedelli et al. (2022)
toi_561_b = planet_parameters(1.397 * R_Earth, 2.02 * M_Earth, 0.843 * R_sun, 0.806 * M_sun, 5372, 0.4465697 * day)

# TOI-2431 b: Taş et al. (2026)
toi_2431_b = planet_parameters(1.534 * R_Earth, 6.2 * M_Earth, 0.651 * R_sun, 0.661 * M_sun, 4109, 0.22419578 * day)

# LHS 1140 b: Cadieux et al. (2024)
lhs_1140_b = planet_parameters(1.73 * R_Earth, 5.6 * M_Earth, 0.2159 * R_sun, 0.1844 * M_sun, 3096, 24.73723 * day)

# Proxima Centauri b: Suárez Mascareño et al. (2025); non-transiting, so M sin i only and the radius is estimated
M_proxima_b = 1.055 # M sin i in M_Earth, a lower bound on the true mass
R_proxima_b = 1.008 * M_proxima_b**0.279 # rocky mass-radius relation of Chen & Kipping (2017)
proxima_b = planet_parameters(R_proxima_b * R_Earth, M_proxima_b * M_Earth, 0.141 * R_sun, 0.1221 * M_sun, 2900, 11.18465 * day)

# TRAPPIST-1 e: Agol et al. (2021)
trappist_1_e = planet_parameters(0.92 * R_Earth, 0.692 * M_Earth, 0.1192 * R_sun, 0.0898 * M_sun, 2566, 6.101013 * day)

# TRAPPIST-1 b: Agol et al. (2021)
trappist_1_b = planet_parameters(1.116 * R_Earth, 1.374 * M_Earth, 0.1192 * R_sun, 0.0898 * M_sun, 2566, 1.510826 * day)

# LHS 3844 b: Nagel et al. (2026)
lhs_3844_b = planet_parameters(1.286 * R_Earth, 2.37 * M_Earth, 0.189 * R_sun, 0.151 * M_sun, 3080, 0.462929709 * day)
