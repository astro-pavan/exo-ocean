import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.special import gamma, gammainc, hyp1f1

from radiation import planck_function_integrated, c
from gas_properties import gas_properties

sigma = 5.670374419e-8
G = 6.6743e-11

R_vapour = 461.5

# Gauss-Legendre nodes on [0, 1] in the absorptance coordinate; 24 nodes match adaptive quadrature to < 0.1% for any tau
_y, _w = np.polynomial.legendre.leggauss(24)
_y, _w = 0.5 * (_y + 1), 0.5 * _w

class planet:

    # tau: scalar (grey), or per-band optical depths for wavelength_bins; None computes them from composition with opacity.py
    def __init__(self, R_planet, M_planet, R_star, T_star, F_star, tau, albedo, P_surface, J_ocean,
                 wavelength_bins=None, composition=None, T_opacity=None, diffusivity=1.66, heat_transport='overturning', alpha_wind=0.1,
                 T_solidus=1700, T_thermo=None):
    
        self.R_planet = R_planet
        self.g = G * M_planet / (R_planet ** 2)
        self.R_star = R_star
        self.T_star = T_star
        L_star = 4 * np.pi * R_star ** 2 * sigma * T_star**4
        self.S0 = F_star
        self.a = np.sqrt(L_star / (4 * np.pi * F_star))
        self.albedo = albedo
        self.P_surf = P_surface

        self.R_gas = 189
        self.c_p = 849
        self.c_v = 658

        self.c_ocean = 1800

        z, z0 = 10, 0.01
        self.C_D = (0.4 / np.log(z / z0))**2

        self.engine_eff = 0.5

        self.J_ocean = J_ocean

        self.T_eq = T_star * np.sqrt(R_star / (2 * self.a)) * (1 - albedo)**0.25

        # ideal-gas mixture properties at one reference temperature (default T_opacity, else T_eq); CO2 at ~300 K without composition
        if composition is not None:
            self.T_thermo = T_thermo if T_thermo is not None else (T_opacity if T_opacity is not None else self.T_eq)
            self.R_gas, self.c_p, self.c_v = gas_properties(composition, self.T_thermo)

        self.beta = self.R_gas / self.c_p # dry adiabat in optical depth, n = 1 as kappa is pressure independent (KA16 eq. 7)
        self.set_bands(tau, wavelength_bins, composition, T_opacity, diffusivity)

        self.nightside_sensible_flux = True

        # 'overturning': surface wind U carries the day-night contrast; 'W26': Wirth et al. (2026) eq. 12 with |v| = alpha c_s
        if heat_transport not in ('overturning', 'W26'):
            raise ValueError("heat_transport must be 'overturning' or 'W26'")
        
        self.heat_transport = heat_transport
        self.alpha = alpha_wind # longitudinal wind efficiency, W26 eq. 30
        self.xi = np.pi / 4 # atmospheric integration factor, W26 Appendix A
        self.k_T = 8e-2 # turbulent diffusivity coefficient, W26 Table 1

        self.T_solidus = T_solidus
        self.T_liquidus = 2000

    def rho(self, T):
        return self.P_surf / (self.R_gas * T) 

    # band edges in frequency (Hz) and band optical depths; the outermost bands are extended to 0 and infinity with the
    # optical depth of the edge band so that the Planck fractions sum to 1
    def set_bands(self, tau, wavelength_bins, composition, T_opacity, diffusivity):
        if wavelength_bins is None:
            self.nu_edges = np.array([0.0, np.inf])
            self.tau_bands = np.atleast_1d(np.asarray(tau, dtype=float))
            return
        if tau is None:
            from opacity import get_opacity # needs exo_k and the downloaded tables
            T_opacity = self.T_eq if T_opacity is None else T_opacity
            tau = get_opacity(self.P_surf, T_opacity, self.g, composition, wavelength_bins, diffusivity=diffusivity)
        tau = np.asarray(tau, dtype=float)
        nu = np.sort(c / (np.asarray(wavelength_bins, dtype=float) * 1e-6))
        tau = tau[::-1] if wavelength_bins[0] < wavelength_bins[-1] else tau # match ascending frequency
        self.nu_edges = np.concatenate(([0.0], nu, [np.inf]))
        self.tau_bands = np.concatenate(([tau[0]], tau, [tau[-1]]))

    # surface emission per band, sigma T^4 f_B(nu_i, nu_i+1, T)
    def band_emission(self, T):
        return planck_function_integrated(T, self.nu_edges[:-1], self.nu_edges[1:])

    # surface emission escaping to space (G_s) and absorbed by the atmosphere (G_a)
    def ground_radiation(self, Tg):

        B = self.band_emission(Tg)
        t = np.exp(-self.tau_bands)

        return np.sum(t * B), np.sum((1 - t) * B)

    # emission of an adiabatic column T = T_a (tau'/tau_i)^beta to space (A_s) and to the ground (A_g), integrated in the
    # absorptance coordinate y = (1 - e^-t) / (1 - e^-tau_i), t measured from the top (A_s) or the bottom (A_g)
    def atmosphere_radiation(self, Ta):

        tau = np.maximum(self.tau_bands, 1e-12)[:, None]
        a = -np.expm1(-tau)
        t = -np.log1p(-_y * a)
        s_up = np.clip(t / tau, 0, 1)

        B = lambda s: planck_function_integrated(Ta * s**self.beta, self.nu_edges[:-1, None], self.nu_edges[1:, None])

        As = np.sum(a[:, 0] * np.sum(_w * B(s_up), axis=1))
        Ag = np.sum(a[:, 0] * np.sum(_w * B(1 - s_up), axis=1))

        return As, Ag

    # day-to-night atmospheric heat flux per unit dayside area
    def atmosphere_transport(self, Ta1, Ta2, U):

        if self.heat_transport == 'W26':
            v = self.alpha * np.sqrt(self.c_p * self.R_gas * Ta1 / self.c_v) # bulk wind, alpha times the dayside sound speed
            return self.xi * self.k_T * self.c_p * self.P_surf * v * (Ta1 - Ta2) / (self.g * self.R_planet) # W26 eq. 12, rho_d R T_a,d = p_s
        
        return self.P_surf * U * self.c_v * (Ta1 - Ta2) / (2 * self.g * self.R_planet)

    def energy_residuals(self, x):

        Tg1, Ta1, Tg2, Ta2, U = x

        # stellar flux
        I = 0.5 * (1 - self.albedo) * self.S0

        # carnot engine efficency
        eta = 1 - (self.T_eq / Tg1)

        # sensible heat flux
        S1 = self.C_D * self.c_p * self.rho(Ta1) * (Tg1 - Ta1) * U
        S2 = self.C_D * self.c_p * self.rho(Ta2) * (Tg2 - Ta2) * U if self.nightside_sensible_flux else 0

        # ground radiation
        Gs1, Ga1 = self.ground_radiation(Tg1)
        Gs2, Ga2 = self.ground_radiation(Tg2)

        # atmosphere radiation
        As1, Ag1 = self.atmosphere_radiation(Ta1)
        As2, Ag2 = self.atmosphere_radiation(Ta2)

        # atmosphere heat flux
        W = self.atmosphere_transport(Ta1, Ta2, U)

        # ocean heat flux
        nightside_melt_fraction = np.clip((Tg2 - self.T_solidus) / (self.T_liquidus - self.T_solidus), 0, 1)
        O = self.J_ocean * self.c_ocean * (Tg1 - Tg2) * nightside_melt_fraction / (2 * np.pi * self.R_planet ** 2)

        # dayside ground
        Fg1 = I - S1 - Ga1 + Ag1 - Gs1 - O

        # dayside atmosphere
        Fa1 = S1 + Ga1 - Ag1 - As1 - W

        # nightside ground
        Fg2 = - S2 - Ga2 + Ag2 - Gs2 + O

        # nightside atmosphere
        Fa2 = S2 + Ga2 - Ag2 - As2 + W

        # atmospheric heat engine
        Feng = self.engine_eff * eta * (Ga1 - Ag1 + S1) - self.C_D * self.rho(Tg1) * U ** 3

        return np.array([Fg1, Fa1, Fg2, Fa2, Feng])

    # energy residuals relative to the stellar flux, engine residual relative to the dissipation
    def scaled_residuals(self, x):
        r = self.energy_residuals(x)
        I = 0.5 * (1 - self.albedo) * self.S0
        return np.concatenate((r[:4] / I, [r[4] / (self.C_D * self.rho(x[0]) * max(x[4], 1e-3)**3)]))

    # dayside at the airless substellar-mean temperature, nightside at T_eq, air slightly colder than the dayside ground
    def initial_guess(self):
        T_day = (0.5 * (1 - self.albedo) * self.S0 / sigma)**0.25
        return np.array([T_day, 0.9 * T_day, 0.7 * self.T_eq, 0.8 * self.T_eq, 5.0])

    # x = (Tg1, Ta1, Tg2, Ta2, U); None if the scaled residuals do not fall below tol
    def solve(self, x0=None, tol=1e-8):
        x0 = self.initial_guess() if x0 is None else np.asarray(x0, dtype=float)
        res = least_squares(self.scaled_residuals, x0, bounds=([1, 1, 1, 1, 0], np.inf), x_scale='jac',
                            ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=5000)
        return res.x if np.abs(self.scaled_residuals(res.x)).max() < tol else None