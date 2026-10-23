import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.special import gamma, gammainc, hyp1f1

xi = np.pi / 4
chi = 1
cp_air = 1005.7
L_water = 2.501e6

R_vapour = 461.5

sigma = 5.670374419e-8
G = 6.6743e-11

# saturation specific humidity, constant-L Clausius-Clapeyron from the triple point (Pierrehumbert 2010, Ch. 2)
def q_sat(T, P):
    e_sat = 611.655 * np.exp(-(L_water / R_vapour) * (1 / T - 1 / 273.16))
    return 0.622 * e_sat / (P - e_sat)

class planet:

    bounds = None
    solver_options = {}

    def __init__(self, R_planet, M_planet, R_star, T_star, F_star):

        self.R_planet = R_planet
        self.g = G * M_planet / (R_planet ** 2)
        self.R_star = R_star
        self.T_star = T_star
        L_star = 4 * np.pi * R_star ** 2 * sigma * T_star**4
        self.S0 = F_star
        self.a = np.sqrt(L_star / (4 * np.pi * F_star))

    # build from an experiments/planet_parameters.planet_parameters object; extra arguments go to the model's __init__
    @classmethod
    def from_parameters(cls, params, *args, **kwargs):
        return cls(params.R_planet, params.M_planet, params.R_star, params.T_star, params.F_star, *args, **kwargs)

    def residuals(self, x):
        return np.array([0])

    def initial_guess(self):
        raise NotImplementedError(f"{type(self).__name__} has no initial_guess, pass x0 to solve")

    def solve(self, x0=None, bounds=None):

        if x0 is None:
            x0 = self.initial_guess()

        res = least_squares(self.residuals, x0, bounds=self.bounds if bounds is None else bounds, **self.solver_options)

        if res.success:
            return res.x

# from Wirth et al. (2026)
class dry_planet(planet):

    bounds = ((0, 0, 0, 0, 0), (np.inf, 1, 1, np.inf, np.inf)) # T3 > 1 allowed: ocean heating can warm the night surface above the air
    solver_options = dict(x_scale='jac', ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=20000) # residuals are tiny and badly scaled at low tau

    def __init__(self, R_planet, M_planet, R_star, T_star, F_star, alpha_wind, P_surface, J_ocean=0, c_ocean=1800, T_solidus=1700,
                 alpha_scaling=None, rotation_period=None):
        super().__init__(R_planet, M_planet, R_star, T_star, F_star)
        self.alpha = alpha_wind # ignored when alpha_scaling is set
        if alpha_scaling not in (None, 'WTG', 'coriolis'):
            raise ValueError("alpha_scaling must be None, 'WTG' or 'coriolis'")
        if alpha_scaling == 'coriolis' and rotation_period is None:
            raise ValueError("the coriolis scaling needs rotation_period (s)")
        self.alpha_scaling = alpha_scaling
        self.rotation_period = rotation_period # in s, equal to the orbital period when tidally locked
        self.P = P_surface # in Pa
        self.J_ocean = J_ocean # magma mass flux from day to night, kg/s
        self.c_ocean = c_ocean # molten silicate, Lai et al. (2024) Table 1
        self.T_solidus = T_solidus # Lai et al. (2024) Table 1

        self.kappa = 5e-5 # m^2/kg (assumes CO2 atmosphere)
        self.R_gas = 189
        self.cp = 849
        self.cv = 658
        self.kT = 8e-2

        self.A_planet = 0

        self.tau = 2 * self.kappa * self.P / self.g

        self.T_max = self.T_star * (0.5 - 0.5 * self.A_planet)**0.25 * (self.a / self.R_star) ** -0.5

        z, z0 = 10, 0.01
        self.CD = (0.4 / np.log(z / z0))**2

        self.CT = self.R_gas * self.T_max * self.kT / (self.R_planet * self.g)

        self.W0 = (self.kappa * self.R_gas * (1 - self.A_planet) * self.S0) / (chi * self.CD * self.cp * self.g)

        self.O = self.J_ocean * self.c_ocean / (2 * np.pi * self.R_planet**2 * sigma * self.T_max**3)

        self.V0 = np.sqrt(self.cp * self.R_gas * self.T_max / self.cv) / self.W0 # V = alpha * V0 (eq. 30)

        if alpha_scaling == 'coriolis':
            T_eq = self.T_star * np.sqrt(self.R_star / (2 * self.a)) * (1 - self.A_planet)**0.25 # eq. 2
            self.Lambda = np.sqrt(self.R_gas * T_eq) * self.rotation_period / (2 * np.pi * self.R_planet) # eq. 1

    # as planet.from_parameters, also taking the rotation period from params
    @classmethod
    def from_parameters(cls, params, *args, **kwargs):
        kwargs.setdefault('rotation_period', params.P_rotation)
        return super().from_parameters(params, *args, **kwargs)

    # longitudinal wind efficiency: fixed, or set by the day-night contrast (Wirth et al. 2026 Section 4.2)
    def alpha_at(self, T0, T1, T2):
        if self.alpha_scaling == 'WTG':
            return np.sqrt(np.maximum(1 - T2, 0)) # eq. 54
        if self.alpha_scaling == 'coriolis':
            return 2**0.25 * (self.cv / self.cp) * np.sqrt(T0 * T1) * self.Lambda * (1 - T2) # eq. 58
        return self.alpha

    # escaping fraction of surface emission, and day/night atmospheric emission to space per sigma T_a^4 (W26 thin limit)
    def column_emission(self):
        return max(1 - self.tau, 0), self.tau, self.tau

    # return-flow wind for the guess: |w| ~ |v|, which W26 eq. 46 nearly reduces to
    def wind_guess(self, T_d, x):
        return self.alpha_at(T_d, 1.0, x) * self.V0 * np.sqrt(T_d)

    # with surface = air on each side the column budgets and global balance reduce to one equation in x = T_a,n / T_a,d
    def initial_guess(self):
        t, U_d, U_n = self.column_emission()
        C = (xi * self.CT) / (chi * self.CD)
        T_day = lambda x: ((t + U_d) + (t + U_n) * x**4)**-0.25 # global balance

        def night_balance(x):
            T_d = T_day(x)
            Fa = self.tau * C * self.alpha_at(T_d, 1.0, x) * self.V0 * T_d**1.5 * (1 - x)
            Fo = self.O * np.maximum(T_d - np.maximum(T_d * x, self.T_solidus / self.T_max), 0)
            return (t + U_n) * (T_d * x)**4 - Fa - Fo

        x = brentq(night_balance, 1e-6, 1.0) if night_balance(1e-6) < 0 else 1e-6 # negative at 0, positive at 1
        T_d = T_day(x)
        return np.array([T_d, 0.99, x, 0.99, max(self.wind_guess(T_d, x), 1e-3)])

    def residuals(self, x):

        T0, T1, T2, T3, W = x

        C = (xi * self.CT) / (chi * self.CD)
        V = self.alpha_at(T0, T1, T2) * self.V0
        L = chi * self.CD * self.cp * self.R_planet * self.g / (self.R_gas**2 * self.T_max)

        # net day-night magma transport (Hall & Bryden 1982); melt returns at T_s,n, or at the solidus if the nightside is frozen
        Fo = self.O * np.maximum(T0 - np.maximum(T0 * T1 * T2 * T3, self.T_solidus / self.T_max), 0)

        r0 = self.tau * (T0**4 * T1**4 + W * (1 - T1**-1)) - Fo - (T0**4 -1) # eq. 32
        r1 = T2**-4 * T1**-4 * T0**-4 * (W * (1 - T3) + Fo / self.tau) - (T3**4/self.tau - 1) # eq.33
        r2 = T0**-4 * (C * V * (T0 * T1)**1.5 * (T2 - 1) + W * (T1**-1 - 1)) - (2 * T1**4 - 1) # eq. 34
        r3 = (T2 * T1 * T0)**-4 * (C * V * (T0 * T1)**1.5 * (1 - T2) + W * (T3 - 1)) - (2 - T3**4) # eq. 35
        r4 = L * T0**-1 * T1**-1 * (T0**4 - 2 * T0**4 * T1**4 + 2 * C * V * (T0 * T1)**1.5 * (T2 - 1) + 2 * W * (T1**-1 - 1)) - ((3 * T2 - 1) * (V * T0**0.5 * T1**0.5 - W)) # eq. 46

        return np.array([r0, r1, r2, r3, r4])

    def get_physical_properties(self, x):

        T0, T1, T2, T3, W = x

        T_surf_day = T0 * self.T_max
        T_atm_day = T1 * T_surf_day
        T_atm_night = T2 * T_atm_day
        T_surf_night = T3 * T_atm_night
        wind = W * self.W0

        return T_surf_day, T_atm_day, T_atm_night, T_surf_night, wind

# dry_planet with the grey adiabatic columns of Koll & Abbot (2016) for any optical depth, closed by their heat-engine wind (eq. 12)
# derivation in docs/derivations/thick_atmosphere_two_column.md, equation numbers from its Section 10
class thick_atmosphere_dry_planet(dry_planet):

    def __init__(self, R_planet, M_planet, R_star, T_star, F_star, alpha_wind, P_surface, J_ocean=0, c_ocean=1800, T_solidus=1700, eps_wind=0.5, beta_night=None,
                 alpha_scaling=None, rotation_period=None):
        super().__init__(R_planet, M_planet, R_star, T_star, F_star, alpha_wind, P_surface, J_ocean, c_ocean, T_solidus, alpha_scaling, rotation_period)
        self.eps = eps_wind # surface wind as a fraction of the ideal heat engine, GCM winds are ~1/2 of it (KA16 Fig. 6b)

        self.beta = self.R_gas / self.cp # dry adiabat in optical depth, n = 1 as kappa is pressure independent (KA16 eq. 7)
        self.beta_night = self.beta if beta_night is None else beta_night # 0 gives an isothermal nightside column

        self.t = np.exp(-self.tau) # transmission of surface emission to space
        self.U_day, self.D_day = self.emission(self.beta)
        self.U_night, self.D_night = self.emission(self.beta_night)

        self.E = self.R_gas * self.T_max * sigma * self.T_max**4 / (self.CD * self.P * self.W0**3)

    # fractions of sigma T_a^4 emitted to space (U) and to the surface (D) by an adiabatic grey column (KA16 eq. 8)
    def emission(self, beta):
        a = 1 + 4 * beta
        U = self.tau**(1 - a) * gammainc(a, self.tau) * gamma(a)
        D = self.tau / a * hyp1f1(1, 1 + a, -self.tau)
        return U, D

    def column_emission(self):
        return self.t, self.U_day, self.U_night

    # heat-engine wind (eq. 5) with surface = air
    def wind_guess(self, T_d, x):
        eta = max(1 - 2**-0.25 / T_d, 0)
        return (self.eps**3 * eta * (1 - self.t * T_d**4) * self.E * T_d)**(1 / 3)

    def residuals(self, x):

        T0, T1, T2, T3, W = x

        C = (xi * self.CT) / (chi * self.CD)
        V = self.alpha_at(T0, T1, T2) * self.V0

        T_sd, T_ad, T_an, T_sn = T0, T0 * T1, T0 * T1 * T2, T0 * T1 * T2 * T3 # in units of T_max

        Fsd = self.tau * W * (1 - T1**-1)
        Fsn = self.tau * W * (1 - T3)
        Fa = self.tau * C * V * T_ad**1.5 * (1 - T2)
        Fo = self.O * np.maximum(T0 - np.maximum(T_sn, self.T_solidus / self.T_max), 0) # as in dry_planet

        Q = 1 - Fo - self.t * T_sd**4 # heat entering the dayside atmosphere from the surface
        eta = np.maximum(1 - 2**-0.25 / T0, 0) # Carnot efficiency between T_s,d and T_eq

        r0 = 1 + self.D_day * T_ad**4 + Fsd - Fo - T_sd**4 # eq. 1
        r1 = (1 - self.t) * T_sd**4 - (self.U_day + self.D_day) * T_ad**4 - Fsd - Fa # eq. 2
        r2 = (1 - self.t) * T_sn**4 - (self.U_night + self.D_night) * T_an**4 - Fsn + Fa # eq. 3
        r3 = self.D_night * T_an**4 + Fsn + Fo - T_sn**4 # eq. 4
        r4 = W**3 - self.eps**3 * eta * Q * self.E * T_ad # eq. 5

        return np.array([r0, r1, r2, r3, r4])

# from Yang & Abbot (2014); the hooks before residuals are overridden by runaway_wet_planet
class wet_planet(planet):

    bounds = ((100, 100, 100, 0, 0), (400, 400, 400, 2000, 5000))

    def __init__(self, R_planet, M_planet, R_star, T_star, F_star, F_ocean, P_surface=1e5):
        super().__init__(R_planet, M_planet, R_star, T_star, F_star)
        self.Fo = F_ocean
        self.P = P_surface # in Pa, for the boundary layer humidity

        self.Tc = 230 # eq. 8

        self.A_planet = 0.09 # seawater albedo, eq. 14

        # Table 1
        self.Pa = 60000
        self.k1 = 0.2
        self.k2 = 1000
        self.k3 = 0.08
        self.RH1 = 0.9
        self.RH2 = 0.8
        self.RH3 = 0.3

        self.Za = 5000 * np.log(self.P / self.Pa) # height of Pa for a 5000 m scale height (Section 3)

    def humidity(self, RH, T, P):
        return RH * q_sat(T, P)

    def emissivity(self, RH, T):
        return 1 - np.exp(- self.k2 * self.humidity(RH, T, self.Pa)) # eq. 10

    def cloud_fraction(self, Fc):
        return np.minimum(self.k3 * np.log(Fc + 1.0), 1) # eq. 13

    def OLR_clear(self, Ta, eps):
        return eps * sigma * Ta**4

    def F_str(self, Ta):
        return 0

    def convection(self, MSE_s, MSE_a, Fc):
        return MSE_s - MSE_a # eq. 7

    def residuals(self, x):

        T1, T2, T4, Fa, Fc = x

        T3 = T2 # eq. 6

        Fd = self.k1 * Fa # eq. 9

        e2 = self.emissivity(self.RH2, T2)
        e3 = self.emissivity(self.RH3, T3)

        fc = self.cloud_fraction(Fc)
        A = self.A_planet + fc - self.A_planet * fc # eq. 14

        r0 = 0.5 * self.S0 * (1 - A) - Fc - self.Fo + (1 - fc) * e2 * sigma * T2**4 + fc * sigma * self.Tc**4 - sigma * T1**4 # eq.2
        r1 = Fc - Fa + (1 - fc) * e2 * sigma * T1**4 + fc * sigma * T1**4 - (1 - fc) * (e2 * sigma * T2**4 + self.OLR_clear(T2, e2)) - 2 * fc * sigma * self.Tc**4 # eq. 3
        r2 = Fa - Fd + e3 * sigma * T4**4 - e3 * sigma * T3**4 - self.OLR_clear(T3, e3) + self.F_str(T3) # eq.4
        r3 = self.Fo + Fd + e3 * sigma * T3**4 - sigma * T4**4 # eq. 5

        MSE_s = cp_air * T1 + L_water * self.humidity(self.RH1, T1, self.P) # eq. 11
        MSE_a = cp_air * T2 + L_water * self.humidity(1, T2, self.Pa) + self.g * self.Za # eq. 12
        r4 = self.convection(MSE_s, MSE_a, Fc)

        return np.array([r0, r1, r2, r3, r4])

    # continuation in stellar flux (rising S0 from a cold guess, falling from a hot one); None where no equilibrium exists
    def solve_branch(self, S0_values, x0, tol=1e-6):
        solutions = []
        for S0 in S0_values:
            self.S0 = S0
            x = None
            for start in [np.asarray(x0) * f for f in (1, 1.001, 1.02, 0.98)]: # nudged retries across the kinks in OLR_clear and F_str
                res = least_squares(self.residuals, start, bounds=self.bounds, max_nfev=2000)
                if np.abs(res.fun).max() < tol:
                    x = x0 = res.x
                    break
            solutions.append(x)
        return solutions

# from Fan et al. (2024): wet_planet with a runaway OLR, nightside cloud, stratified dayside and non-dilute vapour (their eqs. 3-8)
class runaway_wet_planet(wet_planet):

    bounds = ((100, 100, 100, 0, 0), (1500, 1500, 1500, 1e4, 1e5))

    def __init__(self, R_planet, M_planet, R_star, T_star, F_star, F_ocean=0, P_surface=1e5,
                 runaway_olr=True, night_cloud=True, fc_fixed=None):
        super().__init__(R_planet, M_planet, R_star, T_star, F_star, F_ocean, P_surface)

        self.runaway_olr = runaway_olr
        self.night_cloud = night_cloud
        self.fc_fixed = fc_fixed # fixed dayside cloud fraction disables the cloud feedback (Fig. 2b)

        # eq. 1
        self.T_emis = 280
        self.T_end = 700
        self.C_olr = 3

        self.F_NC = 200 # eq. 2

        self.e_max = 1e6 # vapour pressure cap (Section 2)

    def e_vap(self, T):
        return np.minimum(611.655 * np.exp(-(L_water / R_vapour) * (1 / T - 1 / 273.16)), self.e_max)

    # mass ratio of vapour to dry air, non-dilute with P_a = P_c + e
    def humidity(self, RH, T, P):
        return 0.622 * RH * self.e_vap(T) / P

    def emissivity(self, RH, T):
        return 1 - np.exp(- self.k2 * (self.Pa + RH * self.e_vap(T)) / self.Pa * self.humidity(RH, T, self.Pa))

    def cloud_fraction(self, Fc):
        return self.fc_fixed if self.fc_fixed is not None else super().cloud_fraction(Fc)

    # eq. 1, with the plateau added to the last branch so it is continuous at T_end (see Fig. 3b)
    def OLR_clear(self, Ta, eps):
        if not self.runaway_olr or Ta <= self.T_emis:
            return super().OLR_clear(Ta, eps)
        return eps * sigma * self.T_emis**4 + self.C_olr * max(Ta - self.T_end, 0)

    # eq. 2, slopes corrected so the printed plateau values are joined continuously
    def F_str(self, Ta):
        if not self.night_cloud:
            return 0
        return np.interp(Ta, [320, 400, 700, 800], [0, 7 / 11, 7 / 11, 1]) * self.F_NC

    # eq. 8 as a Fischer-Burmeister complementarity: Fc >= 0, MSE_s <= MSE*_a, one of them tight
    def convection(self, MSE_s, MSE_a, Fc):
        a, b = Fc / 100, (MSE_a - MSE_s) / (10 * cp_air)
        return a + b - np.sqrt(a**2 + b**2)
