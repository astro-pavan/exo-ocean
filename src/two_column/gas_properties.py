import numpy as np

R_universal = 8.314462618 # J/mol/K

# molar masses, g/mol
molar_mass = {'CO2': 44.0095, 'H2O': 18.01528, 'CO': 28.0101, 'N2': 28.0134, 'O2': 31.9988, 'H2': 2.01588, 'He': 4.002602}

# Shomate coefficients (A, B, C, D, E) and upper temperature of each range, NIST-JANAF (Chase 1998) via the NIST WebBook;
# the lowest range is extrapolated below its lower limit (H2O 500 K, CO2 and CO 298 K)
shomate = {
    'CO2': [(1200, (24.99735, 55.18696, -33.69137, 7.948387, -0.136638)),
            (6000, (58.16639, 2.720074, -0.492289, 0.038844, -6.447293))],
    'H2O': [(1700, (30.09200, 6.832514, 6.793435, -2.534480, 0.082139)),
            (6000, (41.96426, 8.622053, -1.499780, 0.098119, -11.15764))],
    'CO':  [(1300, (25.56759, 6.096130, 4.054656, -2.671301, 0.131021)),
            (6000, (35.15070, 1.300095, -0.205921, 0.013550, -3.282780))],
    'N2':  [(500, (28.98641, 1.853978, -9.647459, 16.63537, 0.000117)),
            (2000, (19.50583, 19.88705, -8.598535, 1.369784, 0.527601)),
            (6000, (35.51872, 1.128728, -0.196103, 0.014662, -4.553760))],
    'O2':  [(700, (31.32234, -20.23531, 57.86644, -36.50624, -0.007374)),
            (2000, (30.03235, 8.772972, -3.988133, 0.788313, -0.741599)),
            (6000, (20.91111, 10.72071, -2.020498, 0.146449, 9.245722))],
    'H2':  [(1000, (33.066178, -11.363417, 11.432816, -2.772874, -0.158558)),
            (2500, (18.563083, 12.257357, -2.859786, 0.268238, 1.977990)),
            (6000, (43.413560, -4.293079, 1.272428, -0.096876, -20.533862))],
    'He':  [(6000, (20.78603, 0.0, 0.0, 0.0, 0.0))],
}


# molar isobaric heat capacity of one ideal-gas species, J/mol/K
def molar_cp(species, T):
    if T > 6000:
        raise ValueError(f'T = {T} K is above the 6000 K range of the Shomate fits')
    for T_max, (A, B, C, D, E) in shomate[species]:
        if T <= T_max:
            t = T / 1000
            return A + B * t + C * t**2 + D * t**3 + E / t**2


# specific gas constant, c_p and c_v (J/kg/K) of an ideal-gas mixture with mole fractions composition, at temperature T
def gas_properties(composition, T):
    unknown = set(composition) - set(shomate)
    if unknown:
        raise ValueError(f'no heat capacity data for {sorted(unknown)}; known species are {sorted(shomate)}')
    total = sum(composition.values())
    if abs(total - 1) > 1e-3:
        raise ValueError(f'mole fractions sum to {total}, not 1')
    M = sum(x * molar_mass[s] for s, x in composition.items()) * 1e-3 # kg/mol
    cp_molar = sum(x * molar_cp(s, T) for s, x in composition.items())
    R = R_universal / M
    c_p = cp_molar / M
    return R, c_p, c_p - R
