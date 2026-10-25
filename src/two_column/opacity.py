import exo_k as xk
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

N_A = 6.022e23

datapath = 'src/two_column/opacity_data/'

xk.settings.Settings().set_mks(True)
xk.settings.Settings().add_search_path(datapath + 'cia', path_type = 'cia')

k_data = xk.kdatabase.Kdatabase(
    {
        'H2O' : datapath + 'H2O/1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5',
        'CO2' : datapath + 'CO2/12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
        'CO'  : datapath + 'CO/12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    }   
)

mols = ['He','H2','H2O', 'CO2']
cia_data = xk.ciadatabase.CIAdatabase(molecules=mols)
# use_grid_filter: zero (then floored by remove_zeros) outside each table's range, rather than its edge value held across
# the whole k-table grid
cia_data.sample(k_data.wns, use_grid_filter=True, remove_zeros=True)


# Absorption-only flux optical depth -ln<exp(-D tau)> of an isothermal column, per band between consecutive wavelength_bins
# (um). D is the diffusivity factor: 1.66 for diffuse (thermal) flux, 2 to match W26's tau = 2 kappa p_s / g, 1 for a vertical beam.
def get_opacity(P: float, T: float, g: float, compostion: dict[str, float], wavelength_bins: list[float], diffusivity: float = 1.66, rayleigh=True):

    logp_array = np.linspace(0, np.log10(P), num=40) # 10 levels gave 5% errors for 10 bar of CO2, 40 give < 0.5%
    t_array = np.full_like(logp_array, T)

    gas = xk.gas_mix.Gas_mix(compostion, k_database=k_data, cia_database=cia_data)
    sigma = gas.cross_section(logp_array=logp_array, t_array=t_array, rayleigh=rayleigh)
    molecule_mass = gas.molar_mass() / N_A
    tau = diffusivity * np.trapezoid(sigma, 10 ** logp_array, axis=0) / (molecule_mass * g)
    # log of the g-integrated transmission per spectral bin; in log space so opaque bands (tau > ~700) stay finite
    log_transmission = logsumexp(-tau, b=k_data.weights, axis=1)

    tau_bands = []
    for lam_a, lam_b in zip(wavelength_bins[:-1], wavelength_bins[1:]):
        nu_left, nu_right = sorted(1e4 / np.array([lam_a, lam_b], dtype=float))    # cm^-1
        # width of each k-table bin inside the band (bins cut by the band edges count partially)
        overlap = np.clip(np.minimum(gas.wnedges[1:], nu_right) - np.maximum(gas.wnedges[:-1], nu_left), 0, None)
        if overlap.sum() < 0.99 * (nu_right - nu_left):
            raise ValueError(f"band {lam_a}-{lam_b} um extends beyond the tables "
                             f"({1e4 / gas.wnedges[-1]:.2f}-{1e4 / gas.wnedges[0]:.2f} um)")
        tau_bands.append(-logsumexp(log_transmission, b=overlap / overlap.sum()))

    return np.array(tau_bands)

if __name__ == '__main__':

    taus = get_opacity(1e5, 300, 9.8, {'CO2': 0.01, 'H2O': 0.01}, [1, 10, 20, 50])
    print(taus.shape)
    print(taus)

