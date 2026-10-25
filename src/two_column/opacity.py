import exo_k as xk
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

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
cia_data.sample(k_data.wns)


def get_opacity(P, T, compostion, wavelength_bins):

    gas = xk.gas_mix.Gas_mix(compostion, k_database=k_data, cia_database=cia_data)
    gas_k_data = gas.cross_section(logp_array=np.log10(P), t_array=T)
    mmw = gas.molar_mass()

