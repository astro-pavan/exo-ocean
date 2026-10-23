# Reproduces Wirth, Powell & Wordsworth (2026) Figs. 3 and 7 with dry_planet (their model) and thick_atmosphere_dry_planet
import os, sys, itertools, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'src', 'two_column'))
import planet_parameters as pp
import two_column as tc

warnings.filterwarnings('ignore')

# Earth instellation around an M dwarf (W26 Section 4.1); the period follows from F via Kepler's third law
def earth_like(F=1361, Ts=3000, Rs=0.3 * pp.R_sun, Ms=0.3 * pp.M_sun):
    a = Rs * np.sqrt(pp.sigma * Ts**4 / F)
    P = 2 * np.pi * np.sqrt(a**3 / (pp.G * (Ms + pp.M_Earth)))
    return pp.planet_parameters(pp.R_Earth, pp.M_Earth, Rs, Ms, Ts, P)

# LHS 3844 b as used by W26: Vanderspek et al. (2019), whose estimated mass is 2.25 M_Earth (no measured mass)
lhs_3844_b_v19 = pp.planet_parameters(1.303 * pp.R_Earth, 2.25 * pp.M_Earth, 0.189 * pp.R_sun, 0.151 * pp.M_sun, 3036, 0.46292913 * pp.day)

planets = {'Earth-like': earth_like(), 'TRAPPIST-1 b': pp.trappist_1_b, 'LHS 3844 b': lhs_3844_b_v19}
models = {'dry_planet (W26)': tc.dry_planet, 'thick_atmosphere_dry_planet': tc.thick_atmosphere_dry_planet}
alphas = [0.05, 0.1, 0.3, 0.5, 1.0]
pressures = np.geomspace(1e2, 1e6, 33) # 1 mbar to 10 bar, in Pa

colours = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'] # one-hue ordinal ramp, light to dark with alpha
ink, muted, grid = '#0b0b0b', '#6b6a66', '#e4e3df'

# each point solved independently from the model's own initial guess; NaN where it does not converge
def sweep(model, params, alpha, ps, tol=1e-5):
    out = []
    for P in ps:
        p = model.from_parameters(params, alpha, P)
        x = p.solve()
        converged = x is not None and np.abs(p.residuals(x)).max() < tol
        out.append(p.get_physical_properties(x) if converged else (np.nan,) * 5)
    return np.array(out)

def tau_one_pressure(params, model):
    p = model.from_parameters(params, 0.1, 1e5)
    return p.g / (2 * p.kappa) # tau = 2 kappa p / g = 1

results = {}
for (mname, model), (pname, params) in itertools.product(models.items(), planets.items()):
    p_tau1 = tau_one_pressure(params, model)
    ps = pressures if model is tc.thick_atmosphere_dry_planet else pressures[pressures <= p_tau1] # W26 stop at tau = 1
    for alpha in alphas:
        results[mname, pname, alpha] = (ps, sweep(model, params, alpha, ps))
    print(f'{mname:28s} {pname:12s} done')

def plot(index, ylabel, title, filename, airless_line):
    fig, axes = plt.subplots(len(models), len(planets), figsize=(12, 7), sharex=True, sharey='col', constrained_layout=True)
    fig.patch.set_facecolor('#fcfcfb')
    for i, mname in enumerate(models):
        for j, (pname, params) in enumerate(planets.items()):
            ax = axes[i, j]
            ax.set_facecolor('#fcfcfb')
            model = models[mname]
            for alpha, c in zip(alphas, colours):
                ps, T = results[mname, pname, alpha]
                ax.plot(ps / 1e5, T[:, index], color=c, lw=2, label=f'$v = {alpha}\\,c_s$')
            p0 = model.from_parameters(params, 0.1, 1e5)
            T_eq = params.T_star * np.sqrt(params.R_star / (2 * params.semi_major_axis))
            ax.axhline(T_eq, color=muted, lw=1, ls='-')
            if airless_line:
                ax.axhline(p0.T_max, color=muted, lw=1, ls='--')
            ax.axvline(tau_one_pressure(params, model) / 1e5, color=muted, lw=1, ls=':')
            ax.set_xscale('log')
            ax.grid(color=grid, lw=0.6)
            for s in ax.spines.values():
                s.set_color(grid)
            ax.tick_params(colors=muted, labelcolor=ink)
            if i == 0:
                ax.set_title(pname, color=ink)
            if i == len(models) - 1:
                ax.set_xlabel('Surface pressure (bar)', color=ink)
            if j == 0:
                ax.set_ylabel(f'{mname}\n{ylabel}', color=ink)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(alphas), frameon=False, labelcolor=ink)
    note = 'solid: $T_{eq}$ (full redistribution)   dotted: $\\tau = 1$' + ('   dashed: airless dayside $T_0$' if airless_line else '')
    fig.suptitle(f'{title}\n{note}', color=ink, fontsize=11)
    path = os.path.join(here, '..', 'figures', filename)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print('saved', os.path.normpath(path))

plot(3, 'Nightside surface temperature (K)', 'Wirth et al. (2026) Fig. 3: nightside surface temperature, gray CO$_2$', 'wirth2026_fig3_nightside_surface.png', False)
plot(0, 'Dayside surface temperature (K)', 'Wirth et al. (2026) Fig. 7: dayside surface temperature, gray CO$_2$', 'wirth2026_fig7_dayside_surface.png', True)
