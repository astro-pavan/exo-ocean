# thick_atmosphere_dry_planet for 55 Cancri e: grey CO2 atmosphere over a magma ocean, with day-night magma transport
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'src', 'two_column'))
import planet_parameters as pp
import two_column as tc

warnings.filterwarnings('ignore')

planet = pp.cnc_55_e

P_surfaces = [1e4, 1e5, 1e6] # Pa; Hu et al. (2024) retrieve 0.01-100 bar
J_oceans = [0, 1e14]          # magma mass flux day to night, kg/s; ~3.4e14 gives O = 1
alpha_scaling = 'coriolis'    # Lambda ~ 0.5 for 55 Cnc e, so rotation matters (W26 eq. 58); None uses alpha below
alpha = 0.1
eps_wind = 0.5                # heat-engine efficiency (KA16 Fig. 6b)

def build(P_surface, J_ocean, alpha=alpha, alpha_scaling=alpha_scaling, params=planet):
    return tc.thick_atmosphere_dry_planet.from_parameters(params, alpha, P_surface, J_ocean=J_ocean, eps_wind=eps_wind, alpha_scaling=alpha_scaling)

# line variants as (legend label, alpha, alpha_scaling, line style)
fixed_alpha = (('$v = 0.05\\,c_s$', 0.05, None, '--'), ('$v = 1.0\\,c_s$', 1.0, None, '-'))
scaled_alpha = (('Coriolis-dominated $\\alpha$ (W26 eq. 58)', 0.1, 'coriolis', '--'), ('WTG $\\alpha$ (W26 eq. 54)', 0.1, 'WTG', '-'))

# Wirth et al. (2026) Fig. 3/7-style plot: surface temperatures against pressure, coloured by ocean mass flux, line style by the alpha variant
def plot_temperatures(params=planet, name='55 Cnc e', J_values=(0, 1e12, 1e13, 1e14, 1e15), variants=fixed_alpha, suffix='',
                      pressures=np.geomspace(1e2, 1e7, 31), tol=1e-5, cmap='plasma'):
    colours = plt.get_cmap(cmap)(np.linspace(0.2, 0.95, len(J_values))) # skip the palest end, which vanishes on the light background
    ink, muted, grid, surface = '#0b0b0b', '#6b6a66', '#e4e3df', '#fcfcfb'
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(surface)
    for J, c in zip(J_values, colours):
        for _, a, scaling, ls in variants:
            T = []
            for P in pressures:
                p = build(P, J, alpha=a, alpha_scaling=scaling, params=params)
                x = p.solve()
                ok = x is not None and np.abs(p.residuals(x)).max() < tol
                T.append(p.get_physical_properties(x)[::3][:2] if ok else (np.nan, np.nan)) # (T_s,d, T_s,n)
            T = np.array(T)
            axes[0].plot(pressures / 1e5, T[:, 0], color=c, lw=2, ls=ls)
            axes[1].plot(pressures / 1e5, T[:, 1], color=c, lw=2, ls=ls)
    p0 = build(1e5, 0, params=params)
    T_eq = params.T_star * np.sqrt(params.R_star / (2 * params.semi_major_axis))
    for ax, title in zip(axes, ('Dayside', 'Nightside')):
        ax.set_facecolor(surface)
        ax.set_title(title, color=ink)
        ax.set_xlabel('Surface pressure (bar)', color=ink)
        ax.axhline(T_eq, color=muted, lw=1)
        ax.axvline(p0.g / (2 * p0.kappa) / 1e5, color=muted, lw=1, ls=':')
        ax.set_xscale('log')
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelcolor=ink)
    axes[0].set_ylabel('Surface temperature (K)', color=ink)
    axes[0].axhline(p0.T_max, color=muted, lw=1, ls='--')
    # discrete colorbar, one block per J since the values are not evenly spaced and include zero
    n = len(J_values)
    cbar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(np.arange(n + 1), n), cmap=ListedColormap(colours[:n])), ax=axes, ticks=np.arange(n) + 0.5,
                        fraction=0.04, pad=0.02)
    cbar.ax.set_yticklabels(['0' if J == 0 else f'$10^{{{np.log10(J):.0f}}}$\n($\\tilde O$ = {build(1e5, J, params=params).O:.2g})' for J in J_values], color=ink)
    cbar.set_label('Ocean mass flux J (kg s$^{-1}$)', color=ink)
    cbar.outline.set_edgecolor(grid)
    cbar.ax.tick_params(colors=muted, labelcolor=ink, length=0)
    handles = [Line2D([], [], color=ink, lw=2, ls=ls, label=label) for label, _, _, ls in variants]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(variants), frameon=False, labelcolor=ink)
    fig.suptitle(f'{name}, thick_atmosphere_dry_planet, gray CO$_2$\n'
                 'grey solid: $T_{eq}$   grey dashed: airless dayside $T_0$   dotted: $\\tau = 1$',
                 color=ink, fontsize=11)
    slug = ''.join(ch for ch in name.lower() if ch.isalnum())
    path = os.path.normpath(os.path.join(here, '..', 'figures', f'lava_ocean_{slug}_temperatures{suffix}.png'))
    fig.savefig(path, dpi=150, facecolor=surface)
    print('saved', path)

# dayside and nightside surface temperatures over surface pressure and ocean mass flux, at the script's alpha setting
def solve_grid(params, pressures, J_values, tol=1e-5):
    T_sd, T_sn = (np.full((len(J_values), len(pressures)), np.nan) for _ in range(2))
    converged = lambda p, x: x is not None and np.abs(p.residuals(x)).max() < tol
    for j, P in enumerate(pressures):
        x_prev = None
        for i, J in enumerate(J_values):
            p = build(P, J, params=params)
            x = p.solve()
            if not converged(p, x) and x_prev is not None: # retry from the neighbouring J, e.g. at the solidus kink in the ocean flux
                x = p.solve(x_prev)
            if converged(p, x):
                T_sd[i, j], _, _, T_sn[i, j], _ = p.get_physical_properties(x)
                x_prev = x
    return T_sd, T_sn

# contour labels at the point of each line furthest from the axes edges (in log-log space); lines hugging an edge get no label
def label_positions(cl, x_range, y_range, min_distance=0.08):
    positions = []
    for level, segs in zip(cl.levels, cl.allsegs):
        best, best_d = None, 0
        for seg in segs:
            if len(seg) == 0:
                continue
            u = (np.log10(seg[:, 0]) - np.log10(x_range[0])) / np.log10(x_range[1] / x_range[0])
            v = (np.log10(seg[:, 1]) - np.log10(y_range[0])) / np.log10(y_range[1] / y_range[0])
            d = np.minimum.reduce([u, 1 - u, v, 1 - v])
            k = np.argmax(d)
            if d[k] > best_d:
                best, best_d = seg[k], d[k]
        if best is not None and best_d >= min_distance:
            positions.append((level, *best))
    return positions

def plot_map(params, name, pressures, J_values, Z, quantity, filename, cmap):
    ink, muted, grid, surface = '#0b0b0b', '#6b6a66', '#e4e3df', '#fcfcfb'
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(surface)
    levels = np.arange(np.floor(np.nanmin(Z) / 250) * 250, np.nanmax(Z) + 250, 250)
    if levels[0] < 0: # day-night contrast can change sign: diverging map centred on zero
        vmax = max(-levels[0], levels[-1])
        cf = ax.contourf(pressures / 1e5, J_values, Z, levels=levels, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    else:
        cf = ax.contourf(pressures / 1e5, J_values, Z, levels=levels, cmap=cmap)
    cl = ax.contour(pressures / 1e5, J_values, Z, levels=levels[levels % 1000 == 0], colors='white', linewidths=0.8)
    for level, x, y in label_positions(cl, (pressures[0] / 1e5, pressures[-1] / 1e5), (J_values[0], J_values[-1])):
        ax.text(x, y, f'{level:.0f} K', color='white', fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a19', ec='none', alpha=0.6))
    p0 = build(1e5, 0, params=params)
    ax.axvline(p0.g / (2 * p0.kappa) / 1e5, color='white', lw=1, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Surface pressure (bar)', color=ink)
    ax.set_ylabel('Ocean mass flux J (kg s$^{-1}$)', color=ink)
    O_per_J = build(1e5, 1.0, params=params).O
    secax = ax.secondary_yaxis('right', functions=(lambda J: J * O_per_J, lambda O: O / O_per_J))
    secax.set_ylabel('$\\tilde O$', color=ink)
    for a in (ax, secax):
        a.tick_params(colors=muted, labelcolor=ink)
    for s in list(ax.spines.values()) + list(secax.spines.values()):
        s.set_color(grid)
    cbar = fig.colorbar(cf, ax=ax, pad=0.12)
    cbar.set_label(quantity, color=ink)
    cbar.ax.tick_params(colors=muted, labelcolor=ink)
    cbar.outline.set_edgecolor(grid)
    label = {'coriolis': 'Coriolis-dominated $\\alpha$', 'WTG': 'WTG $\\alpha$'}.get(alpha_scaling, f'$v = {alpha}\\,c_s$')
    ax.set_title(f'{name}, thick_atmosphere_dry_planet, gray CO$_2$, {label}\ndotted: $\\tau = 1$', color=ink, fontsize=11)
    slug = ''.join(ch for ch in name.lower() if ch.isalnum())
    path = os.path.normpath(os.path.join(here, '..', 'figures', f'lava_ocean_{slug}_{filename}.png'))
    fig.savefig(path, dpi=150, facecolor=surface)
    print('saved', path, f'({np.isnan(Z).sum()} of {Z.size} points did not converge)')

# nightside, dayside and day-night contrast maps from one grid of solutions
def plot_maps(params=planet, name='55 Cnc e', pressures=np.geomspace(1e2, 1e7, 41), J_values=np.geomspace(1e11, 1e16, 41)):
    T_sd, T_sn = solve_grid(params, pressures, J_values)
    plot_map(params, name, pressures, J_values, T_sn, 'Nightside surface temperature (K)', 'nightside_map', 'inferno')
    plot_map(params, name, pressures, J_values, T_sd, 'Dayside surface temperature (K)', 'dayside_map', 'inferno')
    plot_map(params, name, pressures, J_values, T_sd - T_sn, 'Day-night surface temperature contrast (K)', 'contrast_map', 'viridis')

# temperatures, winds and the main energy fluxes of a solution, fluxes in W/m^2
def summary(p, x):
    T0, T1, T2, T3, W = x
    T_sd, T_ad, T_an, T_sn, w = p.get_physical_properties(x)
    a = p.alpha_at(T0, T1, T2)
    v = a * p.V0 * np.sqrt(T0 * T1) * p.W0
    F = tc.sigma * p.T_max**4
    C = (tc.xi * p.CT) / (tc.chi * p.CD)
    Fa = p.tau * C * a * p.V0 * (T0 * T1)**1.5 * (1 - T2) * F
    Fo = p.O * np.maximum(T0 - np.maximum(T0 * T1 * T2 * T3, p.T_solidus / p.T_max), 0) * F
    OLR_d = (p.t * T0**4 + p.U_day * (T0 * T1)**4) * F
    OLR_n = (p.t * (T0 * T1 * T2 * T3)**4 + p.U_night * (T0 * T1 * T2)**4) * F
    return dict(T_sd=T_sd, T_ad=T_ad, T_an=T_an, T_sn=T_sn, w=w, v=v, alpha=a, Fa=Fa, Fo=Fo, OLR_d=OLR_d, OLR_n=OLR_n,
                residual=np.abs(p.residuals(x)).max())

if __name__ == '__main__':
    p0 = build(1e5, 0)
    print(f"55 Cnc e: T_max = {p0.T_max:.0f} K, g = {p0.g:.1f} m/s2, absorbed dayside flux = {tc.sigma * p0.T_max**4:.3e} W/m2"
          + (f", Lambda = {p0.Lambda:.2f}" if alpha_scaling == 'coriolis' else ""))
    print(f"{'p (bar)':>7} {'tau':>6} {'J (kg/s)':>9} {'O':>6} {'alpha':>6} {'T_sd':>6} {'T_ad':>6} {'T_an':>6} {'T_sn':>6} "
          f"{'w':>6} {'v':>6} {'F_a':>9} {'F_o':>9} {'OLR_d':>9} {'OLR_n':>9} {'|r|':>7}")
    for P in P_surfaces:
        for J in J_oceans:
            p = build(P, J)
            x = p.solve()
            if x is None:
                print(f"{P / 1e5:7.2f} no solution")
                continue
            s = summary(p, x)
            print(f"{P / 1e5:7.2f} {p.tau:6.2f} {J:9.1e} {p.O:6.2f} {s['alpha']:6.3f} {s['T_sd']:6.0f} {s['T_ad']:6.0f} {s['T_an']:6.0f} {s['T_sn']:6.0f} "
                  f"{s['w']:6.0f} {s['v']:6.0f} {s['Fa']:9.2e} {s['Fo']:9.2e} {s['OLR_d']:9.2e} {s['OLR_n']:9.2e} {s['residual']:7.0e}")

    for params, name in ((pp.cnc_55_e, '55 Cnc e'), (pp.k2_141_b, 'K2-141 b'), (pp.toi_561_b, 'TOI-561 b')):
        plot_temperatures(params, name)
        plot_temperatures(params, name, variants=scaled_alpha, suffix='_alpha_scaling')
        plot_maps(params, name)
