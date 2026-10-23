# Wet two-column models with a 1 bar atmosphere: day and night surface temperatures against instellation for several ocean heat fluxes
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'src', 'two_column'))
import two_column as tc

warnings.filterwarnings('ignore')

S_Earth = 1361 # W/m^2

# Earth-like planet, 1 bar; YA14's reference planet is R_p = 2 R_Earth, g = 13.7 m/s^2. The star only sets a, which the wet models do not use
R_p = 6.371e6
M_p = 9.8 * R_p**2 / tc.G
R_star, T_star = 0.3 * 6.957e8, 3000

J_oceans = (0, 1e10, 1e11, 1e12, 1e13) # day-night ocean mass flux, kg/s; Earth's overturning is ~20 Sv ~ 2e10 kg/s
Sv = 1.025e9 # kg/s per sverdrup of seawater

# ocean heat flux J c_o (T_s,d - T_s,n) / (2 pi r_p^2), set inside the model from its own surface temperatures
def sweep(model, J_ocean, S, x0, **kwargs):
    p = model(R_p, M_p, R_star, T_star, S[0], 0, J_ocean=J_ocean, **kwargs)
    xs = p.solve_branch(S, x0)
    return np.array([(x[0], x[2]) if x is not None else (np.nan, np.nan) for x in xs]) # (T_s,d, T_s,n)

# branches as (legend label, model, S values in W/m^2, initial guess, line style); a downward sweep is plotted against its own S
def plot(branches, title, filename, S_range, J_values=J_oceans, cmap='plasma', model_kwargs={}):
    colours = plt.get_cmap(cmap)(np.linspace(0.2, 0.95, len(J_values)))
    ink, muted, grid, surface = '#0b0b0b', '#6b6a66', '#e4e3df', '#fcfcfb'
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(surface)
    for F, c in zip(J_values, colours):
        for label, model, S, x0, ls in branches:
            T = sweep(model, F, S, x0, **model_kwargs)
            axes[0].plot(S / S_Earth, T[:, 0], color=c, lw=2, ls=ls)
            axes[1].plot(S / S_Earth, T[:, 1], color=c, lw=2, ls=ls)
    for ax, side in zip(axes, ('Dayside', 'Nightside')):
        ax.set_facecolor(surface)
        ax.set_title(side, color=ink)
        ax.set_xlabel('Substellar instellation $S_0$ ($S_\\oplus$)', color=ink)
        ax.set_xlim(*S_range)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelcolor=ink)
    axes[0].set_ylabel('Surface temperature (K)', color=ink)
    n = len(J_values) # discrete colorbar, one block per ocean flux
    cbar = fig.colorbar(ScalarMappable(norm=BoundaryNorm(np.arange(n + 1), n), cmap=ListedColormap(colours)), ax=axes,
                        ticks=np.arange(n) + 0.5, fraction=0.04, pad=0.02)
    cbar.ax.set_yticklabels(['0' if J == 0 else f'$10^{{{np.log10(J):.0f}}}$\n({J / Sv:.2g} Sv)' for J in J_values], color=ink)
    cbar.set_label('Ocean mass flux J (kg s$^{-1}$)', color=ink)
    cbar.outline.set_edgecolor(grid)
    cbar.ax.tick_params(colors=muted, labelcolor=ink, length=0)
    if len(branches) > 1:
        handles = [Line2D([], [], color=ink, lw=2, ls=ls, label=label) for label, _, _, _, ls in branches]
        fig.legend(handles=handles, loc='outside lower center', ncol=len(branches), frameon=False, labelcolor=ink)
    fig.suptitle(title, color=ink, fontsize=11)
    path = os.path.normpath(os.path.join(here, '..', 'figures', filename))
    fig.savefig(path, dpi=150, facecolor=surface)
    print('saved', path)

if __name__ == '__main__':
    # Yang & Abbot (2014): single branch, swept upward
    S_ya14 = np.linspace(0.5, 2.0, 61) * S_Earth
    plot([('', tc.wet_planet, S_ya14, (270, 240, 170, 60, 80), '-')],
         'Yang & Abbot (2014) two-column model, 1 bar, $R_p = R_\\oplus$, $g$ = 9.8 m s$^{-2}$',
         'water_ocean_ya14_instellation.png', (0.5, 2.0))

    # Fan et al. (2024): bistable, so sweep up from a cold start and down from a hot start
    S_fan = np.linspace(0.5, 2.25, 71) * S_Earth
    plot([('cold start (S$_0$ increasing)', tc.runaway_wet_planet, S_fan, (270, 240, 170, 60, 80), '-'),
          ('hot start (S$_0$ decreasing)', tc.runaway_wet_planet, S_fan[::-1], (900, 890, 890, 500, 0), '--')],
         'Fan et al. (2024) runaway two-column model, $R_p = R_\\oplus$, $g$ = 9.8 m s$^{-2}$',
         'water_ocean_fan2024_instellation.png', (0.5, 2.25))
