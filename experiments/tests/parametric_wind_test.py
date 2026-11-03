"""Fit the 7-parameter converging wind (peak speed, east/west peak angles and decay rates, equatorial focusing, uniform zonal wind; parameterise_winds.parametric_wind) to every CLERO sweep planet and score it on the ocean-relevant
stress metrics of converging_wind_test.py, against the 7-parameter spherical-harmonic model (3 χ + 4 ψ).

    /data/pt426/ocean-venv/bin/python experiments/tests/parametric_wind_test.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "climate"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import climate_emulator as ce
from parameterise_winds import parametric_wind, fit_parametric_wind, PARAM_NAMES, PARAM_BOUNDS
from spherical_harmonic_winds import decompose_winds, to_tidally_locked, reconstruct_winds
from converging_wind_test import ocean_metrics, model_errors, converging_model, CATEGORICAL, EQ_LAT
from wind_decomposition_test import emulate_sweep_fields

PARAMS = os.path.join(ROOT, "output", "winds", "parametric_wind_fits.npz")
FIGURE = os.path.join(ROOT, "figures", "parametric_wind_test.png")


def fit_all(f, path=PARAMS, u0=True):
    """Stress-fitted params for every planet (cached); u0=False holds the uniform zonal wind at 0."""
    key = "params" if u0 else "params_no_u0"
    cache = dict(np.load(path)) if os.path.exists(path) else {}
    if key not in cache:
        cache[key] = np.array([fit_parametric_wind(f["u"][i], f["v"][i], ce.LAT, ce.LON, f["rho"][i], fit_u0=u0)[0]
                               for i in range(len(f["u"]))])
        np.savez(path, **cache)
    return cache[key]


def main():
    f = emulate_sweep_fields()
    ref = ocean_metrics(f["u"], f["v"], f["rho"], f["P0"])
    t = to_tidally_locked(decompose_winds(f["u"], f["v"], ce.LAT, ce.LON))
    winds = {"spherical harmonics, 3 χ + 4 ψ (7)": reconstruct_winds(converging_model(t, 3, n_psi=4))}
    fits = {}
    for name, u0 in (("parametric, u0 = 0 (6)", False), ("parametric (7)", True)):
        fits[name] = fit_all(f, u0=u0)
        winds[name] = tuple(np.array(x) for x in zip(*[parametric_wind(p, ce.LAT, ce.LON) for p in fits[name]]))
    errs = {}
    print("errors: median [10th-90th pct] over planets (tau_conv, ustar3 signed relative; profiles/maps relative RMS)")
    for name, (u, v) in winds.items():
        errs[name] = model_errors(ref, ocean_metrics(u, v, f["rho"], f["P0"]))
        print(f"  {name}:\n    " + "  ".join(f"{k} {np.median(e):+.3f} [{np.percentile(e, 10):+.3f}, {np.percentile(e, 90):+.3f}]"
                                         for k, e in errs[name].items()))
    for name in errs:
        print(f"\n{name}, |error| median by P_rot:")
        for p in np.unique(f["P_rot"]):
            s = f["P_rot"] == p
            print(f"  P_rot {p:5g} d: " + "  ".join(f"{k} {np.median(np.abs(e[s])):.3f}" for k, e in errs[name].items()))
    P = fits["parametric (7)"]
    print("\nfitted parameters (7-param model), median [10th-90th pct] by P_rot:")
    for j, k in enumerate(PARAM_NAMES):
        conv = np.degrees if k.startswith("theta") else (lambda x: x)
        print(f"  {k:>8}: " + "  ".join(f"{p:g} d {np.median(conv(P[f['P_rot'] == p, j])):.2f} [{np.percentile(conv(P[f['P_rot'] == p, j]), 10):.2f}, "
                                          f"{np.percentile(conv(P[f['P_rot'] == p, j]), 90):.2f}]" for p in np.unique(f["P_rot"])))
    at_bound = {k: np.mean(np.isclose(P[:, j], lo) | np.isclose(P[:, j], hi))
                for j, (k, lo, hi) in enumerate(zip(PARAM_NAMES, *PARAM_BOUNDS))}
    print("  fraction of fits at a bound: " + ", ".join(f"{k} {v:.2f}" for k, v in at_bound.items()))

    # Figure: equatorial zonal stress for example planets (emulator, 7-term harmonics, 7-param fit) + tau_conv scatter
    examples = [np.flatnonzero((f["P_rot"] == p) & (f["P0"] == 3))[0] for p in (6, 25, 100)]
    tx = {name: ocean_metrics(u, v, f["rho"], f["P0"])["tx_eq"] for name, (u, v) in winds.items()}
    show = ["spherical harmonics, 3 χ + 4 ψ (7)", "parametric (7)"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, i in zip(axes, examples):
        ax.axhline(0, color="#9a9a9a", lw=0.8)
        ax.plot(ce.LON, ref["tx_eq"][i], color="#333333", lw=2, label="emulator")
        for name, c in zip(show, CATEGORICAL):
            ax.plot(ce.LON, tx[name][i], color=c, lw=1.8, label=name)
        ax.set(xlabel="longitude from substellar / °", xticks=[-180, -90, 0, 90, 180],
               title=f"P_rot {f['P_rot'][i]:g} d, P0 {f['P0'][i]:g} bar, ΔT {f['T_day'][i] - f['T_night'][i]:g} K")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel(f"zonal stress, |lat| ≤ {EQ_LAT:g}° / N m$^{{-2}}$")
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[-1]
    for name, c in zip(show, CATEGORICAL):
        ax.scatter(ref["tau_conv"], ref["tau_conv"] * (1 + errs[name]["tau_conv"]), s=10, color=c, alpha=0.6, label=name)
    lim = [0, ref["tau_conv"].max() * 1.1]
    ax.plot(lim, lim, color="#9a9a9a", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="emulator τ toward substellar / N m$^{-2}$", ylabel="model", title="equatorial convergent stress")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    print(f"figure -> {os.path.normpath(FIGURE)}")


if __name__ == "__main__":
    main()
