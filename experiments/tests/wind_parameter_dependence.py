"""How the 7 fitted parameters of parameterise_winds.parametric_wind depend on the inputs (T_day, ΔT, P0, P_rot).

For each parameter: Spearman rank correlation with each input, the variance each input explains (drop in R² when it is removed from a
regression on ln ΔT, ln P0, T_day, ln P_rot, ln² P_rot), regressions within P_rot groups, and binned medians.
--sweep space_filling (default) uses experiments/sweeps/parametric_wind_sweep.py; --sweep grid the original convergence sweep.

    /data/pt426/ocean-venv/bin/python experiments/tests/wind_parameter_dependence.py [--sweep grid]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "climate"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameterise_winds import PARAM_NAMES

ORDINAL_BLUE = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")  # as in src/climate/wind_scaling.py
LABELS = {"U_p": "U_p / m s⁻¹", "theta_E": "θ_E / °", "theta_W": "θ_W / °", "k_E": "k_E / rad⁻¹", "k_W": "k_W / rad⁻¹", "f": "f",
          "u0": "u0 / m s⁻¹"}
P_ROT_GROUPS = (4, 8, 15, 30, 60, 120)  # d


def load(sweep):
    if sweep == "grid":
        from wind_decomposition_test import emulate_sweep_fields
        from parametric_wind_test import fit_all
        f = emulate_sweep_fields()
        return f, fit_all(f)
    f = dict(np.load(os.path.join(ROOT, "output", "winds", "parametric_sweep_fields.npz")))
    return f, np.load(os.path.join(ROOT, "output", "winds", "parametric_sweep_fits.npz"))["params"]


def inputs(f):
    return {"T_day": f["T_day"], "dT": f["T_day"] - f["T_night"], "P0": f["P0"], "P_rot": f["P_rot"]}


def regression_features(x, drop=None):
    cols = {"dT": [np.log(x["dT"])], "P0": [np.log(x["P0"])], "T_day": [x["T_day"]],
            "P_rot": [np.log(x["P_rot"]), np.log(x["P_rot"]) ** 2]}
    return np.column_stack([np.ones(len(x["dT"]))] + [c for k, cs in cols.items() if k != drop for c in cs])


def r_squared(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.var(y - X @ c) / np.var(y)


def transform(name, y):
    """Regress positive scale parameters in log space; k, f, u0 can be 0 or negative."""
    return np.log(y) if name in ("U_p", "theta_E", "theta_W") else y


def p_rot_groups(P_rot):
    """(label, mask) for each P_rot group (unique values for the grid sweep, bins otherwise)."""
    values = np.unique(P_rot)
    if values.size <= 6:
        return [(f"{p:g} d", P_rot == p) for p in values]
    edges = P_ROT_GROUPS
    return [(f"{a:g}-{b:g} d", (P_rot >= a) & (P_rot < b)) for a, b in zip(edges[:-1], edges[1:])]


def binned(x, y, n_bins=6, log=False):
    """Bin centres and medians of y in bins of x (unique values if there are few)."""
    if np.unique(x).size <= 8:
        levels = np.unique(x)
        return levels, np.array([np.median(y[x == v]) for v in levels])
    edges = np.geomspace(x.min(), x.max(), n_bins + 1) if log else np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centres = np.sqrt(edges[:-1] * edges[1:]) if log else 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(y[idx == i]) if (idx == i).sum() >= 5 else np.nan for i in range(n_bins)])
    return centres, med


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="space_filling", choices=["space_filling", "grid"])
    args = parser.parse_args()
    f, params = load(args.sweep)
    x = inputs(f)
    figure = os.path.join(ROOT, "figures", f"wind_parameter_dependence_{args.sweep}.png")
    print(f"{len(params)} planets ({args.sweep} sweep); T_day {x['T_day'].min():.0f}-{x['T_day'].max():.0f} °C, "
          f"ΔT {x['dT'].min():.0f}-{x['dT'].max():.0f} K, P0 {x['P0'].min():g}-{x['P0'].max():g} bar, P_rot {x['P_rot'].min():g}-{x['P_rot'].max():g} d\n")

    print("Spearman rank correlation with each input, and variance explained (drop in R² when that input is removed)")
    print(f"{'':>8} | {'T_day':>13} {'ΔT':>13} {'P0':>13} {'P_rot':>13} | R² all")
    for j, name in enumerate(PARAM_NAMES):
        y = transform(name, params[:, j])
        full = r_squared(regression_features(x), y)
        cells = [f"{spearmanr(x[k], params[:, j]).statistic:+.2f} ({full - r_squared(regression_features(x, drop=k), y):.2f})"
                 for k in ("T_day", "dT", "P0", "P_rot")]
        print(f"{name:>8} | " + " ".join(f"{c:>13}" for c in cells) + f" | {full:.2f}")

    print("\nwithin each P_rot group: regression on ln ΔT, ln P0, T_day — slopes (d ln y for U_p, θ; d y otherwise), R², median")
    groups = p_rot_groups(x["P_rot"])
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name}:")
        for label, s in groups:
            X = np.column_stack([np.ones(s.sum()), np.log(x["dT"][s]), np.log(x["P0"][s]), x["T_day"][s]])
            y = transform(name, params[s, j])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            med = np.median(np.degrees(params[s, j]) if name.startswith("theta") else params[s, j])
            print(f"    {label:>9} (n={s.sum():4d}): ΔT {c[1]:+.2f}  P0 {c[2]:+.2f}  T_day {c[3]:+.3f}  R² {r_squared(X, y):.2f}  median {med:.2f}")

    # Figure: binned medians of each parameter against each input, one line per P_rot group; last column against P_rot itself
    cols = (("dT", "ΔT / K", False), ("P0", "P0 / bar", True), ("T_day", "T_day / °C", False))
    fig, axes = plt.subplots(len(PARAM_NAMES), 4, figsize=(14, 2.1 * len(PARAM_NAMES)), sharex="col")
    for j, name in enumerate(PARAM_NAMES):
        y = np.degrees(params[:, j]) if name.startswith("theta") else params[:, j]
        for col, (k, label, log) in enumerate(cols):
            ax = axes[j, col]
            for c, (glabel, s) in zip(ORDINAL_BLUE[1:], groups):
                xc, med = binned(x[k][s], y[s], log=log)
                ax.plot(xc, med, color=c, lw=1.8, marker="o", ms=4, mec="white", label=glabel)
            ax.set_xscale("log" if log else "linear")
            ax.grid(alpha=0.25, lw=0.6)
            if j == len(PARAM_NAMES) - 1:
                ax.set_xlabel(label)
        ax = axes[j, 3]
        xc, med = binned(x["P_rot"], y, n_bins=10, log=True)
        q = [np.percentile(y[(x["P_rot"] >= a) & (x["P_rot"] < b)], (25, 75)) if ((x["P_rot"] >= a) & (x["P_rot"] < b)).sum() >= 5
             else (np.nan, np.nan) for a, b in zip(*_edges(x["P_rot"], 10))]
        q = np.array(q)
        ax.fill_between(xc, q[:, 0], q[:, 1], color=ORDINAL_BLUE[2], alpha=0.2, lw=0)
        ax.plot(xc, med, color=ORDINAL_BLUE[3], lw=1.8, marker="o", ms=4, mec="white")
        ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.6)
        if j == len(PARAM_NAMES) - 1:
            ax.set_xlabel("P_rot / d (median, 25-75%)")
        axes[j, 0].set_ylabel(LABELS[name])
    axes[0, 0].legend(title="P_rot", frameon=False, fontsize=7, title_fontsize=7, ncol=2)
    fig.suptitle(f"Fitted wind parameters vs inputs ({args.sweep} sweep, {len(params)} planets): binned medians", y=1.0)
    fig.tight_layout()
    fig.savefig(figure, dpi=120)
    print(f"\nfigure -> {os.path.normpath(figure)}")


def _edges(x, n_bins):
    if np.unique(x).size <= 8:
        v = np.unique(x)
        return v - 1e-9, v + 1e-9
    e = np.geomspace(x.min(), x.max(), n_bins + 1)
    e[-1] *= 1 + 1e-9
    return e[:-1], e[1:]


if __name__ == "__main__":
    main()
