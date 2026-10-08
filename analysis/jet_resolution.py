"""
Analyse the jet resolution/scheme test (experiments/tests/jet_resolution.sh): jet speed, width and depth, temperature,
near-grid noise and cost for every run, and convergence against the 0.25° reference runs.

Run with (from any directory, after `pip install -r requirements.txt`):  python analysis/jet_resolution.py
"""

import glob
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from analysis.paths import OCEAN_DIR, FIG_DIR

warnings.filterwarnings("ignore")

SCHEMES = ["bih10", "bih100", "weno9", "weno5", "weno9_tr"]
REFERENCES = ["weno9", "bih10"]   # schemes with a 0.25° run
EQ_BAND = 5.0                      # jet = zonal-mean u within ±EQ_BAND° of the equator ...
JET_DEPTH = 100.0                  # ... averaged over the top JET_DEPTH m (Δz-weighted)
MIN_YEARS = 0.5                    # runs with less valid output than this are skipped


def run_name(scheme, res):
    return f"jetres_{scheme}_{res}deg_D200"


def find_runs():
    runs = {}
    for path in glob.glob(os.path.join(OCEAN_DIR, "jetres_*deg_D200_zonal.nc")):
        m = re.match(r"jetres_(.+)_([\d.]+)deg_D200_zonal\.nc", os.path.basename(path))
        if m:
            runs[(m[1], float(m[2]))] = run_name(m[1], m[2])
    return runs


def cost_per_model_year(log_path):
    """GPU minutes per model year from the timestamped log, skipping the first 10 model days."""
    if not os.path.exists(log_path):
        return np.nan
    scale = {"seconds": 1 / 86400, "minutes": 1 / 1440, "hours": 1 / 24, "days": 1.0, "years": 365.0}
    rows = []
    for line in open(log_path, errors="ignore"):
        m = re.search(r"^(\d+) .*Time: ([\d.]+) (seconds|minutes|hours|days|years),", re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m:
            rows.append((int(m[1]), float(m[2]) * scale[m[3]]))
    rows = [r for r in rows if r[1] >= 10]
    if len(rows) < 2 or rows[-1][1] <= rows[0][1]:
        return np.nan
    return (rows[-1][0] - rows[0][0]) / 60 / ((rows[-1][1] - rows[0][1]) / 365)


def near_grid_noise(path_3d, var="w"):
    """Share of the zonal-anomaly variance of `var` at wavelengths below 3 grid cells, in the final 3D snapshot."""
    if not os.path.exists(path_3d):
        return np.nan
    a = nc.Dataset(path_3d)[var][-1].astype(float)
    a = a - a.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(a, axis=-1)) ** 2
    k = np.arange(power.shape[-1])
    return power[..., k > a.shape[-1] / 3].sum() / power[..., 1:].sum()


def analyse(name):
    zonal = nc.Dataset(os.path.join(OCEAN_DIR, f"{name}_zonal.nc"))
    full = nc.Dataset(os.path.join(OCEAN_DIR, f"{name}.nc"))
    t = zonal["time"][:] / (365 * 86400)
    lat, z = zonal["φ_aca"][:], zonal["z_aac"][:]
    dz = full["Δz_aac"][:]
    eq, top = np.abs(lat) < EQ_BAND, z > -JET_DEPTH
    u, T = zonal["u"][:].astype(float), zonal["T"][:].astype(float)
    ok = np.isfinite(u).all(axis=(1, 2)) & np.isfinite(T).all(axis=(1, 2))   # drop records written after a NaN
    t, u, T = t[ok], u[ok], T[ok]
    if len(t) < 2 or t[-1] < MIN_YEARS:
        return None

    jet = np.array([np.average(u[i][top][:, eq].mean(axis=1), weights=dz[top]) for i in range(len(t))])
    window = t >= t[-1] - min(1.0, t[-1] / 2)               # last model year (or last half of a short run)
    u_mean = u[window].mean(axis=0)
    profile = u_mean[:, eq].mean(axis=1)
    k_core = int(np.argmax(profile))
    j0 = int(np.argmin(np.abs(lat)))
    core_row = u_mean[k_core]
    above = np.where(core_row > 0.5 * core_row[j0])[0]
    width = (lat[above].min(), lat[above].max()) if len(above) else (np.nan, np.nan)
    w_lat = np.cos(np.deg2rad(lat))
    mean_T = np.sum(np.average(T[-1], axis=1, weights=w_lat) * dz) / dz.sum()

    return dict(t=t, jet=jet, jet_final=jet[window].mean(), jet_std=jet[window].std(), core=z[k_core],
                width=width, mean_T=mean_T, profile=profile, z=z,
                noise_w=near_grid_noise(os.path.join(OCEAN_DIR, f"{name}.nc")),
                cost=cost_per_model_year(os.path.join(OCEAN_DIR, f"{name}.log")))


def main():
    runs = find_runs()
    if not runs:
        print(f"No jetres_* runs found in {OCEAN_DIR}")
        return
    results = {key: analyse(name) for key, name in sorted(runs.items())}
    for key in [k for k, r in results.items() if r is None]:
        print(f"Skipping {runs[key]}: under {MIN_YEARS} model yr of valid output (crashed?)")
        del results[key]
    resolutions = sorted({res for _, res in results}, reverse=True)

    print(f"{'scheme':9s} {'res':>5s} {'years':>6s} {'jet (m/s)':>16s} {'core (m)':>9s} {'half-width (°)':>15s} "
          f"{'mean T':>7s} {'w noise':>8s} {'min/yr':>7s}  " + "  ".join(f"vs {r} 0.25°" for r in REFERENCES))
    for (scheme, res), r in sorted(results.items(), key=lambda kv: (SCHEMES.index(kv[0][0]) if kv[0][0] in SCHEMES else 99, -kv[0][1])):
        errors = []
        for ref in REFERENCES:
            ref_run = results.get((ref, 0.25))
            errors.append(f"{100 * (r['jet_final'] - ref_run['jet_final']) / ref_run['jet_final']:+6.1f}%" if ref_run else "      —")
        print(f"{scheme:9s} {res:5.2f} {r['t'][-1]:6.2f} {r['jet_final']:9.3f} ± {r['jet_std']:.3f} {r['core']:9.0f} "
              f"{r['width'][0]:6.1f}..{r['width'][1]:<6.1f} {r['mean_T']:7.3f} {100 * r['noise_w']:7.1f}% {r['cost']:7.2f}  "
              + "      ".join(errors))

    os.makedirs(FIG_DIR, exist_ok=True)
    present = [s for s in SCHEMES if any(k[0] == s for k in results)]
    colours = {res: c for res, c in zip(resolutions, plt.cm.viridis(np.linspace(0, 0.85, len(resolutions))))}

    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3.5), sharey=True, squeeze=False)
    for ax, scheme in zip(axes[0], present):
        for res in resolutions:
            if (scheme, res) in results:
                r = results[(scheme, res)]
                ax.plot(r["t"], r["jet"], color=colours[res], label=f"{res}°")
        ax.set_title(scheme)
        ax.set_xlabel("model years")
        ax.legend()
    axes[0][0].set_ylabel(f"jet: zonal-mean u, |φ| < {EQ_BAND:g}°, top {JET_DEPTH:g} m (m/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "jetres_jet_vs_time.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    for scheme in present:
        pts = sorted((res, results[(scheme, res)]["jet_final"]) for res in resolutions if (scheme, res) in results)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=scheme)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing (degrees)")
    ax.set_ylabel("final jet speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "jetres_convergence.png"), dpi=150)

    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.8), sharey=True, squeeze=False)
    for ax, scheme in zip(axes[0], present):
        for res in resolutions:
            if (scheme, res) in results:
                r = results[(scheme, res)]
                ax.plot(r["profile"], r["z"], color=colours[res], label=f"{res}°")
        ax.set_title(scheme)
        ax.set_xlabel("equatorial zonal-mean u (m/s)")
        ax.legend()
    axes[0][0].set_ylabel("z (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "jetres_profiles.png"), dpi=150)
    print(f"\nFigures: {FIG_DIR}/jetres_jet_vs_time.png, jetres_convergence.png, jetres_profiles.png")


if __name__ == "__main__":
    main()
