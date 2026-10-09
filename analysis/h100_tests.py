"""Tables and figures for the five H100 sensitivity tests (experiments/tests/h100_test_*.jl; context in docs/LLM/h100_sensitivity_tests.md).

    python analysis/h100_tests.py

Per run: jet (zonal-mean u, |lat| < 5°, top 100 m, Δz-weighted) at 10 yr and over the last 2 yr with its trend, jet half-width and core
depth, mean and deep (below half depth) T, and peak poleward ocean heat transport (OHT) from the surface-flux budget minus storage, with
the budget residual at the northern wall. Earlier wenoval runs are used as baselines where a test reuses them.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from analysis.paths import FIG_DIR, OCEAN_DIR

EQ_BAND, JET_DEPTH = 5.0, 100.0
R, RHO_CP, TAU_RELAX = 6371e3, 1026 * 3994.0, 30 * 86400
NAME = re.compile(r"(?P<prefix>[a-z]+)_(?P<res>[\d.]+)deg_D(?P<depth>\d+)_P(?P<P>\d+)_dT(?P<dT>\d+)(?P<tags>.*)")


def runs(prefix):
    out = {}
    for path in sorted(glob.glob(os.path.join(OCEAN_DIR, f"{prefix}_*_zonal.nc"))):
        name = os.path.basename(path)[: -len("_zonal.nc")]
        m = NAME.fullmatch(name)
        if m and "smoke" not in name:
            out[name] = dict(res=float(m["res"]), depth=int(m["depth"]), P=int(m["P"]), dT=int(m["dT"]), tags=m["tags"])
    return out


def metrics(name, dT):
    """Diagnostics of one run, or None if it has under a year of valid output."""
    zonal, full = nc.Dataset(os.path.join(OCEAN_DIR, f"{name}_zonal.nc")), nc.Dataset(os.path.join(OCEAN_DIR, f"{name}.nc"))
    t = np.asarray(zonal["time"][:]) / (365 * 86400)
    lat, z, dz = np.asarray(zonal["φ_aca"][:]), np.asarray(zonal["z_aac"][:]), np.asarray(full["Δz_aac"][:])
    u = np.ma.filled(zonal["u"][:].astype(float), np.nan)
    T = np.ma.filled(zonal["T"][:].astype(float), np.nan)
    ok = np.isfinite(u).all(axis=(1, 2)) & np.isfinite(T).all(axis=(1, 2))
    t, u, T = t[ok], u[ok], T[ok]
    if len(t) < 3 or t[-1] < 1:
        return None

    top, eq, deep = z > -JET_DEPTH, np.abs(lat) < EQ_BAND, z < z.min() / 2
    jet = np.array([np.average(u[i][top][:, eq].mean(axis=1), weights=dz[top]) for i in range(len(t))])
    last = t >= t[-1] - 2
    u_mean, T_mean = u[last].mean(axis=0), T[last].mean(axis=0)
    profile = u_mean[:, eq].mean(axis=1)
    k = int(np.argmax(profile))
    j0 = int(np.argmin(np.abs(lat)))
    above = np.where(u_mean[k] > 0.5 * u_mean[k, j0])[0]
    half_width = (lat[above].max() - lat[above].min()) / 2 if len(above) else np.nan

    w = np.cos(np.deg2rad(lat))
    column = np.average(T, axis=2, weights=w)                         # (time, z) area-mean T
    mean_T = np.sum(column * dz, axis=1) / dz.sum()
    deep_T = np.sum(column[:, deep] * dz[deep], axis=1) / dz[deep].sum()

    # OHT: surface heat input (Haney flux, linear in T, so its zonal mean uses the zonal-mean SST) minus column storage, integrated
    # from the southern wall; T_eq's zonal mean is ΔT cos(lat) / π
    flux_in = -RHO_CP * dz[-1] / TAU_RELAX * (T_mean[-1] - dT * w / np.pi)
    i0 = int(np.argmin(np.abs(t - (t[-1] - 2))))
    storage = RHO_CP * np.sum((T[-1] - T[i0]) / ((t[-1] - t[i0]) * 365 * 86400) * dz[:, None], axis=0)
    oht = np.cumsum((flux_in - storage) * 2 * np.pi * R**2 * w * np.deg2rad(lat[1] - lat[0])) / 1e15
    peak_oht = (oht[lat > 0].max() - oht[lat < 0].min()) / 2

    at10 = int(np.argmin(np.abs(t - 10)))
    return dict(t=t, jet=jet, years=t[-1], jet10=jet[at10] if t[-1] >= 9.5 else np.nan, jet_end=jet[last].mean(),
                trend=np.polyfit(t[last], jet[last], 1)[0], half_width=half_width, core=z[k], mean_T=mean_T, deep_T=deep_T,
                mean_T_end=mean_T[-1], deep_T_end=deep_T[-1], oht=peak_oht, residual=oht[-1])


HEADER = f"{'run':58s} {'yr':>5s} {'jet10':>6s} {'jet':>6s} {'trend/yr':>8s} {'hw°':>5s} {'core':>5s} {'T':>6s} {'Tdeep':>6s} {'OHT PW':>7s} {'resid':>6s}"


def row(name, m):
    return (f"{name:58s} {m['years']:5.1f} {m['jet10']:6.3f} {m['jet_end']:6.3f} {m['trend']:+8.4f} {m['half_width']:5.1f} "
            f"{m['core']:5.0f} {m['mean_T_end']:6.2f} {m['deep_T_end']:6.2f} {m['oht']:7.3f} {m['residual']:+6.3f}")


def table(title, names_dT):
    print(f"\n== {title}\n{HEADER}")
    results = {}
    for name, dT in names_dT:
        if not os.path.exists(os.path.join(OCEAN_DIR, f"{name}_zonal.nc")):
            print(f"{name:58s} (missing)")
            continue
        m = metrics(name, dT)
        print(f"{name:58s} (under a year of valid output)" if m is None else row(name, m))
        if m is not None:
            results[name] = m
    return results


def jet_figure(results, filename, title):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, m in results.items():
        ax.plot(m["t"], m["jet"], label=name.split("_", 1)[1])
    ax.set(xlabel="model years", ylabel="jet (m/s)", title=title)
    ax.legend(fontsize=6)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # 1. Vertical mixing
    vm = table("1. Vertical viscosity / diffusivity (200 m, P 10 d, ΔT 30 K)", [(n, 30) for n in runs("vmix")])
    jet_figure(vm, "h100_vertical_mixing.png", "vertical mixing")

    # 2. Horizontal tracer diffusivity, with the 0.25° κ_h = 1000 Float64 baseline from the validation runs
    kh = table("2. Horizontal tracer diffusivity (200 m, P 10 d, ΔT 30 K; compare jet10 with the 0.25° baseline)",
               [(n, 30) for n in runs("khtest")] + [("wenoval_0.25deg_D200_P10_dT30_f64", 30)])
    jet_figure(kh, "h100_tracer_diffusivity.png", "horizontal tracer diffusivity")

    # 3. Wind, with the no-wind baselines
    wd = table("3. Emulator wind stress (P 10 d, ΔT 30 K; no-wind baselines last)",
               [(n, 30) for n in runs("wind")] + [(f"wenoval_{r}deg_D200_P10_dT30", 30) for r in ("1", "0.5")]
               + [("wenoval_1deg_D1000_P10_dT30_warm", 30)])
    jet_figure(wd, "h100_wind.png", "wind stress")

    # 4. Warm starts: each warm run's state n years after the switch against the cold 0.5° reference's equilibrium
    ws = table("4. Warm start (500 m, P 10 d, ΔT 30 K)", [(n, 30) for n in runs("warmtest")])
    ref = ws.get("warmtest_0.5deg_D500_P10_dT30_f64")
    if ref:
        print("\nWarm-started 0.5° runs vs the cold 0.5° equilibrium (% difference) after n years:")
        print(f"{'donor':10s} " + " ".join(f"{f'{n} yr: T / Tdeep / jet':>26s}" for n in (0, 2, 5, 10, 12)))
        for name, m in ws.items():
            if "_from" not in name:
                continue
            cells = []
            for n in (0, 2, 5, 10, 12):
                i = int(np.argmin(np.abs(m["t"] - n)))
                if abs(m["t"][i] - n) > 0.3:
                    cells.append(f"{'—':>26s}")
                    continue
                d = [100 * (m[k][i] / ref[k + "_end" if k != "jet" else "jet_end"] - 1) for k in ("mean_T", "deep_T", "jet")]
                cells.append(f"{d[0]:+7.1f} {d[1]:+7.1f} {d[2]:+7.1f}   ")
            donor = re.search(r"from([\d.]+deg)", name)[1]
            print(f"{donor:10s} " + " ".join(cells))
    jet_figure(ws, "h100_warm_start.png", "warm start (500 m)")

    # 5. Grid requirements: jet at 10 yr by resolution for every (P, ΔT), from the new and the earlier validation runs
    print("\n== 5. Grid requirements (200 m, 10 yr): jet at 10 yr, its % error against the finest grid, and grid cells per half-width")
    cases = {}
    for prefix in ("gridtest", "wenoval"):
        for name, p in runs(prefix).items():
            if p["depth"] != 200 or p["tags"] not in ("", "_f64"):
                continue
            m = metrics(name, p["dT"])
            if m is None or np.isnan(m["jet10"]):
                continue
            cases.setdefault((p["P"], p["dT"]), {})[p["res"]] = m
    print(f"{'P':>3s} {'ΔT':>3s}  " + "  ".join(f"{f'{r}°':>22s}" for r in (1.0, 0.5, 0.25)))
    for (P, dT), by_res in sorted(cases.items()):
        finest = by_res[min(by_res)]["jet10"]
        cells = []
        for r in (1.0, 0.5, 0.25):
            if r in by_res:
                m = by_res[r]
                cells.append(f"{m['jet10']:6.3f} {100 * (m['jet10'] / finest - 1):+6.0f}% {m['half_width'] / r:5.1f} c")
            else:
                cells.append(f"{'—':>22s}")
        print(f"{P:3d} {dT:3d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
