"""Reduce a tidally locked planet's near-surface wind field to a single convergence speed, for scaling-law studies.

Convention (as in the CLERO wind files and ocean_sim.jl): longitude λ and latitude φ in degrees, substellar point at (λ, φ) = (0, 0),
u eastward and v northward, fields shaped (..., lat, lon).
"""

import numpy as np


def toward_substellar_component(u, v, lat, lon, substellar_lon=0.0):
    """Wind component (m/s) along the great-circle direction towards the substellar point; NaN at the substellar and antistellar points.

    The unit vector towards the substellar point is ∇cos(θ)/|∇cos(θ)| with cos θ = cos φ cos λ, i.e. (-sin λ, -sin φ cos λ) / sin θ,
    so on the equator this is -u sign(sin λ).
    """
    lam = np.radians(np.asarray(lon, dtype=float) - substellar_lon)[None, :]
    phi = np.radians(np.asarray(lat, dtype=float))[:, None]
    east, north = -np.sin(lam), -np.sin(phi) * np.cos(lam)
    sin_theta = np.hypot(east, north)
    ok = sin_theta > 1e-6
    east = np.where(ok, east / np.where(ok, sin_theta, 1), np.nan)
    north = np.where(ok, north / np.where(ok, sin_theta, 1), np.nan)
    return np.asarray(u) * east + np.asarray(v) * north


def equatorial_convergence_speed(u, v, lat, lon, lat_width=10.0, lon_width=180.0, substellar_lon=0.0, zonal_only=False):
    """Area-weighted mean wind speed towards the substellar point (m/s) over |lat| <= lat_width and |lon - substellar_lon| <= lon_width.

    Positive means the near-surface flow converges on the substellar point. Leading dimensions of u and v (e.g. samples) are kept.
    lon_width = 180 averages over all longitudes; 90 restricts to the dayside. zonal_only uses -u sign(λ) instead of the full
    great-circle component (which, off the equator near λ = 0 and 180°, also counts meridional flow towards the substellar point).
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dlon = (lon - substellar_lon + 180.0) % 360.0 - 180.0
    mask = (np.abs(lat)[:, None] <= lat_width) & (np.abs(dlon)[None, :] <= lon_width)
    if not mask.any():
        raise ValueError(f"no grid points within |lat| <= {lat_width} and |lon| <= {lon_width}")
    weights = np.cos(np.radians(lat))[:, None] * mask
    if zonal_only:
        w_dir = -np.asarray(u) * np.where(np.sin(np.radians(dlon)) == 0, np.nan, np.sign(np.sin(np.radians(dlon))))[None, :]
    else:
        w_dir = toward_substellar_component(u, v, lat, lon, substellar_lon)
    good = np.isfinite(w_dir) & (weights > 0)
    return (np.where(good, w_dir, 0.0) * weights).sum(axis=(-2, -1)) / (weights * good).sum(axis=(-2, -1))


def wind_file_convergence_speed(path, **kwargs):
    """equatorial_convergence_speed of the lowest-level wind in a climate_emulator.py wind file (NetCDF)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return float(equatorial_convergence_speed(ds.u_0.values, ds.v_0.values, ds.lat.values, ds.lon.values, **kwargs))


# --- Sweep over target climates ----------------------------------------------------------------------------------------

SWEEP_FIELDS = ["P_rot", "P0", "T_day", "T_night", "GCM", "status", "M_star", "T_star", "F_star", "CO2", "feh", "feh_requested", "residual",
                "misfit", "T_day_emu", "T_night_emu", "conv_speed", "conv_speed_dayside", "conv_speed_zonal", "conv_speed_std"]


def _emulator():
    """Import climate_emulator from this directory (it lives next to this file)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import climate_emulator
    return climate_emulator


def sweep_convergence(path, P_rot_values, P0_values, T_day_values, T_night_values, feh=None, vary_feh=True, tol=0.1, GCM="exocam",
                      n_samples=0, prefilter=True, prefilter_margin=2.0, resume=True, device=None, verbose=True):
    """Solve for the emulator planet at every (P_rot, P0, T_day, T_night) with T_night <= T_day and append one CSV row per target to `path`.

    Each solved row holds the planet (M_star, T_star, F_star, CO2, [Fe/H]), the emulated (T_day, T_night) and the equatorial convergence
    speed of the best-estimate lowest-level wind: great-circle over all longitudes (conv_speed), dayside only, and zonal-only.
    With n_samples > 0, conv_speed_std is the spread over that many CLERO draws. status is "ok", "no_solution" (misfit = closest
    approach, K) or "outside_region" (skipped because solution_region, widened by prefilter_margin °C, has no solution there).
    Rows are flushed as they are written; with resume, targets already in the file are skipped.
    """
    import csv, os, time
    from scipy.ndimage import binary_dilation
    ce = _emulator()
    feh = ce.FEH_MEAN if feh is None else feh
    key = lambda P_rot, P0, T_day, T_night: tuple(round(float(x), 6) for x in (P_rot, P0, T_day, T_night))

    done = set()
    if resume and os.path.exists(path):
        with open(path, newline="") as f:
            done = {key(r["P_rot"], r["P0"], r["T_day"], r["T_night"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    targets = [(Td, Tn) for Td in T_day_values for Tn in T_night_values if Tn <= Td]
    n_total, n_done, t0 = len(P_rot_values) * len(P0_values) * len(targets), 0, time.time()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        if new_file:
            writer.writeheader()
        for P_rot in P_rot_values:
            region = None
            if prefilter:
                step = 0.5
                region = ce.solution_region(P_rot, P0_values, T_day_range=(min(T_day_values) - 2, max(T_day_values) + 2),
                                            y_range=(min(T_night_values) - 2, max(T_night_values) + 2), y="T_night", step=step,
                                            feh=feh, GCM=GCM, device=device)
                reach = region["mask_band"] if vary_feh else region["mask"]
                reach = np.array([binary_dilation(m, iterations=max(int(round(prefilter_margin / step)), 1)) for m in reach])
            for p, P0 in enumerate(P0_values):
                for T_day, T_night in targets:
                    n_done += 1
                    if key(P_rot, P0, T_day, T_night) in done:
                        continue
                    row = dict(P_rot=P_rot, P0=P0, T_day=T_day, T_night=T_night, GCM=GCM)
                    if region is not None:
                        i, j = np.abs(region["T_day"] - T_day).argmin(), np.abs(region["y"] - T_night).argmin()
                        if not reach[p, i, j]:
                            writer.writerow(row | dict(status="outside_region"))
                            f.flush()
                            continue
                    try:
                        planet = ce.solve_planet(P_rot, T_day, T_night, P0, feh=feh, vary_feh=vary_feh, tol=tol, GCM=GCM, device=device, verbose=False)
                    except ce.NoSolutionError as err:
                        writer.writerow(row | dict(status="no_solution", misfit=f"{err.misfit:.3f}"))
                        f.flush()
                        continue
                    climate = ce.planet_climate(planet, device)
                    u, v = climate["u_0"], climate["v_0"]
                    row |= dict(status="ok", M_star=f"{planet['M_star']:.5f}", T_star=f"{planet['T_star']:.1f}", F_star=f"{planet['F_star']:.1f}",
                                CO2=f"{planet['CO2']:.4g}", feh=f"{planet['feh']:.3f}", feh_requested=f"{planet.get('feh_requested', planet['feh']):.3f}",
                                residual=f"{planet['residual']:.4f}", T_day_emu=f"{climate['T_day']:.3f}", T_night_emu=f"{climate['T_night']:.3f}",
                                conv_speed=f"{float(equatorial_convergence_speed(u, v, ce.LAT, ce.LON)):.4f}",
                                conv_speed_dayside=f"{float(equatorial_convergence_speed(u, v, ce.LAT, ce.LON, lon_width=90)):.4f}",
                                conv_speed_zonal=f"{float(equatorial_convergence_speed(u, v, ce.LAT, ce.LON, zonal_only=True)):.4f}")
                    if n_samples:
                        s = ce.get_emulator(device).sample({k: planet[k] for k in ce.PLANET_KEYS}, n_samples=n_samples, fields=["u_0", "v_0"],
                                                          seed=0, batch_size=1)
                        row["conv_speed_std"] = f"{float(np.std(equatorial_convergence_speed(np.asarray(s['u_0']), np.asarray(s['v_0']), ce.LAT, ce.LON))):.4f}"
                    writer.writerow(row)
                    f.flush()
                    if verbose:
                        print(f"[{n_done}/{n_total}, {time.time() - t0:.0f}s] P_rot {P_rot:g} d, P0 {P0:g} bar, ({T_day:g}, {T_night:g}) °C: "
                              f"conv_speed {row['conv_speed']} m/s (M_star {row['M_star']}, [Fe/H] {row['feh']})")
    return path


# --- Plotting ----------------------------------------------------------------------------------------------------------

_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")  # fixed-order categorical slots
_ORDINAL_BLUE = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")  # light -> dark single-hue ramp


def _log_speed_axis(ax, speeds, errors=None):
    """Log-scaled speed axis with plain-number ticks, limits set by the data (error bars reaching zero run off the bottom)."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_yscale("log", nonpositive="clip")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
    top = np.nanmax(np.asarray(speeds) + (0 if errors is None else np.nan_to_num(np.asarray(errors))))
    ax.set_ylim(0.5 * np.nanmin(speeds), 1.3 * top)


def plot_sweep(csv_path, fig_path, speed="conv_speed", where=None):
    """Two panels from a sweep_convergence CSV: convergence speed vs P0 (one line per (T_day, T_night), coloured by ΔT) and vs ΔT
    (coloured by P0, marker by T_day). Error bars show conv_speed_std when it was computed. `where` is an optional pandas query
    selecting rows, e.g. "P_rot == 10"."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    d = pd.read_csv(csv_path)
    d = d[d.status == "ok"].copy()
    d["dT"] = d.T_day - d.T_night
    colour_P0 = dict(zip(sorted(d.P0.unique()), _CATEGORICAL * 2))  # fixed per P0 across every figure from this CSV
    cmap = LinearSegmentedColormap.from_list("dT", _ORDINAL_BLUE)
    norm = Normalize(d.dT.min(), d.dT.max())
    if where:
        d = d.query(where)
    if d.empty:
        raise ValueError(f"no solved rows in {csv_path}" + (f" matching {where!r}" if where else ""))
    err = d["conv_speed_std"] if d["conv_speed_std"].notna().any() else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    groups = list(d.sort_values("P0").groupby(["T_day", "T_night"]))
    show_err = err is not None and len(groups) <= 8  # many lines: error bars only in the right panel
    for (T_day, T_night), g in groups:
        a1.errorbar(g.P0, g[speed], yerr=g.conv_speed_std if show_err else None, color=cmap(norm(T_day - T_night)), lw=1.5, marker="o",
                    ms=6, mec="white", mew=1.2, capsize=3, elinewidth=1, alpha=0.9)
    a1.set(xscale="log", xlabel="surface pressure P0 / bar", ylabel="convergence speed / m s$^{-1}$")
    a1.set_xticks(sorted(d.P0.unique()), [f"{p:g}" for p in sorted(d.P0.unique())])
    a1.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a1, label=r"$\Delta T$ / K", pad=0.02)

    markers = dict(zip(sorted(d.T_day.unique()), "osD^vP"))
    T_days = sorted(markers)
    spacing = np.diff(sorted(d.dT.unique())).min() if d.dT.nunique() > 1 else 1.0
    dodge = {T_day: (i - (len(T_days) - 1) / 2) * 0.12 * spacing for i, T_day in enumerate(T_days)}  # small x offsets so T_day groups don't overlap
    for P0, g in d.groupby("P0"):
        c = colour_P0[P0]
        for T_day, gg in g.groupby("T_day"):
            a2.errorbar(gg.dT + dodge[T_day], gg[speed], yerr=None if err is None else gg.conv_speed_std, color=c, ls="none", marker=markers[T_day],
                        ms=8, mec="white", mew=1.5, capsize=3, elinewidth=1)
        a2.plot([], [], color=c, marker="o", ls="none", ms=8, label=f"{P0:g} bar")
    if len(markers) > 1:  # a single T_day goes in the title instead
        for T_day, mk in markers.items():
            a2.plot([], [], color="0.35", marker=mk, ls="none", ms=7, label=rf"$T_{{day}}$ = {T_day:g} °C")
    a2.set(xlabel=r"day-night contrast $\Delta T = T_{day} - T_{night}$ / K", ylabel="convergence speed / m s$^{-1}$")
    a2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a2.set_xticks(sorted(d.dT.unique()))

    P_rots = ", ".join(f"{p:g}" for p in sorted(d.P_rot.unique()))
    for ax in (a1, a2):
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        _log_speed_axis(ax, d[speed], None if err is None else d.conv_speed_std)
    T_day_text = f", T_day = {d.T_day.iloc[0]:g} °C" if d.T_day.nunique() == 1 else ""
    fig.suptitle(f"Equatorial convergence speed ({speed}), P_rot = {P_rots} d{T_day_text}, {d.GCM.iloc[0]}"
                 + ("; error bars: CLERO sample spread" if err is not None else ""))
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    import argparse

    def value_range(text):
        lo, hi, step = (float(x) for x in text.split(","))
        return list(np.round(np.arange(lo, hi + step / 2, step), 6))

    parser = argparse.ArgumentParser(description="Sweep emulator planets over (P_rot, P0, T_day, T_night) and record the convergence speed.")
    parser.add_argument("--out", required=True, help="CSV path (appended to; existing targets are skipped)")
    parser.add_argument("--P_rot", type=float, nargs="+", required=True, help="rotation periods, days")
    parser.add_argument("--P0", type=float, nargs="+", required=True, help="surface pressures, bar")
    parser.add_argument("--T_day", type=value_range, required=True, help="start,stop,step in °C (inclusive), e.g. 10,50,5")
    parser.add_argument("--T_night", type=value_range, required=True, help="start,stop,step in °C (inclusive); write --T_night=-40,20,5 when it starts negative")
    parser.add_argument("--gcm", default="exocam", choices=["exocam", "um"])
    parser.add_argument("--feh", type=float, default=None, help="stellar [Fe/H] (default: nearby M-dwarf mean)")
    parser.add_argument("--fixed_feh", action="store_true", help="don't vary [Fe/H] when there is no solution")
    parser.add_argument("--n_samples", type=int, default=0, help="CLERO draws for conv_speed_std (0 = skip)")
    parser.add_argument("--no_prefilter", action="store_true", help="solve every target, even outside the solution region")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    sweep_convergence(args.out, args.P_rot, args.P0, args.T_day, args.T_night, feh=args.feh, vary_feh=not args.fixed_feh, GCM=args.gcm,
                      n_samples=args.n_samples, prefilter=not args.no_prefilter, device=args.device)
