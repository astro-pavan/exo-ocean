"""
Analysis of the OWTG_sweep.jl runs: how the surface horizontal circulation
speed scales with the day-side forcing temperature, for the no-Coriolis
(non-rotating) ocean.

For each run we load the surface layer, project the horizontal velocity onto
the radial direction (pointing away from the substellar point, exactly as
defined in visualisation.py), and take the spatial average of its magnitude.
We then plot forcing temperature vs. average radial speed on log-log axes and
fit a power law v ~ T^n to look for scaling behaviour.

Run with (from any directory, after `pip install -r requirements.txt`):  python analysis/OWTG_analysis.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from analysis.simulation_reader import SimulationData
from analysis.paths import OCEAN_DIR, FIG_DIR

# Temperature sweep: day-side equilibrium temperature (deg C) at fixed depth.
T_DAY = [1, 3, 5, 10, 30, 50]
TEMP_SWEEP_DEPTH_KM = 1

# Depth sweep: ocean depth (km) at fixed day-side temperature.
# The 0.1 km and 30 km runs are omitted: they had numerical problems (the 30 km
# run blew up to ~10 km/s, the 0.1 km run sat ~100x below the trend).
DEPTH_KM = [0.3, 1.0, 3.0, 10.0]
DEPTH_SWEEP_TEMP = 30

# Contrast sweep: day-night temperature contrast dT (deg C) swept for a range of
# day-side temperatures, all at fixed depth (1 km). Files are named
# OWTG_T_{T}_dT_{dT}_D_1.nc.
CONTRAST_DT = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
CONTRAST_T_DAY = [1, 3, 5, 10, 30, 50]
CONTRAST_SWEEP_DEPTH_M = 1000.0   # contrast runs are all at 1 km depth

# Physical constants matching ocean_sim.jl / constants.jl, for the theoretical
# gravity-current / thermal-wind speed  v = sqrt(g * alpha * dT * H).
GRAVITY = 9.81               # m/s^2  (g_Earth)
THERMAL_EXPANSION = 2e-4     # 1/degC (LinearEquationOfState thermal_expansion)

DATA_DIR = str(OCEAN_DIR)

# Substellar point (matches ocean_sim.jl forcing and visualisation.py).
SUBSTELLAR_LON, SUBSTELLAR_LAT = 0.0, 0.0

# Average the last few snapshots to smooth out any residual transients.
TIME_SEL = slice(-3, None)

# Mean surface speeds above this (m/s) are treated as numerically blown-up runs:
# excluded from the power-law fit and flagged on the plot. Real ocean surface
# currents are ~O(1) m/s, so anything past this is unphysical.
MAX_PHYSICAL_SPEED = 10.0


def radial_velocity(u, v, lon, lat):
    """Component of the horizontal velocity pointing directly away from the
    substellar point (positive = outflow, negative = inflow). Ported from
    visualisation.py so the definition of "radial" stays consistent.

    u, v have shape (..., n_lat, n_lon); lon, lat are 1-D in degrees.
    """
    lon0 = np.radians(SUBSTELLAR_LON)
    lat0 = np.radians(SUBSTELLAR_LAT)
    lon_r = np.radians(lon)[np.newaxis, :]   # (1, n_lon)
    lat_r = np.radians(lat)[:, np.newaxis]   # (n_lat, 1)
    dlon = lon_r - lon0

    cos_rho = np.sin(lat0) * np.sin(lat_r) + np.cos(lat0) * np.cos(lat_r) * np.cos(dlon)
    sin_rho = np.sqrt(np.clip(1 - cos_rho ** 2, 0, None))
    # Undefined exactly at the substellar/antistellar point (no radial direction).
    sin_rho = np.where(sin_rho < 1e-9, np.nan, sin_rho)

    e_east = np.cos(lat0) * np.sin(dlon) / sin_rho
    e_north = (np.sin(lat_r) * np.cos(lat0) * np.cos(dlon)
               - np.sin(lat0) * np.cos(lat_r)) / sin_rho

    return u * e_east + v * e_north


def mean_surface_radial_speed(sim):
    """Area-weighted mean of |v_r| over the surface layer.

    The surface is the last z index (z-centres run bottom -> top, i.e. the
    largest / least-negative depth is the surface). Cells are weighted by
    cos(lat) since the lat-lon grid cell area shrinks towards the poles.
    """
    u_surf = sim.u[-1]   # (n_lat, n_lon)
    v_surf = sim.v[-1]
    v_r = radial_velocity(u_surf, v_surf, sim.lon, sim.lat)
    speed = np.abs(v_r)

    weights = np.cos(np.radians(sim.lat))[:, np.newaxis] * np.ones_like(speed)
    valid = np.isfinite(speed)
    return np.average(speed[valid], weights=weights[valid])


def speed_from_file(path):
    """Mean surface radial speed for a single run file (NaN if the file is
    missing so a sweep can carry on)."""
    if not os.path.exists(path):
        print(f"  [!] missing: {path}")
        return np.nan
    sim = SimulationData(path, time=TIME_SEL)
    v = mean_surface_radial_speed(sim)
    print(f"mean surface |v_r| = {v:.4e} m/s  ({os.path.basename(path)})")
    return v


def run_speed(T, D):
    """Mean surface radial speed for the run with forcing temperature T (degC)
    and ocean depth D (km)."""
    return speed_from_file(os.path.join(DATA_DIR, f"OWTG_T_{T}_D_{D}.nc"))


def loglog_powerlaw(x, y, xlabel, title, out_name, symbol):
    """Log-log scatter of x vs y with a fitted power law y = C * x^n.

    Runs whose speed exceeds MAX_PHYSICAL_SPEED are treated as numerically
    unstable: they are still plotted (as red crosses) but excluded from the fit.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    stable = y <= MAX_PHYSICAL_SPEED
    if not np.all(stable):
        bad = ", ".join(f"{symbol}={xi:g} ({yi:.3e} m/s)"
                        for xi, yi in zip(x[~stable], y[~stable]))
        print(f"  [!] excluding blown-up run(s) from fit: {bad}")

    n, logC = np.polyfit(np.log10(x[stable]), np.log10(y[stable]), 1)
    C = 10 ** logC
    print(f"\nPower-law fit (stable runs): v = {C:.3e} * {symbol}^{n:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x[stable], y[stable], "o", ms=8, color="tab:blue", label="simulations")
    if not np.all(stable):
        ax.loglog(x[~stable], y[~stable], "x", ms=10, mew=2, color="tab:red",
                  label="unstable (excluded)")

    x_fit = np.logspace(np.log10(x[stable].min()), np.log10(x[stable].max()), 100)
    ax.loglog(x_fit, C * x_fit ** n, "--", color="tab:red",
              label=f"power-law fit: $v \\propto {symbol}^{{{n:.2f}}}$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean surface radial speed $\\langle|v_r|\\rangle$ (m/s)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(FIG_DIR, out_name)
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


def contrast_sweep_plot():
    """Log-log of surface radial speed vs. day-night temperature contrast, one
    line per day-side temperature, to see whether the scaling with contrast
    depends on the absolute day-side temperature."""
    dt = np.array(CONTRAST_DT, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(CONTRAST_T_DAY)))

    for T, color in zip(CONTRAST_T_DAY, cmap):
        print(f"\n=== contrast sweep, T_day = {T} degC ===")
        speeds = np.array([
            speed_from_file(os.path.join(DATA_DIR, f"OWTG_T_{T}_dT_{d}_D_1.nc"))
            for d in CONTRAST_DT
        ])

        # Drop missing / numerically blown-up runs before fitting and plotting.
        good = np.isfinite(speeds) & (speeds <= MAX_PHYSICAL_SPEED)
        if good.sum() >= 2:
            n = np.polyfit(np.log10(dt[good]), np.log10(speeds[good]), 1)[0]
            label = f"$T_{{day}}$ = {T} °C  ($v \\propto dT^{{{n:.2f}}}$)"
        else:
            label = f"$T_{{day}}$ = {T} °C"
        print(f"  slope over contrast: {label}")

        ax.loglog(dt[good], speeds[good], "o-", color=color, label=label)

    # Theoretical gravity-current speed v = sqrt(g * alpha * dT * H), with H the
    # (fixed) ocean depth of the contrast sweep. A single curve since all runs
    # share the same depth; slope is exactly 1/2 in log-log.
    dt_fine = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 100)
    v_theory = np.sqrt(GRAVITY * THERMAL_EXPANSION * dt_fine * CONTRAST_SWEEP_DEPTH_M)
    ax.loglog(dt_fine, v_theory, "k--", lw=2,
              label=r"$v=\sqrt{g\,\alpha\,dT\,H}$ (theory)")

    ax.set_xlabel("Day-night temperature contrast $dT$ (°C)")
    ax.set_ylabel("Mean surface radial speed $\\langle|v_r|\\rangle$ (m/s)")
    ax.set_title("Non-rotating ocean: surface speed vs. temperature contrast")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "OWTG_contrast_vs_velocity.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


def main():
    # --- Temperature sweep (fixed depth) ---
    temps = list(T_DAY)
    temp_speeds = [run_speed(T, TEMP_SWEEP_DEPTH_KM) for T in T_DAY]
    loglog_powerlaw(
        temps, temp_speeds,
        xlabel="Day-side forcing temperature $T$ (°C)",
        title="Non-rotating ocean: surface outflow speed vs. forcing temperature",
        out_name="OWTG_temp_vs_velocity.png",
        symbol="T",
    )

    # --- Depth sweep (fixed temperature) ---
    depths = list(DEPTH_KM)
    depth_speeds = [run_speed(DEPTH_SWEEP_TEMP, D) for D in DEPTH_KM]
    loglog_powerlaw(
        depths, depth_speeds,
        xlabel="Ocean depth $D$ (km)",
        title="Non-rotating ocean: surface outflow speed vs. ocean depth",
        out_name="OWTG_depth_vs_velocity.png",
        symbol="D",
    )

    # --- Contrast sweep (per day-side temperature) ---
    contrast_sweep_plot()


if __name__ == "__main__":
    main()
