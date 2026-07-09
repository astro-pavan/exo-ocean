"""
Equatorial-jet diagnostics for the Coriolis sweep
(coriolis_P_{period}_dT_{contrast}_D_1.nc), varying planetary rotation period P
(days) and day-night temperature contrast dT (degC) at fixed depth H = 1 km.

With rotation the substellar-driven overturning is deflected into a zonal
current on the equator. From the surface zonal-mean zonal velocity u_zonal(lat):

  * jet speed : peak eastward u_zonal within a narrow equatorial band
                (|lat| < 2.5 deg) -- i.e. the strength of the flow at the equator.
  * jet width : FWHM of u_zonal about that peak (interpolated half-max crossings).

We then plot speed and width vs. dT and vs. P on log-log axes to look for power
laws, and overlay the theoretical equatorial Rossby radius of deformation on the
width-vs-P panel:

    L_eq = sqrt(c / beta),   c = sqrt(g' H),  g' = g*alpha*dT,  beta = 2*Omega/R,
    Omega = omega_Earth / P.

So L_eq ~ P^{1/2} dT^{1/4}; it is a meridional length scale (half-width-like),
converted to degrees for comparison with the measured FWHM.

Run with:  /data/pt426/big-venv/bin/python coriolis_jet_analysis.py
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from simulation_reader import SimulationData

ROT_PERIOD = [3, 5, 10, 30, 50]                 # days
CONTRAST_DT = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]  # degC
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "ocean")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TIME_SEL = slice(-3, None)   # average last few snapshots
SPEED_BAND = 2.5             # deg; equatorial jet speed = peak u within this band

# Data-quality thresholds.
SPINUP_MIN_DAYS = 500.0      # runs shorter than this aren't spun up -> excluded
JET_MIN_SPEED = 0.01         # m/s; below this there is no coherent equatorial jet

# Physical constants (match constants.jl / ocean_sim.jl).
G = 9.81
ALPHA = 2e-4
OMEGA_EARTH = 7.29e-5
R_PLANET = 6.371e6
DEPTH_M = 1000.0
DEG_PER_M = 180.0 / (np.pi * R_PLANET)   # metres -> degrees latitude


def _half_max_crossing(lat, u, i_peak, half, step):
    """Latitude where u falls to `half` moving away from i_peak in direction
    `step` (+1 north, -1 south), linearly interpolated; None if never reached."""
    i = i_peak
    while 0 <= i + step < len(u):
        if u[i + step] < half:
            u0, u1 = u[i], u[i + step]
            f = (u0 - half) / (u0 - u1)
            return lat[i] + f * (lat[i + step] - lat[i])
        i += step
    return None


def jet_speed_width(sim):
    """Return (speed, width_deg, lat_peak) for the equatorial jet."""
    lat = sim.lat
    u_zonal = sim.u[-1].mean(axis=1)             # surface zonal mean -> u(lat)

    band = np.abs(lat) <= SPEED_BAND
    lat_b, u_b = lat[band], u_zonal[band]
    i_local = int(np.argmax(u_b))                # peak eastward flow at the equator
    speed = float(u_b[i_local])
    lat_peak = float(lat_b[i_local])
    i_peak = int(np.where(lat == lat_peak)[0][0])

    half = speed / 2.0
    north = _half_max_crossing(lat, u_zonal, i_peak, half, +1)
    south = _half_max_crossing(lat, u_zonal, i_peak, half, -1)
    width = np.nan if (north is None or south is None) else north - south
    return speed, width, lat_peak


def rossby_deg(P, dT):
    """Equatorial Rossby radius of deformation (degrees latitude)."""
    Omega = OMEGA_EARTH / P
    beta = 2 * Omega / R_PLANET
    c = np.sqrt(G * ALPHA * dT * DEPTH_M)        # reduced-gravity wave speed
    L_eq = np.sqrt(c / beta)                     # metres
    return L_eq * DEG_PER_M


def load(P, dT):
    path = os.path.join(DATA_DIR, f"coriolis_P_{P}_dT_{dT}_D_1.nc")
    if not os.path.exists(path):
        print(f"  [!] missing: {os.path.basename(path)}")
        return None, np.nan
    with xr.open_dataset(path, decode_times=False) as ds:
        units = ds["time"].attrs.get("units", "seconds")
        spu = 3600 if units.startswith("hours") else 1
        t_final_days = float(np.array(ds["time"])[-1]) * spu / 86400.0
    return SimulationData(path, time=TIME_SEL), t_final_days


def fit_slope(x, y, mask):
    """Power-law exponent of y vs x over masked points (NaN if <2 points)."""
    m = mask & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0]


def collect():
    nP, nD = len(ROT_PERIOD), len(CONTRAST_DT)
    speed = np.full((nP, nD), np.nan)
    width = np.full((nP, nD), np.nan)
    spun = np.zeros((nP, nD), dtype=bool)

    print(f"\n{'P (d)':>6} {'dT':>6} {'speed':>10} {'width':>9} {'lat_pk':>7} {'t_fin(d)':>9}")
    for i, P in enumerate(ROT_PERIOD):
        for j, dT in enumerate(CONTRAST_DT):
            sim, t_final = load(P, dT)
            if sim is None:
                continue
            s, w, lat_pk = jet_speed_width(sim)
            speed[i, j], width[i, j] = s, w
            spun[i, j] = t_final >= SPINUP_MIN_DAYS
            flag = "" if spun[i, j] else "  <-- not spun up"
            print(f"{P:>6} {dT:>6} {s:>10.4f} {w:>9.2f} {lat_pk:>7.1f} {t_final:>9.0f}{flag}")
    return speed, width, spun


def main():
    speed, width, spun = collect()
    P = np.array(ROT_PERIOD, float)
    dT = np.array(CONTRAST_DT, float)

    # Only spun-up runs with a coherent equatorial jet enter the plots/fits;
    # at low contrast / fast rotation the equatorial flow is ~0 (no jet).
    jet_present = spun & (speed >= JET_MIN_SPEED)
    width_ok = jet_present & np.isfinite(width)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    cP = plt.cm.viridis(np.linspace(0, 0.9, len(ROT_PERIOD)))
    cD = plt.cm.plasma(np.linspace(0, 0.9, len(CONTRAST_DT)))

    # ---- (A) speed vs dT, one line per P ----
    ax = axes[0, 0]
    for i, Pi in enumerate(ROT_PERIOD):
        m = jet_present[i, :]
        if m.sum() == 0:
            continue
        n = fit_slope(dT, speed[i, :], m)
        lbl = f"P = {Pi} d" + (f"  ($\\propto dT^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        ax.loglog(dT[m], speed[i, m], "o-", color=cP[i], label=lbl)
    ax.set_xlabel("Temperature contrast $dT$ (°C)")
    ax.set_ylabel("Jet speed (m/s)")
    ax.set_title("(A) Jet speed vs. contrast")

    # ---- (B) speed vs P, one line per dT ----
    ax = axes[0, 1]
    for j, dTj in enumerate(CONTRAST_DT):
        m = jet_present[:, j]
        if m.sum() == 0:
            continue
        n = fit_slope(P, speed[:, j], m)
        lbl = f"dT = {dTj} °C" + (f"  ($\\propto P^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        ax.loglog(P[m], speed[m, j], "o-", color=cD[j], label=lbl)
    ax.set_xlabel("Rotation period $P$ (days)")
    ax.set_ylabel("Jet speed (m/s)")
    ax.set_title("(B) Jet speed vs. rotation period")

    # ---- (C) width vs dT, one line per P ----
    ax = axes[1, 0]
    for i, Pi in enumerate(ROT_PERIOD):
        m = width_ok[i, :]
        if m.sum() == 0:
            continue
        n = fit_slope(dT, width[i, :], m)
        lbl = f"P = {Pi} d" + (f"  ($\\propto dT^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        ax.loglog(dT[m], width[i, m], "o-", color=cP[i], label=lbl)
    ax.set_xlabel("Temperature contrast $dT$ (°C)")
    ax.set_ylabel("Jet width, FWHM (°)")
    ax.set_title("(C) Jet width vs. contrast")

    # ---- (D) width vs P, one line per dT, with Rossby radius overlaid ----
    ax = axes[1, 1]
    for j, dTj in enumerate(CONTRAST_DT):
        m = width_ok[:, j]
        if m.sum() == 0:
            continue
        # theoretical equatorial deformation radius for this dT (only where we
        # have data to compare against), same colour as the measured line.
        ax.loglog(P, [rossby_deg(Pi, dTj) for Pi in P], ":", color=cD[j], lw=1.5)
        n = fit_slope(P, width[:, j], m)
        lbl = f"dT = {dTj} °C" + (f"  ($\\propto P^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        ax.loglog(P[m], width[m, j], "o-", color=cD[j], label=lbl)
    ax.plot([], [], "k:", label="$L_{eq}=\\sqrt{c/\\beta}$ (theory)")
    ax.set_xlabel("Rotation period $P$ (days)")
    ax.set_ylabel("Jet width, FWHM (°)")
    ax.set_title("(D) Jet width vs. rotation period + Rossby radius")

    for ax in axes.flat:
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle("Equatorial jet: Coriolis sweep (spun-up runs only)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "coriolis_jet_sweep.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved sweep plot to {out}")


if __name__ == "__main__":
    main()
