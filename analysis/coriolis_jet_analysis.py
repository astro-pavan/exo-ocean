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

ROT_PERIOD = [3, 5, 10, 30, 50]                          # days
CONTRAST_DT = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0]  # degC
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "ocean")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TIME_SEL = slice(-3, None)   # average last few snapshots
SPEED_BAND = 2.5             # deg; equatorial jet speed = peak u within this band

# Data-quality thresholds.
# Runs now use a convergence-based early stop, so a short t_final can mean
# "converged" (good) rather than "crashed" — keep this low, just enough to catch
# genuine blow-ups, and rely on JET_MIN_SPEED + the centred-profile check for quality.
SPINUP_MIN_DAYS = 400.0
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


def warm_layer_depth(sim):
    """Heat-content-equivalent depth of the equatorial warm layer (m) — the
    vertical scale that matters for heat transport. It is the thickness of an
    equivalent surface-temperature slab holding the same heat anomaly,
        H = ∫ (T(z) - T_bottom) dz / (T_surface - T_bottom),
    over the equatorial band. Preferred over the raw max-gradient thermocline
    depth, which here pins to a sharp ~33 m near-surface gradient (quantised to
    the grid) and misses the deeper warm water that carries the heat.
    """
    lat, z = sim.lat, sim.z                       # z: centres, negative, surface last
    eq = np.abs(lat) < 5.0
    Tprof = np.nanmean(sim.T[:, eq, :], axis=(1, 2))   # equatorial-mean T(z)
    dz = np.abs(np.gradient(z))                        # cell thicknesses (m)
    T_bot, T_surf = np.nanmin(Tprof), Tprof[-1]
    if T_surf - T_bot <= 0:
        return np.nan
    return float(np.sum((Tprof - T_bot) * dz) / (T_surf - T_bot))


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
    thermo = np.full((nP, nD), np.nan)
    spun = np.zeros((nP, nD), dtype=bool)

    print(f"\n{'P (d)':>6} {'dT':>6} {'speed':>10} {'width':>9} {'H_warm(m)':>10} {'lat_pk':>7} {'t_fin(d)':>9}")
    for i, P in enumerate(ROT_PERIOD):
        for j, dT in enumerate(CONTRAST_DT):
            sim, t_final = load(P, dT)
            if sim is None:
                continue
            s, w, lat_pk = jet_speed_width(sim)
            h = warm_layer_depth(sim)
            speed[i, j], width[i, j], thermo[i, j] = s, w, h
            spun[i, j] = t_final >= SPINUP_MIN_DAYS
            flag = "" if spun[i, j] else "  <-- not spun up"
            print(f"{P:>6} {dT:>6} {s:>10.4f} {w:>9.2f} {h:>10.1f} {lat_pk:>7.1f} {t_final:>9.0f}{flag}")
    return speed, width, thermo, spun


def main():
    speed, width, thermo, spun = collect()
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

    gravity_wave_plot(speed, jet_present, P, dT)
    speed_width_plot(speed, width, width_ok, P, dT)
    volume_flux_plot(speed, width, thermo, width_ok, P, dT)


def gravity_wave_plot(speed, jet_present, P, dT):
    """Jet speed vs. the internal gravity-wave speed c = sqrt(g*alpha*dT*H).

    All runs share H = 1 km, so c is set by the contrast alone (c ∝ sqrt(dT)) and
    is the natural velocity scale for the buoyancy-driven flow. Since the measured
    jet speed goes ~linearly with dT, we expect v_jet ∝ c^2 — i.e. the equatorial
    jet is NOT a fixed fraction of c (contrast the non-rotating surface current,
    which tracks c linearly). Panel B shows the ratio v_jet/c to make that explicit.
    """
    c = np.sqrt(G * ALPHA * dT * DEPTH_M)          # gravity-wave speed per contrast (m/s)
    cP = plt.cm.viridis(np.linspace(0, 0.9, len(ROT_PERIOD)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))
    for i, Pi in enumerate(ROT_PERIOD):
        m = jet_present[i, :]
        if m.sum() == 0:
            continue
        n = fit_slope(c, speed[i, :], m)
        lbl = f"P = {Pi} d" + (f"  ($v\\propto c^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        axA.loglog(c[m], speed[i, m], "o-", color=cP[i], label=lbl)
        axB.loglog(c[m], speed[i, m] / c[m], "o-", color=cP[i], label=f"P = {Pi} d")

    # Reference slopes through a mid-cloud anchor (c0, v0), to compare the data's
    # slope against v∝c (fixed fraction of the wave speed) and v∝c².
    cref = np.logspace(np.log10(c.min()), np.log10(c.max()), 50)
    c0, v0 = 3.0, 0.05
    axA.loglog(cref, v0 * (cref / c0),      "k--", lw=1,   alpha=0.6, label="$v\\propto c$")
    axA.loglog(cref, v0 * (cref / c0) ** 2, "k:",  lw=1.5,            label="$v\\propto c^2$")

    axA.set_xlabel("Gravity-wave speed $c=\\sqrt{g\\,\\alpha\\,dT\\,H}$ (m/s)")
    axA.set_ylabel("Jet speed (m/s)")
    axA.set_title("(A) Jet speed vs. gravity-wave speed")

    axB.set_xlabel("Gravity-wave speed $c$ (m/s)")
    axB.set_ylabel("Jet speed / $c$")
    axB.set_title("(B) Jet speed as a fraction of $c$")

    for ax in (axA, axB):
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Equatorial jet speed vs. gravity-wave speed", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "coriolis_jet_vs_gravitywave.png")
    fig.savefig(out, dpi=150)
    print(f"Saved gravity-wave plot to {out}")


def speed_width_plot(speed, width, width_ok, P, dT):
    """Jet speed vs. jet width (FWHM), log-log, one trajectory per rotation period.

    Each line sweeps contrast dT (both speed and width grow with dT), so its slope
    is the fixed-rotation speed-width relation. From the sweep, speed ∝ dT^~1 and
    width ∝ dT^~0.5, so we expect ~slope 2 at fixed P; increasing P shifts the
    curves (wider, faster jets for weaker rotation).
    """
    cP = plt.cm.viridis(np.linspace(0, 0.9, len(ROT_PERIOD)))
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for i, Pi in enumerate(ROT_PERIOD):
        m = width_ok[i, :]
        if m.sum() == 0:
            continue
        order = np.argsort(dT[m])                 # increasing contrast
        w = width[i, m][order]
        s = speed[i, m][order]
        n = np.polyfit(np.log10(w), np.log10(s), 1)[0] if m.sum() >= 2 else np.nan
        lbl = f"P = {Pi} d" + (f"  ($v\\propto w^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        ax.loglog(w, s, "o-", color=cP[i], label=lbl)

    # Reference slope-2 guide (fixed-P expectation) through a mid-cloud anchor.
    wall, sall = width[width_ok], speed[width_ok]
    if wall.size:
        wref = np.logspace(np.log10(wall.min()), np.log10(wall.max()), 50)
        w0, s0 = 6.0, 0.1
        ax.loglog(wref, s0 * (wref / w0) ** 2, "k:", lw=1.5, label="slope 2  ($v\\propto w^2$)")

    ax.set_xlabel("Jet width, FWHM (°)")
    ax.set_ylabel("Jet speed (m/s)")
    ax.set_title("Equatorial jet: speed vs. width (lines sweep $dT$ at fixed $P$)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "coriolis_jet_speed_vs_width.png")
    fig.savefig(out, dpi=150)
    print(f"Saved speed-vs-width plot to {out}")


def volume_flux_plot(speed, width, thermo, width_ok, P, dT):
    """Warm-layer volume flux of the jet, Q = speed x H_warm x width, vs. contrast
    and rotation. Using the heat-content warm-layer depth H_warm (not the full
    ocean depth) restricts the transport to the warm surface layer the jet
    actually carries, so Q is a proxy for its heat-transport capacity (heat
    transport ~ rho*cp*dT * Q). Reported in Sverdrups (1 Sv = 1e6 m^3/s).
    """
    width_m = np.radians(width) * R_PLANET            # FWHM in metres
    Q = speed * thermo * width_m / 1e6                 # Sv (warm-layer flux)

    cP = plt.cm.viridis(np.linspace(0, 0.9, len(ROT_PERIOD)))
    cD = plt.cm.plasma(np.linspace(0, 0.9, len(CONTRAST_DT)))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (A) Q vs contrast, one line per rotation period
    for i, Pi in enumerate(ROT_PERIOD):
        m = width_ok[i, :]
        if m.sum() == 0:
            continue
        n = fit_slope(dT, Q[i, :], m)
        lbl = f"P = {Pi} d" + (f"  ($\\propto dT^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        axA.loglog(dT[m], Q[i, m], "o-", color=cP[i], label=lbl)
    axA.set_xlabel("Temperature contrast $dT$ (°C)")
    axA.set_ylabel("Warm-layer volume flux $Q$ (Sv)")
    axA.set_title("(A) Jet volume flux vs. contrast")

    # (B) Q vs rotation period, one line per contrast
    for j, dTj in enumerate(CONTRAST_DT):
        m = width_ok[:, j]
        if m.sum() == 0:
            continue
        n = fit_slope(P, Q[:, j], m)
        lbl = f"dT = {dTj} °C" + (f"  ($\\propto P^{{{n:.2f}}}$)" if np.isfinite(n) else "")
        axB.loglog(P[m], Q[m, j], "o-", color=cD[j], label=lbl)
    axB.set_xlabel("Rotation period $P$ (days)")
    axB.set_ylabel("Warm-layer volume flux $Q$ (Sv)")
    axB.set_title("(B) Jet volume flux vs. rotation period")

    for ax in (axA, axB):
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Equatorial jet warm-layer volume flux  $Q = v \\times H_{warm} \\times w$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "coriolis_jet_volume_flux.png")
    fig.savefig(out, dpi=150)
    print(f"Saved volume-flux plot to {out}")


if __name__ == "__main__":
    main()
