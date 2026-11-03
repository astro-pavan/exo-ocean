"""Does a 1-3 parameter purely converging wind (the axisymmetric χ modes in tidally locked coordinates) give the ocean-relevant stress?

Metrics, over the CLERO sweep planets, using the ocean model's bulk stress (src/ocean/emulator_wind.jl: Large & Yeager C_D at the
equivalent neutral 10 m wind):
  tau_conv   equatorial (|lat| <= 10°) mean zonal stress towards the substellar point, -τx sign(sin λ): opposes both ocean jets
  tau_prof   RMS error of the equatorial zonal-stress profile τx(λ), relative to its RMS
  ustar3     ocean-area (|lat| <= 80°) mean u*³ = (|τ|/ρ_w)^{3/2}, the wind's TKE input to mixing
  ustar3_map RMS error of the u*³ map, relative to its RMS

    /data/pt426/ocean-venv/bin/python experiments/tests/converging_wind_test.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "climate"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import climate_emulator as ce
from spherical_harmonic_winds import decompose_winds, to_tidally_locked, reconstruct_winds
from wind_decomposition_test import emulate_sweep_fields

FIGURE = os.path.join(ROOT, "figures", "converging_wind_test.png")
RHO_SEAWATER, G, LOWEST_LEVEL, VON_KARMAN = 1026.0, 9.81, 0.925, 0.4  # as in src/ocean (emulator_wind.jl, ocean_sim.jl)
EQ_LAT, OCEAN_LAT = 10.0, 80.0
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")  # as in src/climate/wind_scaling.py


def large_yeager_cd(U):
    U = np.maximum(U, 0.5)
    return 2.7e-3 / U + 1.42e-4 + 7.64e-5 * U


def bulk_stress(u, v, rho, P0):
    """(τx, τy) in N/m^2 from the lowest-level wind, as emulator_wind_stress: C_D(U₁₀) ρ U₁₀² along the wind (sign: stress on the ocean)."""
    W = np.hypot(u, v)
    z = (P0 * 1e5 / (rho * G) * np.log(1 / LOWEST_LEVEL))
    U10 = W.copy()
    for _ in range(5):
        U10 = W / (1 + np.sqrt(large_yeager_cd(U10)) / VON_KARMAN * np.log(z / 10))
    t = rho * large_yeager_cd(U10) * U10 ** 2 / np.where(W > 0, W, 1)
    return t * u, t * v


def ocean_metrics(u, v, rho, P0, lat=ce.LAT, lon=ce.LON):
    tx, ty = bulk_stress(u, v, rho, P0[:, None, None])
    coslat = np.cos(np.radians(lat))
    eq = (np.abs(lat) <= EQ_LAT)
    oc = (np.abs(lat) <= OCEAN_LAT)
    tx_eq = (tx[:, eq] * coslat[eq, None]).sum(1) / coslat[eq].sum()  # (planet, lon) equatorial-band zonal stress
    ustar3 = (np.hypot(tx, ty) / RHO_SEAWATER) ** 1.5
    w = (coslat * oc)[:, None]
    return {"tau_conv": (-tx_eq * np.sign(np.sin(np.radians(lon)))).mean(-1),
            "tx_eq": tx_eq,
            "ustar3": (ustar3 * w).sum((1, 2)) / (w.sum() * lon.size),
            "ustar3_field": ustar3}


def model_errors(ref, mod):
    rel = lambda a, b: (b - a) / a
    rms = lambda a, b, axes: np.sqrt((b - a) ** 2).mean(axes) if False else np.sqrt(((b - a) ** 2).mean(axes) / (a ** 2).mean(axes))
    oc = (np.abs(ce.LAT) <= OCEAN_LAT)
    return {"tau_conv": rel(ref["tau_conv"], mod["tau_conv"]),
            "tau_prof": rms(ref["tx_eq"], mod["tx_eq"], -1),
            "ustar3": rel(ref["ustar3"], mod["ustar3"]),
            "ustar3_map": rms(ref["ustar3_field"][:, oc], mod["ustar3_field"][:, oc], (-2, -1))}


# Leading tidally locked ψ terms by median energy over the sweep; with x = cos φ cos λ, y = cos φ sin λ, z = sin φ they are
# ψ(1,1) ∝ z, ψ(1,2) ∝ xz, ψ(2,2) ∝ yz, ψ(3,3) ∝ z(3 - 3x² - 4z²)
LEADING_PSI = [(3, 3), (2, 2), (1, 2), (1, 1)]


def converging_model(t, N, with_rotational=False, n_psi=0):
    """Keep the tidally locked χ modes m = 0, n = 1..N (N real parameters), plus the full rotational part or its n_psi leading terms."""
    keep = (t["m"] == 0) & (t["n"] <= N)
    keep_psi = np.zeros_like(keep)
    for m, n in LEADING_PSI[:n_psi]:
        keep_psi |= (t["m"] == m) & (t["n"] == n)
    psi = t["psi"] if with_rotational else 1j * np.where(keep_psi, t["psi"].imag, 0)
    return t | {"chi": np.where(keep, t["chi"].real, 0) + 0j, "psi": psi}


def power_law(y, f):
    """Least-squares exponents of y ∝ ΔT^a P0^b P_rot^c (T_day dependence negligible in the sweep) and the RMS ln residual."""
    dT = f["T_day"] - f["T_night"]
    X = np.column_stack([np.ones(len(y)), np.log(dT), np.log(f["P0"]), np.log(f["P_rot"])])
    c, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    return c, np.std(np.log(y) - X @ c)


def main():
    f = emulate_sweep_fields()
    t = to_tidally_locked(decompose_winds(f["u"], f["v"], ce.LAT, ce.LON))
    ref = ocean_metrics(f["u"], f["v"], f["rho"], f["P0"])
    print(f"{len(f['u'])} planets. Emulator: tau_conv median {np.median(ref['tau_conv']):.3f} N/m^2 "
          f"[{np.percentile(ref['tau_conv'], 10):.3f}-{np.percentile(ref['tau_conv'], 90):.3f}], "
          f"ustar3 median {np.median(ref['ustar3']):.2e} m^3/s^3")
    for k in ("tau_conv", "ustar3"):
        c, r = power_law(ref[k], f)
        print(f"  {k} ∝ ΔT^{c[1]:.2f} P0^{c[2]:.2f} P_rot^{c[3]:.2f}  (rms ln residual {r:.2f})")

    models = {f"converging, {N} param": converging_model(t, N) for N in (1, 2, 3, 4)}
    models |= {f"converging 3 + {k} leading ψ": converging_model(t, 3, n_psi=k) for k in (2, 4)}
    models["converging 3 + full rotational"] = converging_model(t, 3, with_rotational=True)
    errs = {}
    print("\nmodel errors: median [10th-90th pct] over planets (tau_conv, ustar3 signed relative; profiles/maps relative RMS)")
    for name, h in models.items():
        u, v = reconstruct_winds(h)
        errs[name] = model_errors(ref, ocean_metrics(u, v, f["rho"], f["P0"]))
        print(f"  {name}:\n    " + "  ".join(f"{k} {np.median(e):+.3f} [{np.percentile(e, 10):+.3f}, {np.percentile(e, 90):+.3f}]"
                                         for k, e in errs[name].items()))
    for name in ("converging, 3 param", "converging 3 + 4 leading ψ"):
        print(f"\n{name}, |error| median by P_rot:")
        for p in np.unique(f["P_rot"]):
            s = f["P_rot"] == p
            print(f"  P_rot {p:5g} d: " + "  ".join(f"{k} {np.median(np.abs(e[s])):.3f}" for k, e in errs[name].items()))

    # Figure: equatorial zonal stress vs longitude for example planets, emulator vs 1- and 3-parameter converging models
    examples = [np.flatnonzero((f["P_rot"] == p) & (f["P0"] == 3))[0] for p in (6, 25, 100) if ((f["P_rot"] == p) & (f["P0"] == 3)).any()]
    fig, axes = plt.subplots(1, len(examples) + 1, figsize=(4.2 * (len(examples) + 1), 3.6))
    tx_mod = {N: ocean_metrics(*reconstruct_winds(converging_model(t, N)), f["rho"], f["P0"])["tx_eq"] for N in (1, 3)}
    for ax, i in zip(axes, examples):
        ax.axhline(0, color="#9a9a9a", lw=0.8)
        ax.plot(ce.LON, ref["tx_eq"][i], color="#333333", lw=2, label="emulator")
        for (N, txm), c in zip(tx_mod.items(), CATEGORICAL):
            ax.plot(ce.LON, txm[i], color=c, lw=1.8, ls="--" if N == 1 else "-", label=f"converging, {N} param")
        ax.set(xlabel="longitude from substellar / °", title=f"P_rot {f['P_rot'][i]:g} d, P0 {f['P0'][i]:g} bar, ΔT {f['T_day'][i] - f['T_night'][i]:g} K",
               xticks=[-180, -90, 0, 90, 180])
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel(f"zonal stress, |lat| ≤ {EQ_LAT:g}° / N m$^{{-2}}$")
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[-1]
    for N, c in zip((1, 2, 3), CATEGORICAL):
        e = errs[f"converging, {N} param"]
        ax.scatter(ref["tau_conv"], ref["tau_conv"] * (1 + e["tau_conv"]), s=10, color=c, alpha=0.6, label=f"{N} param")
    lim = [0, ref["tau_conv"].max() * 1.1]
    ax.plot(lim, lim, color="#9a9a9a", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="emulator τ toward substellar / N m$^{-2}$", ylabel="converging model", title="equatorial convergent stress")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    print(f"figure -> {os.path.normpath(FIGURE)}")


if __name__ == "__main__":
    main()
