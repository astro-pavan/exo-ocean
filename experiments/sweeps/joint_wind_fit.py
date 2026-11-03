"""Joint fit of parameterise_winds.parametric_wind with its parameters as functions of (ΔT, P0, P_rot), across the space-filling sweep.

With L = ln(ΔT / 40 K), p = P0 / 2 bar and the regime switch s(P_rot) = 1 / (1 + (P_rot / P_t)^n) (1 fast, 0 slow):
    ln U_p = ln U_0 + (b_U + db_U s) L + q_U L² + c_U ln p
    θ_E    = [θ_E,slow + (θ_E,fast - θ_E,slow) s] p^(c_E + dc_E s)
    θ_W    = [θ_W,slow + (θ_W,fast - θ_W,slow) s] (ΔT / 40 K)^d_W
    f      = f_slow + (f_fast - f_slow) s + (e_f + de_f s) L
    k_E    = k_E + k_E_bump exp(-ln²(P_rot / P_k) / 2 w_k²)
    k_W    = k_W p^c_kW
    u0 = 0; T_day is not used.
Variants free different coefficients; the rest stay at their "off" values (0):
    simple   16 coefficients
    slopes   + regime-dependent slopes db_U, dc_E, de_f                     (19)
    shapes   + curvature q_U, k_E bump in P_rot, k_W ∝ P0^c_kW              (21)
    flexible + both                                                         (24)
The coefficients minimise the mean squared relative stress misfit (each planet weighted equally) on a random half of the sweep, starting
from the simple fit when it exists; the other half is held out and scored with experiments/tests/converging_wind_test.py's metrics.

    /data/pt426/ocean-venv/bin/python experiments/sweeps/joint_wind_fit.py [--variant simple|slopes|shapes|flexible]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "climate"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "tests"))
import climate_emulator as ce
from parameterise_winds import parametric_wind, PARAM_BOUNDS
from converging_wind_test import ocean_metrics, model_errors

FIELDS = os.path.join(ROOT, "output", "winds", "parametric_sweep_fields.npz")
FITS = os.path.join(ROOT, "output", "winds", "parametric_sweep_fits.npz")
OUT = os.path.join(ROOT, "output", "winds", "joint_wind_fit_{}.json")
OCEAN_LAT = 80.0

# name: (initial value, lower bound, upper bound); angles in radians. Initial values of the simple set come from the per-planet fits;
# the extra coefficients start "off".
COEFFS = {
    "U_0": (10.0, 1.0, 50.0), "b_U": (0.3, -2.0, 2.0), "c_U": (-0.34, -2.0, 2.0),
    "P_t": (12.0, 2.0, 100.0), "n": (3.0, 0.5, 20.0),
    "theta_E_slow": (0.45, 0.02, 2.5), "theta_E_fast": (1.3, 0.02, 2.5), "c_E": (-0.4, -2.0, 2.0),
    "theta_W_slow": (0.48, 0.02, 2.5), "theta_W_fast": (0.17, 0.02, 2.5), "d_W": (-0.6, -3.0, 3.0),
    "f_slow": (0.18, -1.0, 3.0), "f_fast": (0.85, -1.0, 3.0), "e_f": (-0.2, -3.0, 3.0),
    "k_E": (1.0, 0.0, 50.0), "k_W": (1.0, 0.0, 50.0),
    # option 1: regime-dependent slopes
    "db_U": (0.0, -2.0, 2.0), "dc_E": (0.0, -2.0, 2.0), "de_f": (0.0, -3.0, 3.0),
    # option 3: curvature and extra shape
    "q_U": (0.0, -2.0, 2.0), "k_E_bump": (0.0, -5.0, 20.0), "P_k": (20.0, 4.0, 120.0), "w_k": (0.7, 0.2, 3.0), "c_kW": (0.0, -3.0, 3.0),
}
SIMPLE = ["U_0", "b_U", "c_U", "P_t", "n", "theta_E_slow", "theta_E_fast", "c_E", "theta_W_slow", "theta_W_fast", "d_W",
          "f_slow", "f_fast", "e_f", "k_E", "k_W"]
SLOPES = ["db_U", "dc_E", "de_f"]
SHAPES = ["q_U", "k_E_bump", "P_k", "w_k", "c_kW"]
VARIANTS = {"simple": SIMPLE, "slopes": SIMPLE + SLOPES, "shapes": SIMPLE + SHAPES, "flexible": SIMPLE + SLOPES + SHAPES}


def wind_parameters(c, dT, P0, P_rot):
    """The 7 parametric_wind parameters, shape (7, n), from coefficient dict c (missing ones off) and ΔT (K), P0 (bar), P_rot (d)."""
    c = {k: COEFFS[k][0] for k in SLOPES + SHAPES} | c  # coefficients not in the variant stay off
    L, lnp = np.log(dT / 40), np.log(P0 / 2)
    s = 1 / (1 + (P_rot / c["P_t"]) ** c["n"])
    U_p = c["U_0"] * np.exp((c["b_U"] + c["db_U"] * s) * L + c["q_U"] * L ** 2 + c["c_U"] * lnp)
    theta_E = (c["theta_E_slow"] + (c["theta_E_fast"] - c["theta_E_slow"]) * s) * np.exp((c["c_E"] + c["dc_E"] * s) * lnp)
    theta_W = (c["theta_W_slow"] + (c["theta_W_fast"] - c["theta_W_slow"]) * s) * (dT / 40) ** c["d_W"]
    f = c["f_slow"] + (c["f_fast"] - c["f_slow"]) * s + (c["e_f"] + c["de_f"] * s) * L
    k_E = c["k_E"] + c["k_E_bump"] * np.exp(-np.log(P_rot / c["P_k"]) ** 2 / (2 * c["w_k"] ** 2))
    k_W = c["k_W"] * np.exp(c["c_kW"] * lnp)
    params = np.stack([U_p, theta_E, theta_W, k_E, k_W, f, 0 * U_p])
    return np.clip(params, np.array(PARAM_BOUNDS[0])[:, None], np.array(PARAM_BOUNDS[1])[:, None])


def batch_wind(params):
    """parametric_wind for many planets at once: params (7, n) -> u, v of shape (n, lat, lon)."""
    return parametric_wind(params[:, :, None, None], ce.LAT, ce.LON)


class StressCost:
    """Mean over planets of the squared relative RMS stress misfit, ρ|U|U over |lat| <= OCEAN_LAT (C_D cancels)."""

    def __init__(self, f, idx, names):
        rows = np.abs(ce.LAT) <= OCEAN_LAT
        self.w = np.cos(np.radians(ce.LAT[rows]))[None, :, None]
        self.rows = rows
        self.rho = f["rho"][idx][:, rows].astype(float)
        self.tx, self.ty = self.stress(f["u"][idx][:, rows].astype(float), f["v"][idx][:, rows].astype(float))
        self.norm = (self.w * (self.tx ** 2 + self.ty ** 2)).sum((1, 2))
        self.inputs = tuple(np.asarray(f[k])[idx] for k in ("dT", "P0", "P_rot"))
        self.names = names

    def stress(self, u, v):
        speed = np.hypot(u, v)
        return self.rho * speed * u, self.rho * speed * v

    def __call__(self, x):
        u, v = batch_wind(wind_parameters(dict(zip(self.names, x)), *self.inputs))
        tx, ty = self.stress(u[:, self.rows], v[:, self.rows])
        return np.mean((self.w * ((tx - self.tx) ** 2 + (ty - self.ty) ** 2)).sum((1, 2)) / self.norm)


def score(name, f, idx, params):
    """Ocean-relevant errors of the winds from params (7, n) for planets idx, printed overall and by P_rot group."""
    ref = ocean_metrics(f["u"][idx].astype(float), f["v"][idx].astype(float), f["rho"][idx].astype(float), f["P0"][idx])
    u, v = batch_wind(params)
    e = model_errors(ref, ocean_metrics(u, v, f["rho"][idx].astype(float), f["P0"][idx]))
    print(f"  {name}: " + "  ".join(f"{k} {np.median(x):+.3f} [{np.percentile(x, 10):+.2f}, {np.percentile(x, 90):+.2f}]" for k, x in e.items()))
    for a, b in zip((4, 8, 15, 30, 60), (8, 15, 30, 60, 120)):
        s = (f["P_rot"][idx] >= a) & (f["P_rot"][idx] < b)
        print(f"      P_rot {a:3g}-{b:3g} d (n={s.sum():4d}): " + "  ".join(f"{k} {np.median(np.abs(x[s])):.3f}" for k, x in e.items()))
    return e


def split(n, seed=1):
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="simple", choices=list(VARIANTS))
    args = parser.parse_args()
    names = VARIANTS[args.variant]

    f = dict(np.load(FIELDS))
    f["dT"] = f["T_day"] - f["T_night"]
    train, test = split(len(f["P0"]))
    cost = StressCost(f, train, names)

    start = {k: COEFFS[k][0] for k in names}
    if args.variant != "simple" and os.path.exists(OUT.format("simple")):
        start |= json.load(open(OUT.format("simple")))
    x0 = np.array([start[k] for k in names])
    bounds = [COEFFS[k][1:] for k in names]
    per_planet_cost = np.mean(np.load(FITS)["misfit"][train] ** 2)
    print(f"variant {args.variant} ({len(names)} coefficients): {len(train)} training planets, {len(test)} held out; "
          f"initial cost {cost(x0):.4f} (per-planet fits: {per_planet_cost:.4f})")

    t0 = time.time()
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "eps": 1e-6})
    c = dict(zip(names, res.x))
    print(f"L-BFGS-B: {res.message}, {res.nit} iterations, {time.time() - t0:.0f} s; training cost {res.fun:.4f}, "
          f"held-out cost {StressCost(f, test, names)(res.x):.4f}")
    for k, v in c.items():
        shown = f"{np.degrees(v):.1f}°" if k.startswith("theta") else f"{v:.3f}"
        print(f"  {k:>13} = {shown}" + ("  (at a bound)" if np.isclose(v, COEFFS[k][1]) or np.isclose(v, COEFFS[k][2]) else ""))
    with open(OUT.format(args.variant), "w") as fh:
        json.dump(c, fh, indent=2)
    print(f"coefficients -> {os.path.normpath(OUT.format(args.variant))}")

    print("\nheld-out planets: median [10th, 90th pct] error (tau_conv, ustar3 signed relative; tau_prof, ustar3_map relative RMS)")
    score("per-planet fits (upper bound)", f, test, np.load(FITS)["params"].T[:, test])
    score(f"joint fit, {args.variant}", f, test, wind_parameters(c, f["dT"][test], f["P0"][test], f["P_rot"][test]))


if __name__ == "__main__":
    main()
