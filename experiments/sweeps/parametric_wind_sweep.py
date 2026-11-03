"""Space-filling sweep of CLERO planets for fitting parameterise_winds.parametric_wind against (T_day, ΔT, P0, P_rot).

Instead of inverting the emulator for chosen (T_day, T_night), sample its inputs (M_star, CO2, [Fe/H], P0, P_rot), read off each planet's
emulated (T_day, T_night), keep those in the target box, and thin them to even coverage in (T_day, ΔT, ln P0, ln P_rot). Then fit the
parametric wind to every kept planet.

    /data/pt426/ocean-venv/bin/python experiments/sweeps/parametric_wind_sweep.py [--n_candidates 60000] [--cap 6] [--workers 16]
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "climate"))
import climate_emulator as ce
from parameterise_winds import fit_parametric_wind

FIELDS = os.path.join(ROOT, "output", "winds", "parametric_sweep_fields.npz")
FITS = os.path.join(ROOT, "output", "winds", "parametric_sweep_fits.npz")
P_ROT_RANGE, P0_RANGE, CO2_RANGE = (4.0, 120.0), (0.5, 10.0), (1e-6, 1.0)
T_DAY_BOX, DT_BOX = (0.0, 60.0), (5.0, 80.0)


def sample_candidates(n, rng):
    """Random in-domain planets: log-uniform P_rot, P0, CO2, M_star; [Fe/H] ~ nearby M dwarfs (clipped at ±2σ)."""
    log_uniform = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    M = log_uniform(*ce.SEQ_MASS_RANGE)
    P_rot, P0, CO2 = log_uniform(*P_ROT_RANGE), log_uniform(*P0_RANGE), log_uniform(*CO2_RANGE)
    feh = np.clip(rng.normal(ce.FEH_MEAN, ce.FEH_STD, n), ce.FEH_MEAN - 2 * ce.FEH_STD, ce.FEH_MEAN + 2 * ce.FEH_STD)
    inputs = {"T_star": ce.star_properties(M, feh)[0], "F_star": ce.instellation(M, P_rot, feh), "radius": np.ones(n),
              "gravity": np.full(n, 9.81), "P_rot": P_rot, "P0": P0, "CO2": CO2, "CH4": np.zeros(n), "GCM": ["exocam"] * n}
    ok = np.isfinite(inputs["T_star"]) & ce.in_domain(inputs)
    extra = {"M_star": M, "feh": feh}
    return {k: (np.asarray(v)[ok] if k != "GCM" else [g for g, o in zip(v, ok) if o]) for k, v in (inputs | extra).items()}


def subset(planets, idx):
    return {k: (np.asarray(v)[idx] if k != "GCM" else [v[i] for i in idx]) for k, v in planets.items()}


def emulate(planets, fields, batch=4000):
    keys = list(ce.PLANET_KEYS)
    n = len(planets["P0"])
    out = {k: [] for k in fields}
    for s in range(0, n, batch):
        p = subset(planets, np.arange(s, min(s + batch, n)))
        pred = ce.predict({k: p[k] for k in keys}, fields)
        for k in fields:
            out[k].append(pred[k].astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def thin(T_day, dT, P0, P_rot, cap, rng):
    """Indices keeping at most `cap` planets per bin of (T_day, ΔT, ln P0, ln P_rot): even coverage of the input space."""
    bins = [np.digitize(T_day, np.linspace(*T_DAY_BOX, 7)), np.digitize(dT, np.linspace(*DT_BOX, 6)),
            np.digitize(np.log(P0), np.linspace(*np.log(P0_RANGE), 6)), np.digitize(np.log(P_rot), np.linspace(*np.log(P_ROT_RANGE), 7))]
    key = np.ravel_multi_index(bins, [8, 7, 7, 8])
    keep = []
    for k in np.unique(key):
        members = np.flatnonzero(key == k)
        keep.extend(rng.choice(members, min(cap, members.size), replace=False))
    return np.sort(np.array(keep))


def build_sweep(n_candidates, cap, seed=0):
    rng = np.random.default_rng(seed)
    planets = sample_candidates(n_candidates, rng)
    T = emulate(planets, ["surface_temperature"])["surface_temperature"]
    T_day, T_night = ce.diagnose_T(T)
    dT = T_day - T_night
    box = (T_day >= T_DAY_BOX[0]) & (T_day <= T_DAY_BOX[1]) & (dT >= DT_BOX[0]) & (dT <= DT_BOX[1])
    print(f"{n_candidates} candidates -> {len(T_day)} in domain -> {box.sum()} in the (T_day, ΔT) box")
    idx = np.flatnonzero(box)[thin(T_day[box], dT[box], planets["P0"][box], planets["P_rot"][box], cap, rng)]
    planets = subset(planets, idx)
    f = emulate(planets, ["u_0", "v_0", "temperature_0", "specific_humidity_0"])
    rho = ce.air_density(planets["P0"][:, None, None], planets["CO2"][:, None, None], 0.0, f["temperature_0"], f["specific_humidity_0"])
    out = {"u": f["u_0"], "v": f["v_0"], "rho": rho.astype(np.float32), "T_day": T_day[idx], "T_night": T_night[idx]}
    out |= {k: np.asarray(planets[k], dtype=float) for k in ("P_rot", "P0", "CO2", "M_star", "feh", "T_star", "F_star")}
    np.savez(FIELDS, **out)
    print(f"kept {len(idx)} planets -> {os.path.normpath(FIELDS)}")
    return out


def _fit_one(args):
    u, v, rho = args
    return fit_parametric_wind(u.astype(float), v.astype(float), ce.LAT, ce.LON, rho.astype(float))


def fit_sweep(f, workers):
    with Pool(workers) as pool:
        res = pool.map(_fit_one, zip(f["u"], f["v"], f["rho"]), chunksize=8)
    params, misfit = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    np.savez(FITS, params=params, misfit=misfit)
    print(f"fitted {len(params)} planets -> {os.path.normpath(FITS)}; stress misfit median {np.median(misfit):.3f}")
    return params


def coverage(f):
    dT = f["T_day"] - f["T_night"]
    for name, x, edges in (("T_day", f["T_day"], np.linspace(*T_DAY_BOX, 7)), ("ΔT", dT, np.linspace(*DT_BOX, 6)),
                           ("P0", f["P0"], np.geomspace(*P0_RANGE, 6)), ("P_rot", f["P_rot"], np.geomspace(*P_ROT_RANGE, 7))):
        counts, _ = np.histogram(x, edges)
        print(f"  {name:>5}: " + "  ".join(f"[{a:.3g}, {b:.3g}) {c}" for a, b, c in zip(edges[:-1], edges[1:], counts)))
    print("  rank correlations between inputs (want ~0):")
    from scipy.stats import spearmanr
    X = np.column_stack([f["T_day"], dT, np.log(f["P0"]), np.log(f["P_rot"])])
    print("   ", np.array2string(spearmanr(X).statistic, precision=2).replace("\n", "\n    "))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_candidates", type=int, default=60000)
    parser.add_argument("--cap", type=int, default=6, help="max planets per (T_day, ΔT, ln P0, ln P_rot) bin")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no_fit", action="store_true", help="build the sweep and report coverage only")
    args = parser.parse_args()
    f = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else build_sweep(args.n_candidates, args.cap)
    coverage(f)
    if not args.no_fit:
        fit_sweep(f, args.workers)


if __name__ == "__main__":
    main()
