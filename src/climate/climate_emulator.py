"""Invert the CLERO GCM emulator for the tidally locked planet that matches an ocean run's (P_rot, T_day, T_night) at a chosen
surface pressure P0, and save its near-surface winds and air density for use as ocean wind forcing (src/ocean/emulator_wind.jl).

Run with the repo's Python environment (requirements.txt includes `clero`):
    python src/climate/climate_emulator.py solve --P_rot 10 --T_day 30 --T_night 0 --P0 3 --out output/winds/P10_Td30_Tn0_P3.nc
    python src/climate/climate_emulator.py region --P_rot 10 --P0 1 3 10 --mark 30,30    # which (T_day, dT) have a solution at each P0
"""

import argparse
import os
import warnings
from importlib.metadata import version

import numpy as np
import matplotlib.pyplot as plt
from clero import Emulator, EXTENDED_DOMAIN, CORE_DOMAIN

SOLAR_CONSTANT = 1361  # W/m^2
CELSIUS = 273.15
R_GAS = 8.314462618  # J/(mol K)
MOLAR_MASS = {"H2O": 18.01528e-3, "N2": 28.0134e-3, "CO2": 44.0095e-3, "CH4": 16.043e-3}  # kg/mol; N2 background as in clero.climate_analysis.states
OCEAN_LAT_MAX = 80.0  # ocean grid spans latitude (-80, 80), see ocean_grid in src/ocean/ocean_sim.jl
PLANET_KEYS = ("T_star", "F_star", "radius", "gravity", "P_rot", "P0", "CO2", "CH4", "GCM")  # CLERO inputs

# Emulator grid: 32 Gaussian latitudes S->N, 64 longitudes with the substellar point at lon = 0
_x, _w = np.polynomial.legendre.leggauss(32)
LAT = np.degrees(np.arcsin(_x))
LON = np.linspace(-180.0 + 180.0 / 64, 180.0 - 180.0 / 64, 64)
_AREA = _w[:, None] * np.ones((1, 64))
_NIGHT = (np.abs(LON)[None, :] > 90) & (np.abs(LAT)[:, None] <= OCEAN_LAT_MAX)


class NoSolutionError(RuntimeError):
    """No in-domain planet reproduces the target (T_day, T_night) at the requested P_rot and P0; `misfit` (K) is the closest approach."""

    def __init__(self, message, misfit=np.inf):
        super().__init__(message)
        self.misfit = misfit


# --- Emulator access ---------------------------------------------------------------------------------------------------

_emulators = {}

def get_emulator(device=None):
    """Cached CLERO emulator; CUDA when available (~10x faster than CPU for batches)."""
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device not in _emulators:
        _emulators[device] = Emulator(device=device)
    return _emulators[device]

def predict(inputs, fields, device=None):
    """Batched emulator call returning (n, 32, 64) arrays; out-of-domain warnings are silenced, so check `in_domain` first."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*outside CLERO")
        out = get_emulator(device).predict(inputs, fields=list(fields))
    return {k: np.asarray(v).reshape(-1, 32, 64) for k, v in out.items()}


# --- Planet parameters -------------------------------------------------------------------------------------------------
# Stars are labelled by mass. T_eff and L follow Mamajek's mean dwarf sequence (taken to represent the field mean [Fe/H]), shifted
# at fixed mass by metallicity; P_orb = P_rot then fixes F_star via Kepler's third law.

FEH_MEAN, FEH_STD = -0.07, 0.22  # [Fe/H] of nearby M dwarfs (Gaidos et al. 2014, MNRAS 443, 2561)
DLOGT_DFEH, DLOGL_DFEH = -0.0368, -0.108  # d log10(T_eff, L)/d[Fe/H] at fixed mass, fitted to Mann et al. (2015) Eq. 5 + Mann et al. (2019) over 0.12-0.45 M_sun; extrapolated elsewhere
_SEQ = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamajek_dwarf_sequence.txt"), usecols=(1, 2, 3))
_SEQ = _SEQ[np.argsort(_SEQ[:, 2])]  # columns T_eff (K), log10 L (L_sun), M (M_sun), sorted by mass
SEQ_MASS_RANGE = (_SEQ[0, 2], _SEQ[-1, 2])

def star_properties(M_star, feh=FEH_MEAN):
    """(T_eff [K], L [L_sun]) of a main-sequence star of mass M_star (M_sun) and metallicity feh; NaN outside the tabulated masses."""
    logM = np.log10(np.atleast_1d(np.asarray(M_star, dtype=float)))
    logT = np.interp(logM, np.log10(_SEQ[:, 2]), np.log10(_SEQ[:, 0]), left=np.nan, right=np.nan)
    logL = np.interp(logM, np.log10(_SEQ[:, 2]), _SEQ[:, 1], left=np.nan, right=np.nan)
    d = np.asarray(feh, dtype=float) - FEH_MEAN
    return 10 ** (logT + DLOGT_DFEH * d), 10 ** (logL + DLOGL_DFEH * d)

def instellation(M_star, P_rotation, feh=FEH_MEAN):
    """Instellation (W/m^2) of a tidally locked planet (P_orb = P_rot, days) around a star of mass M_star (M_sun) and metallicity feh."""
    _, L = star_properties(M_star, feh)
    a = (np.atleast_1d(np.asarray(M_star, dtype=float)) * (P_rotation / 365.25) ** 2) ** (1 / 3)  # AU
    return SOLAR_CONSTANT * L / a ** 2

def planet_inputs(M_star, P_rotation, X_CO2, P_surf, R_p=1, g=9.81, X_CH4=0, GCM="exocam", feh=FEH_MEAN):
    """CLERO input batch (dict of arrays) for tidally locked planets; units: M_sun, days, mole fraction, bar, R_earth, m/s^2."""
    M_star, X_CO2, P_surf = (a.ravel().astype(float) for a in np.broadcast_arrays(*map(np.atleast_1d, (M_star, X_CO2, P_surf))))
    n = M_star.size
    return {
        "T_star": star_properties(M_star, feh)[0],
        "F_star": instellation(M_star, P_rotation, feh),
        "radius": np.full(n, float(R_p)),
        "gravity": np.full(n, float(g)),
        "P_rot": np.full(n, float(P_rotation)),
        "P0": P_surf,
        "CO2": X_CO2,
        "CH4": np.full(n, float(X_CH4)),
        "GCM": [GCM] * n,
    }

def in_domain(inputs, domain=EXTENDED_DOMAIN):
    ok = np.ones(len(inputs["P0"]), dtype=bool)
    for key, (lo, hi) in domain.items():
        ok &= (np.asarray(inputs[key]) >= lo) & (np.asarray(inputs[key]) <= hi)
    return ok

def viable_M_star(P_rotation, feh=FEH_MEAN, n=2000, domain=EXTENDED_DOMAIN):
    """Range of stellar masses (M_sun) whose T_eff and tidally locked instellation at this P_rot lie inside the emulator domain."""
    M = np.geomspace(*SEQ_MASS_RANGE, n)
    ok = in_domain(planet_inputs(M, P_rotation, 1e-3, 1.0, feh=feh), domain)
    if not ok.any():
        raise NoSolutionError(f"no stellar mass in {SEQ_MASS_RANGE} M_sun ([Fe/H] = {feh}) gives an in-domain planet at P_rot = {P_rotation} d")
    return M[ok].min(), M[ok].max()


# --- Diagnostics -------------------------------------------------------------------------------------------------------

def diagnose_T(T_s):
    """(T_day, T_night) in °C as the ocean's T_eq defines them: substellar 2x2-cell mean, nightside (|lat| <= 80°) area mean."""
    T_s = np.asarray(T_s)
    T_day = T_s[..., 15:17, 31:33].mean(axis=(-1, -2))
    T_night = (T_s * _AREA * _NIGHT).sum(axis=(-1, -2)) / (_AREA * _NIGHT).sum()
    return T_day - CELSIUS, T_night - CELSIUS

def air_density(P0, CO2, CH4, T_air, q):
    """Near-surface moist-air density (kg/m^3) from the ideal gas law: surface pressure P0 (bar), lowest-level T (K) and specific humidity."""
    M_dry = MOLAR_MASS["N2"] * (1 - CO2 - CH4) + MOLAR_MASS["CO2"] * CO2 + MOLAR_MASS["CH4"] * CH4
    M_air = 1 / (q / MOLAR_MASS["H2O"] + (1 - q) / M_dry)
    return P0 * 1e5 * M_air / (R_GAS * T_air)


# --- Inversion ---------------------------------------------------------------------------------------------------------

def solve_planet(P_rotation, T_day, T_night, P_surf, feh=FEH_MEAN, vary_feh=True, feh_range=(FEH_MEAN - 2 * FEH_STD, FEH_MEAN + 2 * FEH_STD),
                 feh_step=0.05, domain=EXTENDED_DOMAIN, verbose=True, **kwargs):
    """Find (M_star, X_CO2) matching the emulated (T_day, T_night) [°C] at fixed P_rot (days) and P0 = P_surf (bar), for a star of [Fe/H] = feh.

    If there is no solution and `vary_feh`, retries metallicities across feh_range (nearest to feh first, then bisected to ~0.01 dex) and
    returns the solution needing the smallest change in [Fe/H]; raises NoSolutionError (with the closest planet found) if none works.
    """
    try:
        return _solve_planet_fixed_feh(P_rotation, T_day, T_night, P_surf, feh, domain=domain, verbose=verbose, **kwargs)
    except NoSolutionError as err:
        if not vary_feh or not domain["P0"][0] <= P_surf <= domain["P0"][1]:
            raise
        best_err = err
    solve = lambda fe: _solve_planet_fixed_feh(P_rotation, T_day, T_night, P_surf, fe, domain=domain, verbose=False, **kwargs)

    steps = np.arange(feh_step, max(feh - feh_range[0], feh_range[1] - feh) + feh_step / 2, feh_step)
    trials = {fe for d in steps for fe in (feh - d, feh + d) if feh_range[0] < fe < feh_range[1]} | {fe for fe in feh_range if fe != feh}
    trials = sorted(trials, key=lambda fe: abs(fe - feh))  # nearest metallicity first; the band edges are always tried
    last_fail = {-1: feh, 1: feh}  # nearest failing metallicity on each side of feh
    for fe in trials:
        side = int(np.sign(fe - feh))
        try:
            planet = solve(fe)
        except NoSolutionError as err:
            last_fail[side] = fe
            best_err = min(best_err, err, key=lambda e: e.misfit)
            continue
        fail, ok = last_fail[side], fe
        for _ in range(3):  # bisect towards feh for the smallest metallicity change that still solves
            mid = 0.5 * (fail + ok)
            try:
                planet, ok = solve(mid), mid
            except NoSolutionError:
                fail = mid
        planet["feh_requested"] = feh
        if verbose:
            print(f"No solution at [Fe/H] = {feh:+.2f}; nearest solution at [Fe/H] = {planet['feh']:+.3f} (within {feh_range[0]:+.2f} to {feh_range[1]:+.2f}): "
                  f"M_star = {planet['M_star']:.3f} (T_star = {planet['T_star']:.0f} K, F_star = {planet['F_star']:.0f} W/m^2), "
                  f"CO2 = {planet['CO2']:.3g}, residual {planet['residual']:.3f} K")
        return planet
    raise NoSolutionError(f"{best_err} [no solution for any [Fe/H] in {feh_range[0]:+.2f} to {feh_range[1]:+.2f}]", best_err.misfit)

def _solve_planet_fixed_feh(P_rotation, T_day, T_night, P_surf, feh=FEH_MEAN, tol=0.1, n_M=40, n_CO2=40, CO2_bounds=(1e-6, 1.0), n_starts=8,
                            max_iter=40, domain=EXTENDED_DOMAIN, R_p=1, g=9.81, GCM="exocam", device=None, verbose=True):
    """solve_planet at a single [Fe/H] = feh.

    Grid-seeded, batched Levenberg-Marquardt; raises NoSolutionError (with the closest planet) if no root is within `tol` K.
    """
    if not domain["P0"][0] <= P_surf <= domain["P0"][1]:
        raise NoSolutionError(f"P0 = {P_surf} bar is outside the emulator domain {domain['P0']} bar")
    M_lo, M_hi = viable_M_star(P_rotation, feh, domain=domain)
    lo = np.array([np.log10(M_lo), np.log10(CO2_bounds[0])])
    span = np.array([np.log10(M_hi / M_lo), np.log10(CO2_bounds[1] / CO2_bounds[0])])
    to_phys = lambda x: lo + np.asarray(x) * span  # normalised x in [0, 1]^2 -> (log10 M_star, log10 CO2)

    def residual(x):
        p = to_phys(np.atleast_2d(x))
        r = np.full((len(p), 2), np.nan)
        good = np.isfinite(p).all(axis=1)
        inputs = planet_inputs(10 ** np.where(good, p[:, 0], lo[0]), P_rotation, 10 ** np.where(good, p[:, 1], lo[1]), P_surf, R_p, g, GCM=GCM, feh=feh)
        ok = good & in_domain(inputs, domain)
        if ok.any():
            T_s = predict({k: np.asarray(v)[ok] for k, v in inputs.items()}, ["surface_temperature"], device)["surface_temperature"]
            r[ok] = np.column_stack(diagnose_T(T_s)) - [T_day, T_night]
        return r

    # Seeds: local misfit minima of a coarse grid
    X = np.stack(np.meshgrid(np.linspace(0, 1, n_M), np.linspace(0, 1, n_CO2), indexing="ij"), -1).reshape(-1, 2)
    R = residual(X)
    mis = np.hypot(*R.T).reshape(n_M, n_CO2)
    if not np.isfinite(mis).any():
        raise NoSolutionError(f"no in-domain planet at P_rot = {P_rotation} d, P0 = {P_surf} bar")
    pad = np.pad(np.where(np.isfinite(mis), mis, np.inf), 1, constant_values=np.inf)
    neigh = np.min([pad[1 + a:1 + a + n_M, 1 + b:1 + b + n_CO2] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)], axis=0)
    seeds = np.flatnonzero((mis <= neigh).ravel() & np.isfinite(mis).ravel())
    seeds = seeds[np.argsort(mis.ravel()[seeds])][:n_starts]
    x, r = X[seeds].copy(), R[seeds].copy()
    lam = np.full(len(x), 1e-2)

    # Levenberg-Marquardt on all seeds at once
    h = 1e-3
    for _ in range(max_iter):
        f = np.hypot(*r.T)
        active = np.flatnonzero((f >= tol) & (lam < 1e8))
        if not len(active):
            break
        xa, ra, n = x[active], r[active], len(active)
        hs = np.where(xa + h > 1, -h, h)  # backward difference at the upper bound
        Rj = residual(np.vstack([xa + hs * [1, 0], xa + hs * [0, 1]]))
        J = np.stack([(Rj[:n] - ra) / hs[:, :1], (Rj[n:] - ra) / hs[:, 1:]], axis=-1)  # (n, residual, variable)
        JtJ = np.einsum("nki,nkj->nij", J, J)
        A = JtJ + (lam[active, None] * np.einsum("nii->ni", JtJ) + 1e-9)[:, :, None] * np.eye(2)
        step = -np.linalg.solve(np.nan_to_num(A, nan=1.0), np.nan_to_num(np.einsum("nki,nk->ni", J, ra))[..., None])[..., 0]
        trial = np.clip(xa + step, 0, 1)
        rt = residual(trial)
        better = np.hypot(*rt.T) < f[active]  # NaN (out of domain) counts as worse
        x[active[better]], r[active[better]] = trial[better], rt[better]
        lam[active[better]] /= 3
        lam[active[~better]] *= 4

    f = np.hypot(*r.T)
    roots = []
    for k in np.argsort(f):
        if f[k] < tol and all(np.abs(x[k] - x[j]).max() > 0.02 for j in roots):
            roots.append(k)

    if not roots:
        k = np.nanargmin(f)
        lM_b, lc_b = to_phys(x[k])
        T_b, F_b = star_properties(10 ** lM_b, feh)[0][0], instellation(10 ** lM_b, P_rotation, feh)[0]
        T_d, T_n = r[k] + [T_day, T_night]
        edges = [name for name, v in (("M_star", x[k, 0]), ("CO2", x[k, 1])) if v <= 1e-3 or v >= 1 - 1e-3]
        raise NoSolutionError(
            f"No planet with P0 = {P_surf} bar reproduces (T_day, T_night) = ({T_day}, {T_night}) °C at P_rot = {P_rotation} d. "
            f"Closest ([Fe/H] = {feh:+.2f}): M_star = {10 ** lM_b:.3f} (T_star = {T_b:.0f} K, F_star = {F_b:.0f} W/m^2), CO2 = {10 ** lc_b:.2g} -> ({T_d:.1f}, {T_n:.1f}) °C, misfit {f[k]:.2f} K "
            f"(day-night contrast {T_d - T_n:.1f} K vs target {T_day - T_night:.1f} K)"
            + (f"; at the edge of the {'/'.join(edges)} search range" if edges else ""), misfit=f[k])

    (lM_best, lc_best), *others = [to_phys(x[k]) for k in roots]
    M_best = 10 ** lM_best
    planet = {k: v[0] for k, v in planet_inputs(M_best, P_rotation, 10 ** lc_best, P_surf, R_p, g, GCM=GCM, feh=feh).items()}
    planet |= {"M_star": M_best, "L_star": float(star_properties(M_best, feh)[1][0]), "feh": feh, "residual": f[roots[0]],
               "alternatives": [dict(M_star=10 ** lM, CO2=10 ** lc, residual=f[k]) for (lM, lc), k in zip(others, roots[1:])]}
    if verbose:
        print(f"P_rot = {P_rotation} d, P0 = {P_surf} bar, [Fe/H] = {feh:+.2f}: M_star = {M_best:.3f} (T_star = {planet['T_star']:.0f} K, "
              f"F_star = {planet['F_star']:.0f} W/m^2), CO2 = {10 ** lc_best:.3g}, residual {planet['residual']:.3f} K")
        for alt in planet["alternatives"]:
            print(f"  WARNING: another root at M_star = {alt['M_star']:.3f}, CO2 = {alt['CO2']:.3g} (residual {alt['residual']:.3f} K)")
    return planet


# --- Achievable targets ------------------------------------------------------------------------------------------------

def solution_region(P_rotation, P0_values=(0.5, 1, 2, 3, 5, 10), T_day_range=(-10, 70), y_range=(0, 70), y="dT", step=0.5,
                    feh=FEH_MEAN, feh_range=(FEH_MEAN - 2 * FEH_STD, FEH_MEAN + 2 * FEH_STD), n_feh=5,
                    n_M=60, n_CO2=60, CO2_bounds=(1e-6, 1.0), domain=EXTENDED_DOMAIN, R_p=1, g=9.81, GCM="exocam", device=None):
    """Which targets have a solve_planet solution: for each P0, the image of solve_planet's (M_star, log CO2) search box in the
    (T_day, y) plane, y = "dT" (T_day - T_night) or "T_night" (°C), rasterised every `step` °C. mask[P0, T_day, y] is for [Fe/H] = feh;
    mask_band is the union over n_feh metallicities spanning feh_range (default: nearby M dwarfs' mean ± 2 sigma)."""
    if y not in ("dT", "T_night"):
        raise ValueError(f"y must be 'dT' or 'T_night', got {y!r}")
    xs = np.arange(T_day_range[0], T_day_range[1] + step / 2, step)
    ys = np.arange(y_range[0], y_range[1] + step / 2, step)
    fehs = np.unique(np.r_[feh, np.linspace(*feh_range, n_feh)])
    masks = np.zeros((len(fehs), len(P0_values), len(xs), len(ys)), dtype=bool)
    for f, fe in enumerate(fehs):
        try:
            M_lo, M_hi = viable_M_star(P_rotation, fe, domain=domain)
        except NoSolutionError:
            continue
        M, LC = np.meshgrid(np.geomspace(M_lo, M_hi, n_M), np.linspace(*np.log10(CO2_bounds), n_CO2), indexing="ij")
        for p, P0 in enumerate(P0_values):
            if not domain["P0"][0] <= P0 <= domain["P0"][1]:
                continue
            inputs = planet_inputs(M, P_rotation, 10 ** LC, P0, R_p, g, GCM=GCM, feh=fe)
            ok = in_domain(inputs, domain)
            T_d, T_n = np.full(M.size, np.nan), np.full(M.size, np.nan)
            if ok.any():
                T_d[ok], T_n[ok] = diagnose_T(predict({k: np.asarray(v)[ok] for k, v in inputs.items()}, ["surface_temperature"], device)["surface_temperature"])
            masks[f, p] = _rasterise_image(T_d.reshape(M.shape), (T_d - T_n if y == "dT" else T_n).reshape(M.shape), xs, ys)
    return {"P_rot": P_rotation, "P0": np.asarray(P0_values, dtype=float), "T_day": xs, "y": ys, "y_name": y, "GCM": GCM,
            "feh": feh, "feh_range": tuple(feh_range), "mask": masks[np.flatnonzero(fehs == feh)[0]], "mask_band": masks.any(axis=0)}

def _rasterise_image(X, Y, xs, ys):
    """Points of the (xs, ys) grid covered by the mapped grid (X, Y), each mapped cell split into two triangles."""
    covered = np.zeros((len(xs), len(ys)), dtype=bool)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    corners = [(X[:-1, :-1], Y[:-1, :-1]), (X[1:, :-1], Y[1:, :-1]), (X[1:, 1:], Y[1:, 1:]), (X[:-1, 1:], Y[:-1, 1:])]
    for a, b, c in ((0, 1, 2), (0, 2, 3)):
        tri = np.stack([*corners[a], *corners[b], *corners[c]], -1).reshape(-1, 6)
        for x1, y1, x2, y2, x3, y3 in tri[np.isfinite(tri).all(axis=1)]:
            i0, i1 = max(int(np.ceil((min(x1, x2, x3) - xs[0]) / dx)), 0), min(int(np.floor((max(x1, x2, x3) - xs[0]) / dx)), len(xs) - 1)
            j0, j1 = max(int(np.ceil((min(y1, y2, y3) - ys[0]) / dy)), 0), min(int(np.floor((max(y1, y2, y3) - ys[0]) / dy)), len(ys) - 1)
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if i0 > i1 or j0 > j1 or det == 0:
                continue
            px, py = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
            l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det  # barycentric coordinates
            l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
            covered[i0:i1 + 1, j0:j1 + 1] |= (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    return covered

def target_in_region(region, T_day, T_night):
    """Achievable P0 values (bar) for one target at the nominal [Fe/H] and anywhere in the [Fe/H] band (nearest raster point)."""
    yv = T_day - T_night if region["y_name"] == "dT" else T_night
    i, j = np.abs(region["T_day"] - T_day).argmin(), np.abs(region["y"] - yv).argmin()
    return region["P0"][region["mask"][:, i, j]], region["P0"][region["mask_band"][:, i, j]]


# --- Climate, output and plotting --------------------------------------------------------------------------------------

def planet_climate(planet, device=None):
    """Emulated (32, 64) surface temperature, lowest-level winds u_0/v_0 and near-surface air density for one planet."""
    out = predict({k: [planet[k]] for k in PLANET_KEYS}, ["surface_temperature", "u_0", "v_0", "temperature_0", "specific_humidity_0"], device)
    out = {k: v[0] for k, v in out.items()}
    out["rho_air"] = air_density(planet["P0"], planet["CO2"], planet["CH4"], out["temperature_0"], out["specific_humidity_0"])
    out["T_day"], out["T_night"] = (float(t) for t in diagnose_T(out["surface_temperature"]))
    return out

def emulate_planet(M_star, P_rotation, X_CO2, P_surf, R_p=1, g=9.81, X_CH4=0, feh=FEH_MEAN, device=None):
    """Full emulated climate (dict of 53 (32, 64) fields) for one tidally locked planet."""
    inputs = planet_inputs(M_star, P_rotation, X_CO2, P_surf, R_p, g, X_CH4, feh=feh)
    if not in_domain(inputs)[0]:
        raise NoSolutionError(f"planet is outside CLERO's extended domain: { {k: v[0] for k, v in inputs.items()} }")
    return get_emulator(device).predict({k: v[0] for k, v in inputs.items()})

def save_planet_winds(path, planet, climate, T_day, T_night, tol):
    """NetCDF read by src/ocean/emulator_wind.jl: (lat, lon) fields, planet parameters and targets as global attributes."""
    import xarray as xr
    field = lambda data, units, name: (("lat", "lon"), data, {"units": units, "long_name": name})
    ds = xr.Dataset(
        {
            "u_0": field(climate["u_0"], "m s-1", "eastward wind, lowest model level (p ~ 0.925 P0)"),
            "v_0": field(climate["v_0"], "m s-1", "northward wind, lowest model level (p ~ 0.925 P0)"),
            "T_s": field(climate["surface_temperature"], "K", "surface temperature"),
            "rho_air": field(climate["rho_air"], "kg m-3", "near-surface air density: P0 with lowest-level T and q (N2 background + CO2 + CH4 + H2O)"),
        },
        coords={"lat": ("lat", LAT, {"units": "degrees_north"}), "lon": ("lon", LON, {"units": "degrees_east", "note": "substellar point at 0"})},
        attrs={
            "P_rot_days": planet["P_rot"], "P0_bar": planet["P0"], "T_star_K": planet["T_star"], "M_star_Msun": planet["M_star"], "L_star_Lsun": planet["L_star"], "FeH": planet["feh"], "FeH_requested": planet.get("feh_requested", planet["feh"]),
            "F_star_W_m2": planet["F_star"], "CO2": planet["CO2"], "CH4": planet["CH4"], "radius_Rearth": planet["radius"],
            "gravity_m_s2": planet["gravity"], "GCM": planet["GCM"],
            "T_day_target_C": T_day, "T_night_target_C": T_night, "T_day_emu_C": climate["T_day"], "T_night_emu_C": climate["T_night"],
            "tol_K": tol, "emulator": f"clero {version('clero')}",
        },
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path)
    return ds

def plot_planet_winds(ds, path):
    """Quick look: surface temperature and lowest-level wind speed with arrows."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    pc = a1.pcolormesh(ds.lon, ds.lat, ds.T_s - CELSIUS, shading="nearest", cmap="RdYlBu_r")
    fig.colorbar(pc, ax=a1, label="surface temperature / °C")
    pc = a2.pcolormesh(ds.lon, ds.lat, np.hypot(ds.u_0, ds.v_0), shading="nearest", cmap="magma")
    a2.quiver(ds.lon[::2], ds.lat[::2], ds.u_0[::2, ::2], ds.v_0[::2, ::2], color="w")
    fig.colorbar(pc, ax=a2, label="|U| / m s$^{-1}$")
    for ax in (a1, a2):
        ax.axhline(OCEAN_LAT_MAX, c="w", ls=":"); ax.axhline(-OCEAN_LAT_MAX, c="w", ls=":")
        ax.set(xlabel="longitude from substellar point / deg", ylabel="latitude / deg")
    fig.suptitle(f"P_rot = {ds.P_rot_days} d, P0 = {ds.P0_bar} bar: M_star = {ds.M_star_Msun:.3f} ([Fe/H] = {ds.FeH:+.2f}), T_star = {ds.T_star_K:.0f} K, F_star = {ds.F_star_W_m2:.0f} W/m$^2$, CO2 = {ds.CO2:.2g}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solution_region(region, path):
    """One panel per P0: achievable region at the nominal [Fe/H] (solid) and over the [Fe/H] band (light, dashed); other P0 in grey."""
    n = len(region["P0"])
    ncol = min(n, 3)
    fig, axes = plt.subplots(-(-n // ncol), ncol, figsize=(4.2 * ncol, 3.6 * -(-n // ncol)), sharex=True, sharey=True, squeeze=False)
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, n))
    ylabel = r"$\Delta T = T_{day} - T_{night}$ / °C" if region["y_name"] == "dT" else r"$T_{night}$ / °C"
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        for j, m in enumerate(region["mask"]):
            if j != k and m.any():
                ax.contour(region["T_day"], region["y"], m.T.astype(float), levels=[0.5], colors="0.75", linewidths=0.8)
        band = region["mask_band"][k].T.astype(float)
        if band.any():
            ax.contourf(region["T_day"], region["y"], band, levels=[0.5, 1.5], colors=[colours[k]], alpha=0.18)
            ax.contour(region["T_day"], region["y"], band, levels=[0.5], colors=[colours[k]], linewidths=1.2, linestyles="--")
        m = region["mask"][k].T.astype(float)
        if m.any():
            ax.contourf(region["T_day"], region["y"], m, levels=[0.5, 1.5], colors=[colours[k]], alpha=0.5)
            ax.contour(region["T_day"], region["y"], m, levels=[0.5], colors=[colours[k]], linewidths=1.8)
        if region["y_name"] == "dT":  # T_night = T_day - dT > 0 °C below this line
            ax.plot(region["T_day"], region["T_day"], c="k", ls=":", lw=1.2, label=r"$T_{night} = 0$ °C (ice-free nightside below)" if k == 0 else None)
        else:
            ax.axhline(0, c="k", ls=":", lw=1.2, label=r"$T_{night} = 0$ °C (ice-free nightside above)" if k == 0 else None)
        ax.set_xlim(region["T_day"][[0, -1]])
        ax.set_ylim(region["y"][[0, -1]])
        ax.set_title(f"P0 = {region['P0'][k]:g} bar" + ("" if band.any() else " (no solutions)"))
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel(r"$T_{day}$ / °C")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    lo, hi = region["feh_range"]
    fig.suptitle(f"Targets with an emulator solution, P_rot = {region['P_rot']:g} d ({region['GCM']}): "
                 f"[Fe/H] = {region['feh']:+.2f} (solid), {lo:+.2f} to {hi:+.2f} (dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == '__main__':

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gcm", default="exocam", choices=["exocam", "um"])
    common.add_argument("--core", action="store_true", help="restrict to CLERO's core (densest) domain")
    common.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    common.add_argument("--feh", type=float, default=FEH_MEAN, help=f"stellar [Fe/H] (default {FEH_MEAN}, nearby M-dwarf mean)")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", parents=[common], help="solve for one planet and save its winds for the ocean model")
    solve.add_argument("--P_rot", type=float, required=True, help="rotation (= orbital) period, days")
    solve.add_argument("--T_day", type=float, required=True, help="substellar surface temperature, °C")
    solve.add_argument("--T_night", type=float, required=True, help="nightside mean surface temperature, °C")
    solve.add_argument("--P0", type=float, required=True, help="surface pressure, bar")
    solve.add_argument("--out", required=True, help="output NetCDF path")
    solve.add_argument("--tol", type=float, default=0.1, help="max misfit sqrt(dT_day^2 + dT_night^2), K")
    solve.add_argument("--fixed_feh", action="store_true", help="don't vary [Fe/H] (within the nearby M-dwarf mean ± 2 sigma) when there is no solution")

    reg = sub.add_parser("region", parents=[common], help="map which (T_day, dT) targets have a solution at each P0")
    reg.add_argument("--P_rot", type=float, required=True, help="rotation (= orbital) period, days")
    reg.add_argument("--P0", type=float, nargs="+", default=[0.5, 1, 2, 3, 5, 10], help="surface pressures, bar")
    reg.add_argument("--T_day_range", type=float, nargs=2, default=[-10, 70], help="°C")
    reg.add_argument("--y_range", type=float, nargs=2, default=[0, 70], help="dT or T_night range, °C")
    reg.add_argument("--y", default="dT", choices=["dT", "T_night"], help="vertical axis")
    reg.add_argument("--step", type=float, default=0.5, help="raster spacing, °C")
    reg.add_argument("--mark", nargs="*", default=[], help="targets to mark as T_day,y (e.g. 30,30)")
    reg.add_argument("--feh_range", type=float, nargs=2, default=[FEH_MEAN - 2 * FEH_STD, FEH_MEAN + 2 * FEH_STD],
                     help="[Fe/H] band (default nearby M-dwarf mean ± 2 sigma)")
    reg.add_argument("--out", default=None, help="figure path (default figures/solution_region_P<P_rot>.png)")
    args = parser.parse_args()
    domain = CORE_DOMAIN if args.core else EXTENDED_DOMAIN

    if args.command == "region":
        region = solution_region(args.P_rot, args.P0, args.T_day_range, args.y_range, args.y, args.step, feh=args.feh,
                                 feh_range=args.feh_range, domain=domain,
                                 GCM=args.gcm, device=args.device)
        marks = [tuple(float(v) for v in m.split(",")) for m in args.mark]
        for P0, m, mb in zip(region["P0"], region["mask"], region["mask_band"]):
            print(f"P0 = {P0:g} bar: {m.mean() * 100:.1f}% of the plotted (T_day, {args.y}) box has a solution at [Fe/H] = {args.feh:+.2f}, "
                  f"{mb.mean() * 100:.1f}% within [Fe/H] {args.feh_range[0]:+.2f} to {args.feh_range[1]:+.2f}")
        for x, yv in marks:
            T_n = x - yv if args.y == "dT" else yv
            nominal, band = target_in_region(region, x, T_n)
            print(f"target (T_day, T_night) = ({x:g}, {T_n:g}) °C: solutions for P0 = {nominal.tolist()} bar ([Fe/H] = {args.feh:+.2f}), "
                  f"{band.tolist()} bar (within the [Fe/H] band)")
        fig_path = args.out or os.path.join(os.path.dirname(__file__), "..", "..", "figures", f"solution_region_P{args.P_rot:g}.png")
        plot_solution_region(region, fig_path)
        print(f"figure -> {os.path.normpath(fig_path)}")
        raise SystemExit(0)

    try:
        planet = solve_planet(args.P_rot, args.T_day, args.T_night, args.P0, feh=args.feh, vary_feh=not args.fixed_feh, tol=args.tol,
                              GCM=args.gcm, device=args.device, domain=domain)
    except NoSolutionError as err:
        raise SystemExit(f"NoSolutionError: {err}")
    climate = planet_climate(planet, args.device)
    ds = save_planet_winds(args.out, planet, climate, args.T_day, args.T_night, args.tol)

    U = np.hypot(ds.u_0, ds.v_0).where(np.abs(ds.lat) <= OCEAN_LAT_MAX)
    print(f"-> {args.out}: T_star {planet['T_star']:.0f} K, F_star {planet['F_star']:.0f} W/m^2, "
          f"emulated (T_day, T_night) = ({climate['T_day']:.2f}, {climate['T_night']:.2f}) °C")
    print(f"   rho_air {float(ds.rho_air.min()):.2f}-{float(ds.rho_air.max()):.2f} kg/m^3, max |U_0| over |lat| <= 80: {float(U.max()):.1f} m/s")

    fig_path = os.path.join(os.path.dirname(__file__), "..", "..", "figures",
                            "emulator_wind_" + os.path.splitext(os.path.basename(args.out))[0] + ".png")
    plot_planet_winds(ds, fig_path)
    print(f"   figure -> {os.path.normpath(fig_path)}")
