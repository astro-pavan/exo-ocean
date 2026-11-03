"""Parametric near-surface wind for tidally locked planets: a flow converging on the substellar point, fitted to CLERO emulator winds.

The wind blows along great circles towards the substellar point with speed

    M(θ, α) = U_p · S(θ; θ_p(α), k(α)) · (1 − f sin²α)

plus a uniform zonal wind u0 cos φ, where θ is the angle from the substellar point and α the azimuth about it (0 = east along the
equator, 90° = north, 180° = west). The profile S rises as sin(πθ / 2θ_p) to 1 at the peak angle θ_p, then falls as
sech(k (θ − θ_p)) · [1 − ((θ − θ_p) / (π − θ_p))²]², reaching zero at the antistellar point. The peak angle and decay rate blend from
their east values (θ_E, k_E) to their west values (θ_W, k_W) as cos²(α/2). f > 0 focuses the inflow on the equator; f > 1 reverses it
at high latitude near the substellar meridian.

Grids follow the CLERO wind files: latitude φ and longitude λ in degrees, substellar point at (0, 0), fields shaped (lat, lon).
"""

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("U_p", "theta_E", "theta_W", "k_E", "k_W", "f", "u0")  # m/s, rad, rad, 1/rad, 1/rad, -, m/s
PARAM_BOUNDS = ((0.0, 0.02, 0.02, 0.0, 0.0, -1.0, -50.0),
                (200.0, 2.5, 2.5, 50.0, 50.0, 3.0, 50.0))


def tidally_locked_angles(lat, lon):
    """θ (angle from the substellar point) and α (azimuth about it, 0 = east along the equator) in radians, shaped (lat, lon)."""
    phi, lam = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    theta = np.arccos(np.clip(np.cos(phi) * np.cos(lam), -1, 1))
    alpha = np.arctan2(np.sin(phi), np.cos(phi) * np.sin(lam))
    return theta, alpha


def speed_profile(theta, theta_p, k):
    """S(θ): 0 at the substellar point, 1 at θ_p, 0 at the antistellar point."""
    rise = np.sin(np.pi * np.minimum(theta, theta_p) / (2 * theta_p))
    x = np.maximum(theta - theta_p, 0)
    fall = (1 - (x / (np.pi - theta_p)) ** 2) ** 2 / np.cosh(k * x)
    return np.where(theta < theta_p, rise, fall)


def parametric_wind(params, lat, lon):
    """(u, v) in m/s for params (U_p, theta_E, theta_W, k_E, k_W, f, u0) on the (lat, lon) grid."""
    U_p, theta_E, theta_W, k_E, k_W, f, u0 = params
    theta, alpha = tidally_locked_angles(lat, lon)
    phi, lam = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")

    east = np.cos(alpha / 2) ** 2  # 1 east of the substellar point, 0 west
    theta_p = theta_E * east + theta_W * (1 - east)
    k = k_E * east + k_W * (1 - east)
    speed = U_p * speed_profile(theta, theta_p, k) * (1 - f * np.sin(alpha) ** 2)

    # The unit vector towards the substellar point is (−sin λ, −sin φ cos λ) / sin θ; speed / sin θ → U_p π / 2θ_p at θ = 0
    sin_theta = np.sin(theta)
    near_pole = sin_theta < 1e-6
    speed_over_sin = np.where(near_pole, np.where(theta < 1, U_p * np.pi / (2 * theta_p), 0.0), speed / np.where(near_pole, 1, sin_theta))

    u = -speed_over_sin * np.sin(lam) + u0 * np.cos(phi)
    v = -speed_over_sin * np.sin(phi) * np.cos(lam)
    return u, v


def fit_parametric_wind(u, v, lat, lon, rho=1.0, lat_max=80.0, fit_u0=True):
    """Fit parametric_wind to one wind field over |lat| <= lat_max, matching the bulk stress ρ|U|U (C_D cancels).

    Each fit first matches the wind, then the stress, from several starting points (the fit has local minima).
    Returns (params, relative RMS stress misfit). fit_u0=False holds u0 at 0.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    area = np.sqrt(np.cos(np.radians(lat)) * (np.abs(lat) <= lat_max))[:, None]  # sqrt of area weight, ocean latitudes only

    def stress(uu, vv):
        speed = np.hypot(uu, vv)
        return rho * speed * uu, rho * speed * vv

    def residual(x, target):
        model = parametric_wind(x, lat, lon)
        if target is stress_ref:
            model = stress(*model)
        return np.concatenate([((m - t) * area).ravel() for m, t in zip(model, target)])

    wind_ref, stress_ref = (u, v), stress(u, v)
    lower, upper = np.array(PARAM_BOUNDS[0]), np.array(PARAM_BOUNDS[1])
    if not fit_u0:
        lower[-1], upper[-1] = -1e-9, 1e-9

    U0 = max(np.abs(u[np.abs(lat) < 5]).max(), 1.0)
    starts = [(U0, tp, tp, k, k, 0.3, 0.0) for tp in (0.3, 0.7) for k in (0.7, 2.5)]
    starts.append((U0, 1.2, 0.3, 0.1, 3.0, 0.5, 0.0))  # fast rotator: broad east, narrow west

    best = None
    for x0 in starts:
        fit = least_squares(residual, x0, bounds=(lower, upper), args=(wind_ref,))
        fit = least_squares(residual, fit.x, bounds=(lower, upper), args=(stress_ref,))
        if best is None or fit.cost < best.cost:
            best = fit

    norm = np.sqrt(sum(((t * area) ** 2).sum() for t in stress_ref))
    return best.x, np.linalg.norm(best.fun) / norm #  type: ignore
