"""2D circulation plots: a velocity field drawn as streamlines or quiver arrows over a coloured background."""

from analysis.simulation_reader import SimulationData, shapiro_filter
from analysis.paths import OCEAN_DIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.projections.geo import GeoAxes
from scipy.interpolate import RegularGridInterpolator

TERMINATOR_STYLE = dict(color='0.3', linestyle='--', linewidth=0.8)  # day-night boundary at lon = ±90° (substellar at lon = lat = 0)
SUBSTELLAR_STYLE = dict(marker='*', markersize=14, color='gold', mec='k', zorder=5, clip_on=False)
ARROW_STYLE = dict(width=0.0025, headwidth=4, color='k')
DAY_NIGHT_LABELS = ((-135, 'Nightside'), (-45, 'Dayside'), (0, 'Substellar'), (45, 'Dayside'), (135, 'Nightside'))


def _interp(x, y, field, xi, yi):
    """Bilinearly interpolate a (y, x) field onto the grid (yi, xi)."""
    points = np.stack(np.meshgrid(yi, xi, indexing='ij'), axis=-1)
    return RegularGridInterpolator((y, x), field)(points)


def _cell_centres(a, n):
    """n evenly spaced points at the centres of n equal cells spanning a."""
    edges = np.linspace(a.min(), a.max(), n + 1)
    return 0.5 * (edges[1:] + edges[:-1])


def _zonal_rate(u, lat, radius):
    """Zonal velocity in m/s -> angular rate in degrees longitude per second."""
    return np.degrees(u / (radius * np.cos(np.radians(lat))))


def _streamplot(ax, x, y, u, v, density):
    # streamplot needs a uniform grid; oversample so stretched (e.g. vertical) grids keep their detail
    xs, ys = (np.linspace(a.min(), a.max(), 2 * a.size) for a in (x, y))
    us, vs = (_interp(x, y, f, xs, ys) for f in (u, v))
    ax.streamplot(xs, ys, us, vs, density=density, color='k', linewidth=0.6, arrowsize=0.8)


def _to_pixels(ax, x, y):
    """Map data points (x, y) of any shape to display pixels, shape (..., 2)."""
    return ax.transData.transform(np.stack([x.ravel(), y.ravel()], axis=-1)).reshape(*x.shape, 2)


def _pixel_speed(ax, x, y, u, v, h=1e-6):
    """Speed of (u, v) in display pixels per unit time, via the local data->pixel Jacobian (works on polar axes)."""
    p0 = _to_pixels(ax, x, y)
    hx, hy = h * np.ptp(ax.get_xlim()), h * np.ptp(ax.get_ylim())
    dx = (_to_pixels(ax, x + hx, y) - p0) / hx
    dy = (_to_pixels(ax, x, y + hy) - p0) / hy
    return np.linalg.norm(dx * u[..., None] + dy * v[..., None], axis=-1)


def _arrow_key(ax, length, label):
    """Reference arrow `length` pixels long with its label, at the bottom-right of the axes."""
    x, y = (0.86, -0.12) if ax.name == 'rectilinear' else (0.86, 0.02)  # curved axes leave this corner empty
    frac = length / ax.bbox.width
    ax.autoscale(False)  # the key lives in axes coordinates and must not stretch the data limits
    ax.quiver(x, y, frac, 0, transform=ax.transAxes, scale_units='width', scale=1, pivot='tail',
              clip_on=False, **ARROW_STYLE)
    ax.text(x + frac + 0.01, y, label, transform=ax.transAxes, va='center')


def _quiver(ax, x, y, u, v, speed, n_arrows):
    # arrows on an even grid, pointing along (u, v), with length proportional to the physical speed
    xq, yq = (_cell_centres(a, n) for a, n in zip((x, y), n_arrows))
    uq, vq, sq = (_interp(x, y, f, xq, yq) for f in (u, v, speed))
    xq, yq = np.meshgrid(xq, yq)

    ax.figure.draw_without_rendering()  # settle the layout so the axes' pixel size is final
    pixels = _to_pixels(ax, xq, yq)
    row_spacing = np.median(np.linalg.norm(np.diff(pixels, axis=1), axis=-1), axis=1)
    spacing = min(row_spacing.max(), np.median(np.linalg.norm(np.diff(pixels, axis=0), axis=-1)))
    # thin rows that are squeezed on screen (high latitudes on a world map, inner radii on a ring)
    stride = np.maximum(1, np.round(spacing / row_spacing)).astype(int)
    keep = np.arange(xq.shape[1]) % stride[:, None] == 0
    xq, yq, uq, vq, sq = xq[keep], yq[keep], uq[keep], vq[keep], sq[keep]

    # a round reference speed near the 95th percentile spans one arrow spacing
    ref = float(f'{np.nanpercentile(sq, 95):.1g}')
    pixels = _pixel_speed(ax, xq, yq, uq, vq)
    scale = np.divide(spacing * sq / ref, pixels, out=np.zeros_like(pixels), where=pixels > 0)

    ax.quiver(xq, yq, uq * scale, vq * scale, angles='xy', scale_units='xy', scale=1, pivot='mid', **ARROW_STYLE)
    _arrow_key(ax, spacing, f'{ref:g} m/s')


def plot_flow(ax, x, y, u, v, speed, background, label, style='stream', cmap='Blues', norm=None, extend='neither',
              n_arrows=(36, 16), density=1.5):
    """
    Colour `background` on the (y, x) grid and overlay the flow (u, v) as streamlines or arrows.

    u, v must be in the units of the x and y axes per unit time, so the flow is drawn along its true path.
    speed    : physical flow speed (m/s) setting the quiver arrow lengths.
    style    : 'stream' (streamplot) or 'quiver' (arrows on an n_arrows = (nx, ny) grid, length ∝ speed).
    density  : streamplot line density.
    """
    mesh = ax.pcolormesh(x, y, background, cmap=cmap, norm=norm, shading='nearest')
    ax.figure.colorbar(mesh, ax=ax, label=label, extend=extend, shrink=1 if ax.name == 'rectilinear' else 0.7)
    if style == 'stream':
        _streamplot(ax, x, y, u, v, density)
    elif style == 'quiver':
        _quiver(ax, x, y, u, v, speed, n_arrows)
    else:
        raise ValueError(f"style must be 'stream' or 'quiver', not {style!r}")


def _longitude_label(theta, _):
    """Polar tick label: angle in radians -> longitude in (-180°, 180°]."""
    lon = np.degrees(theta) % 360
    lon = lon - 360 if lon > 180.5 else lon
    return 'Substellar\n0°' if round(lon) == 0 else f'{lon:.0f}°'


def _mark_day_night(ax, substellar_lat=None):
    """Label dayside/nightside, draw the terminators and mark the substellar point (lon = 0)."""
    if ax.name == 'polar':
        ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, **TERMINATOR_STYLE)
        for y, side in ((0.56, 'Dayside'), (0.44, 'Nightside')):
            ax.text(0.5, y, side, transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.plot(0, ax.get_ylim()[1], **SUBSTELLAR_STYLE)
        return

    if isinstance(ax, GeoAxes):  # world map: data are (lon, lat) in radians
        lat = np.linspace(-np.pi / 2, np.pi / 2, 181)
        for lon in (-np.pi / 2, np.pi / 2):
            ax.plot(np.full_like(lat, lon), lat, **TERMINATOR_STYLE)
        to_axes = ax.transData + ax.transAxes.inverted()
        for lon, side in DAY_NIGHT_LABELS:
            ax.text(to_axes.transform((np.radians(lon), 0))[0], 1.02, side, transform=ax.transAxes, ha='center')
        ax.plot(0, 0, **SUBSTELLAR_STYLE)
        return

    for lon in (-90, 90):
        ax.axvline(lon, **TERMINATOR_STYLE)
    top = ax.secondary_xaxis('top')
    top.set_xticks(*zip(*DAY_NIGHT_LABELS))
    top.tick_params(length=0)
    if substellar_lat is not None:
        ax.plot(0, substellar_lat, **SUBSTELLAR_STYLE)


def _colour_field(colour, u, v, w):
    """Background field and its plot_flow styling for colour = 'u', 'v', 'w' or 'speed'."""
    fields = {
        'u': (u, 'Zonal velocity $u$ (m/s)'),
        'v': (v, 'Meridional velocity $v$ (m/s)'),
        'w': (w * 86400, 'Vertical velocity $w$ (m/day)'),
        'speed': (np.hypot(u, v), 'Horizontal speed (m/s)'),
    }
    if colour not in fields:
        raise ValueError(f"colour must be one of {list(fields)}, not {colour!r}")
    field, label = fields[colour]
    if colour == 'speed':
        return dict(background=field, label=label, cmap='Blues')
    # clip at the 99th percentile so isolated grid-scale spikes (notably in w) don't wash out the scale
    norm = CenteredNorm(halfrange=np.nanpercentile(np.abs(field), 99))
    return dict(background=field, label=label, cmap='RdBu_r', norm=norm, extend='both')


def _vertical_velocity(sim, passes):
    """sim.w, optionally Shapiro-filtered in (lat, lon) to remove the 2Δ checkerboard of the diagnosed w."""
    return shapiro_filter(sim.w, order=2, passes=passes) if passes > 0 else sim.w


def _axes(xlabel, ylabel, title, figsize=(10, 5), projection=None):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained', subplot_kw={'projection': projection})
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, pad=6 if projection is None else 24)  # curved axes: clear the day/night labels
    return ax


def equatorial_section(sim: SimulationData, style='stream', polar=False, colour='u', w_filter_passes=0, **kwargs):
    """
    Longitude-depth section of (u, w) along the latitude row nearest the equator.

    colour : background field, 'u', 'v', 'w' or 'speed'.
    w_filter_passes : Shapiro-filter passes applied to w (colour and arrows) for display; 0 = raw.
    polar  : draw the section as a ring viewed from the north pole (longitude as angle, height as
             radius, substellar point at the top), with a central hole as wide as the ocean is deep.
    """
    i = np.argmin(np.abs(sim.lat))
    u, v, w = sim.u[:, i, :], sim.v[:, i, :], _vertical_velocity(sim, w_filter_passes)[:, i, :]
    dlon_dt = _zonal_rate(u, sim.lat[i], sim.radius)
    title = f'Equatorial circulation (lat = {sim.lat[i]:.1f}°)'

    if polar:
        ax = _axes('Longitude', '', title, figsize=(8, 7), projection='polar')
        ax.set_theta_zero_location('N')
        ax.xaxis.set_major_formatter(_longitude_label)
        ax.yaxis.set_major_formatter(lambda r, _: f'{abs(r):.0f} m')
        ax.set_rlabel_position(-22.5)
        kwargs.setdefault('n_arrows', (48, 5))
        bottom = 1.5 * sim.z[0] - 0.5 * sim.z[1]  # bottom cell edge
        ax.set_ylim(bottom, 0)
        ax.set_rorigin(2 * bottom)
        x, y, dx_dt, dy_dt = np.radians(sim.lon), sim.z, np.radians(dlon_dt), w
    else:
        ax = _axes('Longitude (°)', 'Depth (m)', title)
        ax.invert_yaxis()
        x, y, dx_dt, dy_dt = sim.lon, -sim.z, dlon_dt, -w

    kwargs.setdefault('n_arrows', (36, 12))
    plot_flow(ax, x, y, dx_dt, dy_dt, np.hypot(u, w), style=style, **_colour_field(colour, u, v, w), **kwargs)
    _mark_day_night(ax)
    return ax


def horizontal_circulation(sim: SimulationData, level=-1, style='stream', colour='speed', w_filter_passes=0,
                           projection=None, **kwargs):
    """
    Latitude-longitude map of (u, v) at one depth level; colour and w_filter_passes as in equatorial_section.

    projection : None for a flat lon-lat map, or a matplotlib world-map projection ('mollweide', 'hammer', 'aitoff').
    """
    u, v, w = sim.u[level], sim.v[level], _vertical_velocity(sim, w_filter_passes)[level]
    lat = sim.lat[:, np.newaxis]
    to_axis = np.radians if projection else np.asarray  # world-map projections take (lon, lat) in radians

    ax = _axes('Longitude (°)', 'Latitude (°)', f'Circulation at {-sim.z[level]:.0f} m depth', projection=projection)
    if projection:
        ax.grid(color='0.6', linewidth=0.4)
    plot_flow(ax, to_axis(sim.lon), to_axis(sim.lat), to_axis(_zonal_rate(u, lat, sim.radius)),
              to_axis(np.degrees(v / sim.radius)), np.hypot(u, v), style=style, **_colour_field(colour, u, v, w), **kwargs)
    _mark_day_night(ax, substellar_lat=0)
    return ax


if __name__ == '__main__':

    data = SimulationData('deepeq_P_10_dT_30.0_D_1.nc')

    for style in ('stream', 'quiver'):
        equatorial_section(data, style=style)
        equatorial_section(data, style=style, polar=True, colour='w')
        horizontal_circulation(data, style=style)
        horizontal_circulation(data, style=style, projection='mollweide')
    plt.show()
