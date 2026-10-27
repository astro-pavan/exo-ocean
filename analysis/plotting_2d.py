"""2D circulation plots: a velocity field drawn as streamlines or quiver arrows over a coloured background."""

import io
from pathlib import Path

from analysis.simulation_reader import SimulationData, shapiro_filter
from analysis.paths import OCEAN_DIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.projections.geo import GeoAxes
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

try:  # cartopy (ocean-venv) is only needed for the orthographic globe
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes as CartopyAxes
except ImportError:
    ccrs = CartopyAxes = None

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
    # plain streamplot: the grid is already in the axes' coordinates, and cartopy's would re-interpolate it
    Axes.streamplot(ax, xs, ys, us, vs, density=density, color='k', linewidth=0.6, arrowsize=0.8)


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
    # plain matplotlib quiver: cartopy's would try to reproject the key
    Axes.quiver(ax, x, y, frac, 0, transform=ax.transAxes, scale_units='width', scale=1, pivot='tail',
                clip_on=False, **ARROW_STYLE)
    ax.text(x - 0.01, y, label, transform=ax.transAxes, ha='right', va='center')


def _round_1sf(x):
    return float(f'{x:.1g}')


def _draw_arrows(ax, xq, yq, uq, vq, sq, thin=True, spacing=None, ref_speed=None):
    """
    Arrows at the rows of points (xq, yq) (NaN = hidden) along (uq, vq), length ∝ physical speed sq, plus a key.

    thin      : drop arrows from rows squeezed on screen; False when the rows are already thinned.
    spacing   : arrow spacing in data units of isotropic axes (e.g. the globe); None measures it on screen.
    ref_speed : speed (m/s) drawn one spacing long; None uses a round value near the 95th percentile.
    """
    ax.figure.draw_without_rendering()  # settle the layout so the axes' pixel size is final
    pixels = _to_pixels(ax, xq, yq)

    def gaps(axis):  # on-screen distances between neighbouring arrows, ignoring hidden ones
        return np.ma.masked_invalid(np.linalg.norm(np.diff(pixels, axis=axis), axis=-1))

    if spacing is not None:
        spacing = np.linalg.norm(np.diff(ax.transData.transform([(0, 0), (spacing, 0)]), axis=0))
    keep = np.isfinite(xq) & np.isfinite(uq)
    if thin:  # thin rows that are squeezed on screen (high latitudes on a map, inner radii on a ring)
        row_spacing = np.ma.median(gaps(1), axis=1)
        spacing = spacing or min(row_spacing.max(), np.ma.median(gaps(0)))
        stride = np.maximum(1, np.round(np.ma.divide(spacing, row_spacing)).filled(1)).astype(int)
        keep &= np.arange(xq.shape[1]) % stride[:, None] == 0
    elif spacing is None:
        spacing = np.ma.median(gaps(0))
    xq, yq, uq, vq, sq = xq[keep], yq[keep], uq[keep], vq[keep], sq[keep]

    ref = ref_speed or _round_1sf(np.nanpercentile(sq, 95))
    pixels = _pixel_speed(ax, xq, yq, uq, vq)
    scale = np.divide(spacing * sq / ref, pixels, out=np.zeros_like(pixels), where=pixels > 0)

    ax.quiver(xq, yq, uq * scale, vq * scale, angles='xy', scale_units='xy', scale=1, pivot='mid', **ARROW_STYLE)
    _arrow_key(ax, spacing, f'{ref:g} m/s')


def _quiver(ax, x, y, u, v, speed, n_arrows, ref_speed=None):
    # arrows on an even n_arrows grid of the axes' data coordinates
    xq, yq = (_cell_centres(a, n) for a, n in zip((x, y), n_arrows))
    uq, vq, sq = (_interp(x, y, f, xq, yq) for f in (u, v, speed))
    _draw_arrows(ax, *np.meshgrid(xq, yq), uq, vq, sq, ref_speed=ref_speed)


def plot_flow(ax, x, y, u, v, speed, background, label, style='stream', cmap='Blues', norm=None, extend='neither',
              n_arrows=(36, 16), density=1.5, cax=None, arrows=None, ref_speed=None):
    """
    Colour `background` on the (y, x) grid and overlay the flow (u, v) as streamlines or arrows.

    u, v must be in the units of the x and y axes per unit time, so the flow is drawn along its true path.
    speed    : physical flow speed (m/s) setting the quiver arrow lengths.
    style    : 'stream' (streamplot) or 'quiver' (arrows on an n_arrows = (nx, ny) grid, length ∝ speed).
    density  : streamplot line density.
    cax      : axes to draw a horizontal colorbar in; None puts a vertical one beside ax.
    arrows   : _draw_arrows arguments for precomputed quiver rows, e.g. on a lon/lat graticule.
    ref_speed: quiver speed (m/s) drawn one arrow spacing long; None picks one from the arrows shown.
    """
    mesh = ax.pcolormesh(x, y, background, cmap=cmap, norm=norm, shading='nearest')
    placement = dict(ax=ax, shrink=1 if ax.name == 'rectilinear' else 0.7) if cax is None else \
        dict(cax=cax, orientation='horizontal')
    ax.figure.colorbar(mesh, label=label, extend=extend, **placement)
    if style == 'stream':
        _streamplot(ax, x, y, u, v, density)
    elif style == 'quiver':
        if arrows is None:
            _quiver(ax, x, y, u, v, speed, n_arrows, ref_speed)
        else:
            _draw_arrows(ax, **arrows, ref_speed=ref_speed)
    else:
        raise ValueError(f"style must be 'stream' or 'quiver', not {style!r}")


def _longitude_label(theta, _):
    """Polar tick label: angle in radians -> longitude in (-180°, 180°]."""
    lon = np.degrees(theta) % 360
    lon = lon - 360 if lon > 180.5 else lon
    return 'Substellar\n0°' if round(lon) == 0 else f'{lon:.0f}°'


def _latitude_label(theta, _):
    """Polar tick label: angle in radians -> latitude with N/S."""
    lat = round(np.degrees(theta))
    return f'{abs(lat)}°' + ('N' if lat > 0 else 'S' if lat < 0 else '')


def _sample_lonlat(lon, lat, f, lon_pts, lat_pts):
    """Bilinear, longitude-periodic interpolation of a (lat, lon) field at points (NaN where hidden or outside the data)."""
    lon_wrapped = np.concatenate([lon[-1:] - 360, lon, lon[:1] + 360])
    f_wrapped = np.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    out = np.full(np.shape(lon_pts), np.nan)
    ok = np.isfinite(lon_pts) & np.isfinite(lat_pts)
    out[ok] = RegularGridInterpolator((lat, lon_wrapped), f_wrapped, bounds_error=False)(
        np.column_stack([lat_pts[ok], lon_pts[ok]]))
    return out


def _project_flow(proj, lon_pts, lat_pts, dlon, dlat):
    """Projected positions and flow components for lon/lat points and rates (°/s); NaN where hidden or without data."""
    xy = proj.transform_points(ccrs.PlateCarree(), lon_pts, lat_pts)[..., :2]
    ok = np.all(np.isfinite(xy), axis=-1) & np.isfinite(dlon)
    xy[~ok] = np.nan
    ux, uy = np.full(dlon.shape, np.nan), np.full(dlon.shape, np.nan)
    ux[ok], uy[ok] = proj.transform_vectors(ccrs.PlateCarree(), lon_pts[ok], lat_pts[ok], dlon[ok], dlat[ok])
    return xy[..., 0], xy[..., 1], ux, uy


def _orthographic_regrid(proj, lon, lat, scalars, rates, n=400):
    """
    Sample lon-lat fields on an even n×n grid of the orthographic projection `proj` (NaN off the visible disc).

    rates (dlon/dt, dlat/dt) in °/s become flow components along the projected x, y axes.
    """
    x, y = (np.linspace(*limits, n) for limits in (proj.x_limits, proj.y_limits))
    lonlat = ccrs.PlateCarree().transform_points(proj, *np.meshgrid(x, y))
    lon_t, lat_t = lonlat[..., 0], lonlat[..., 1]
    dlon, dlat = (_sample_lonlat(lon, lat, r, lon_t, lat_t) for r in rates)
    _, _, ux, uy = _project_flow(proj, lon_t, lat_t, dlon, dlat)
    return x, y, ux, uy, [_sample_lonlat(lon, lat, f, lon_t, lat_t) for f in scalars]


def _graticule_arrows(proj, lon, lat, rates, speed, step):
    """_draw_arrows arguments for arrows at lon/lat graticule points every `step` degrees, projected by `proj`."""
    lon_q, lat_q = np.meshgrid(np.arange(-180, 180, step), np.arange(-90 + step, 90, step))
    # even spacing on the sphere: keep every n-th arrow along a latitude, n ≈ 1/cos(lat) rounded to a power of 3
    # so the kept arrows stay on the meridians drawn every 3 steps
    stride = 3 ** np.round(np.log(1 / np.cos(np.radians(lat_q))) / np.log(3))
    lon_q = np.where(np.round((lon_q + 180) / step) % stride == 0, lon_q, np.nan)
    dlon, dlat, sq = (_sample_lonlat(lon, lat, f, lon_q, lat_q) for f in (*rates, speed))
    xq, yq, uq, vq = _project_flow(proj, lon_q, lat_q, dlon, dlat)
    radius = np.ptp(proj.x_limits) / 2  # fixed spacing (arc length of one step) keeps arrows comparable between views
    return dict(xq=xq, yq=yq, uq=uq, vq=vq, sq=sq, thin=False, spacing=radius * np.radians(step))


def _mark_day_night(ax, labels=True, substellar_y=None):
    """
    Draw the terminators (lon = ±90°), optionally label the day/night sides and mark the substellar longitude.

    substellar_y : y coordinate (latitude, or height on a section) of the substellar marker at lon = 0; None = no marker.
    """
    if CartopyAxes is not None and isinstance(ax, CartopyAxes):  # globe: lat/lon via transforms, only where visible
        lat = np.linspace(-90, 90, 181)
        for lon in (-90, 90):
            ax.plot(np.full_like(lat, lon), lat, transform=ccrs.Geodetic(), **TERMINATOR_STYLE)
        if labels:
            for lon, side in ((0, 'Dayside'), (180, 'Nightside')):
                xy = ax.projection.transform_point(lon, 50, ccrs.PlateCarree())
                if np.all(np.isfinite(xy)):
                    ax.text(*xy, side, ha='center', va='center', bbox=dict(fc='w', ec='none', alpha=0.7, pad=1.5))
        if substellar_y is not None:
            xy = ax.projection.transform_point(0, substellar_y, ccrs.PlateCarree())
            if np.all(np.isfinite(xy)):
                ax.plot(*xy, **SUBSTELLAR_STYLE)
        return

    if ax.name == 'polar':
        ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, **TERMINATOR_STYLE)
        if labels:
            for y, side in ((0.56, 'Dayside'), (0.44, 'Nightside')):
                ax.text(0.5, y, side, transform=ax.transAxes, ha='center', va='center', fontsize=11)
    elif isinstance(ax, GeoAxes):  # world map: data are (lon, lat) in radians
        lat = np.linspace(-np.pi / 2, np.pi / 2, 181)
        for lon in (-np.pi / 2, np.pi / 2):
            ax.plot(np.full_like(lat, lon), lat, **TERMINATOR_STYLE)
        if labels:
            to_axes = ax.transData + ax.transAxes.inverted()
            for lon, side in DAY_NIGHT_LABELS:
                ax.text(to_axes.transform((np.radians(lon), 0))[0], 1.02, side, transform=ax.transAxes, ha='center')
    else:
        for lon in (-90, 90):
            ax.axvline(lon, **TERMINATOR_STYLE)
        if labels:
            top = ax.secondary_xaxis('top')
            top.set_xticks(*zip(*DAY_NIGHT_LABELS))
            top.tick_params(length=0)
    if substellar_y is not None:
        ax.plot(0, substellar_y, **SUBSTELLAR_STYLE)


def _colour_field(colour, u, v, w, T):
    """Background field and its plot_flow styling for colour = 'u', 'v', 'w' or 'speed'."""
    fields = {
        'u': (u, 'Zonal velocity $u$ (m/s)'),
        'v': (v, 'Meridional velocity $v$ (m/s)'),
        'w': (w * 86400, 'Vertical velocity $w$ (m/day)'),
        'speed': (np.hypot(u, v), 'Horizontal speed (m/s)'),
        'T': (T, 'Temperature (C)')
    }
    if colour not in fields:
        raise ValueError(f"colour must be one of {list(fields)}, not {colour!r}")
    field, label = fields[colour]
    # colour ranges come from the whole field, so a partial view (e.g. one side of the globe) keeps the same scale
    if colour == 'speed':
        return dict(background=field, label=label, cmap='Blues', norm=Normalize(0, np.nanmax(field)))
    if colour == 'T':
        return dict(background=field, label=label, cmap='RdBu_r', norm=Normalize(np.nanmin(field), np.nanmax(field)))
    # clip at the 99th percentile so isolated grid-scale spikes (notably in w) don't wash out the scale
    norm = CenteredNorm(halfrange=np.nanpercentile(np.abs(field), 99))
    return dict(background=field, label=label, cmap='RdBu_r', norm=norm, extend='both')


def _vertical_velocity(sim, passes):
    """sim.w, optionally Shapiro-filtered in (lat, lon) to remove the 2Δ checkerboard of the diagnosed w."""
    return shapiro_filter(sim.w, order=2, passes=passes) if passes > 0 else sim.w


def _axes(xlabel, ylabel, title, figsize=(10, 5), projection=None):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained', subplot_kw={'projection': projection})
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title is not None:
        ax.set_title(title, pad=6 if projection is None else 24)  # curved axes: clear the day/night labels
    return ax


def _section_axes(sim, xlabel, title, polar, theta_formatter=None, theta_zero='N', theta_direction=1, thetalim=None,
                  figsize=(8, 7)):
    """Axes for a vertical section: flat with depth downwards, or polar with height as radius and a hole as deep as the ocean."""
    if not polar:
        ax = _axes(f'{xlabel} (°)', 'Depth (m)', None)
        ax.invert_yaxis()
        return ax
    ax = _axes(xlabel, '', title, figsize=figsize, projection='polar')
    ax.set_theta_zero_location(theta_zero)
    ax.set_theta_direction(theta_direction)
    if thetalim is not None:
        ax.set_thetalim(*np.radians(thetalim))
    ax.xaxis.set_major_formatter(theta_formatter)
    ax.yaxis.set_major_formatter(lambda r, _: f'{abs(r):.0f} m')
    ax.set_rlabel_position(-22.5)
    bottom = 1.5 * sim.z[0] - 0.5 * sim.z[1]  # bottom cell edge
    ax.set_ylim(bottom, 0)
    ax.set_rorigin(2 * bottom)
    return ax


def _section_coords(ax, sim, coord, rate, w):
    """Section (x, y, dx/dt, dy/dt) for a horizontal coordinate and its rate in degrees: radians and height on polar axes, depth on flat ones."""
    if ax.name == 'polar':
        return np.radians(coord), sim.z, np.radians(rate), w
    return coord, -sim.z, rate, -w


def equatorial_section(sim: SimulationData, style='stream', polar=False, colour='u', w_filter_passes=0, substellar=True,
                       **kwargs):
    """
    Longitude-depth section of (u, w) along the latitude row nearest the equator.

    colour : background field, 'u', 'v', 'w', 'speed' or 'T'.
    w_filter_passes : Shapiro-filter passes applied to w (colour and arrows) for display; 0 = raw.
    polar  : draw the section as a ring viewed from the north pole (longitude as angle, height as
             radius, substellar point at the top), with a central hole as wide as the ocean is deep.
    substellar : mark the substellar point at the surface.
    """
    i = np.argmin(np.abs(sim.lat))
    u, v, w = sim.u[:, i, :], sim.v[:, i, :], _vertical_velocity(sim, w_filter_passes)[:, i, :]
    T = sim.T[:, i, :]

    ax = _section_axes(sim, 'Longitude', f'Equatorial circulation (lat = {sim.lat[i]:.1f}°)', polar,
                       theta_formatter=_longitude_label)
    kwargs.setdefault('n_arrows', (48, 5) if polar else (36, 12))
    x, y, dx_dt, dy_dt = _section_coords(ax, sim, sim.lon, _zonal_rate(u, sim.lat[i], sim.radius), w)
    plot_flow(ax, x, y, dx_dt, dy_dt, np.hypot(u, w), style=style, **_colour_field(colour, u, v, w, T), **kwargs)
    _mark_day_night(ax, substellar_y=ax.get_ylim()[1] if substellar else None)  # ylim[1] is the surface on both axes
    return ax


def meridional_section(sim: SimulationData, lon=0, style='stream', polar=True, landscape=False, colour='u',
                       w_filter_passes=0, substellar=True, **kwargs):
    """
    Latitude-depth section of (v, w) along the longitude column nearest `lon` (default: substellar).

    polar : draw the section as a semicircle seen from the side (latitude as angle, north at the top,
            height as radius) with a hole as deep as the ocean; False gives a flat latitude-depth plot.
    landscape : lay the semicircle flat-side down (equator up, north on the right) instead of upright.
    colour, w_filter_passes, substellar as in equatorial_section; arrow lengths use the in-plane speed |(v, w)|.
    """
    j = np.argmin(np.abs((sim.lon - lon + 180) % 360 - 180))
    u, v, w = sim.u[:, :, j], sim.v[:, :, j], _vertical_velocity(sim, w_filter_passes)[:, :, j]
    T = sim.T[:, :, j]
    side = 'dayside' if abs(sim.lon[j]) < 90 else 'nightside'

    orientation = dict(theta_zero='N', theta_direction=-1, figsize=(7, 6.2)) if landscape else \
        dict(theta_zero='E', figsize=(6, 7))
    ax = _section_axes(sim, 'Latitude', f'Meridional circulation (lon = {sim.lon[j]:.1f}°, {side})', polar,
                       theta_formatter=_latitude_label, thetalim=(-90, 90), **orientation)
    if polar and landscape:
        # the half disc fills only the middle of the square polar box: keep title and colorbar inside that box
        ax.set_title(ax.get_title(), y=0.84, pad=0)
        ax.set_xlabel('')
        ax.yaxis.set_major_locator(MaxNLocator(3))  # depth labels share the short flat edge
        kwargs.setdefault('cax', ax.inset_axes([0.2, 0.13, 0.55, 0.03]))
    kwargs.setdefault('n_arrows', (24, 5) if polar else (32, 12))
    x, y, dx_dt, dy_dt = _section_coords(ax, sim, sim.lat, np.degrees(v / sim.radius), w)
    plot_flow(ax, x, y, dx_dt, dy_dt, np.hypot(v, w), style=style, **_colour_field(colour, u, v, w, T), **kwargs)
    if substellar and abs(sim.lon[j]) < np.diff(sim.lon[:2])[0]:  # section passes through the substellar point
        ax.plot(0, 0 if polar else -sim.z[-1], **SUBSTELLAR_STYLE)
    return ax


def horizontal_circulation(sim: SimulationData, level=-1, style='stream', colour='speed', w_filter_passes=0,
                           projection=None, centre=(0, 0), arrow_step=10, substellar=True, **kwargs):
    """
    Latitude-longitude map of (u, v) at one depth level; colour and w_filter_passes as in equatorial_section.

    projection : None for a flat lon-lat map, a matplotlib world-map projection ('mollweide', 'hammer', 'aitoff'),
                 or 'orthographic' for the sphere seen from space (needs cartopy, e.g. ocean-venv).
    centre     : (lat, lon) at the centre of the orthographic globe; default is the substellar point.
    arrow_step : globe quiver arrows sit on the lon/lat graticule every arrow_step degrees (gridlines every 3 steps).
    substellar : mark the substellar point (lon = lat = 0).
    """
    u, v, w = sim.u[level], sim.v[level], _vertical_velocity(sim, w_filter_passes)[level]
    T = sim.T[level]
    colour_kw = _colour_field(colour, u, v, w, T)
    speed = np.hypot(u, v)
    dlon_dt, dlat_dt = _zonal_rate(u, sim.lat[:, np.newaxis], sim.radius), np.degrees(v / sim.radius)

    if projection == 'orthographic':
        if ccrs is None:
            raise ImportError("projection='orthographic' needs cartopy (installed in ocean-venv)")
        proj = ccrs.Orthographic(central_longitude=centre[1], central_latitude=centre[0])
        ax = _axes('', '', None, figsize=(7, 6), projection=proj)
        ax.set_global()
        grid_step = 3 * arrow_step
        ax.gridlines(color='0.6', linewidth=0.4, xlocs=np.arange(-180, 180, grid_step),
                     ylocs=np.arange(-90, 91, grid_step))
        kwargs.setdefault('arrows', _graticule_arrows(proj, sim.lon, sim.lat, (dlon_dt, dlat_dt), speed, arrow_step))
        x, y, dx_dt, dy_dt, (colour_kw['background'], speed) = _orthographic_regrid(
            proj, sim.lon, sim.lat, (colour_kw['background'], speed), (dlon_dt, dlat_dt))
    else:
        to_axis = np.radians if projection else np.asarray  # world-map projections take (lon, lat) in radians
        ax = _axes('Longitude (°)', 'Latitude (°)', None, projection=projection)
        if projection:
            ax.grid(color='0.6', linewidth=0.4)
        x, y, dx_dt, dy_dt = (to_axis(a) for a in (sim.lon, sim.lat, dlon_dt, dlat_dt))

    plot_flow(ax, x, y, dx_dt, dy_dt, speed, style=style, **colour_kw, **kwargs)
    _mark_day_night(ax, labels=False, substellar_y=0 if substellar else None)
    return ax


def _save_animation(frames, path, fps):
    """Write PIL frames to a looping GIF (Pillow) or an H.264 MP4 (imageio-ffmpeg)."""
    if path.suffix == '.gif':
        # GIFs hold 256 colours per frame; max-coverage quantizing keeps small features such as the gold marker
        frames = [f.quantize(256, method=Image.Quantize.MAXCOVERAGE) for f in frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ImportError("MP4 output needs imageio-ffmpeg (pip install imageio-ffmpeg); or save a .gif") from None
    width, height = (n - n % 2 for n in frames[0].size)  # H.264 needs even dimensions
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, macro_block_size=1)
    writer.send(None)  # start the ffmpeg process
    for frame in frames:
        writer.send(frame.crop((0, 0, width, height)).tobytes())
    writer.close()


def rotating_globe(sim: SimulationData, path, lat=20, n_frames=72, fps=12, dpi=100, style='quiver', **kwargs):
    """
    Animate horizontal_circulation on the orthographic globe turning once eastward, seen from latitude `lat`.

    path   : output file, '.gif' (Pillow) or '.mp4' (needs imageio-ffmpeg).
    kwargs : passed to horizontal_circulation (level, colour, w_filter_passes, arrow_step, substellar, ref_speed, ...).
    The colour scale and the arrow reference speed come from the whole level, so they are fixed between frames.
    """
    path = Path(path)
    if path.suffix not in ('.gif', '.mp4'):
        raise ValueError(f"path must end in .gif or .mp4, not {path.suffix!r}")
    level = kwargs.get('level', -1)
    kwargs.setdefault('ref_speed', _round_1sf(np.nanpercentile(np.hypot(sim.u[level], sim.v[level]), 95)))

    frames = []
    for lon in -np.linspace(0, 360, n_frames, endpoint=False):  # view moves west, so the surface turns east
        ax = horizontal_circulation(sim, projection='orthographic', centre=(lat, lon), style=style, **kwargs)
        buffer = io.BytesIO()
        ax.figure.savefig(buffer, format='png', dpi=dpi)
        plt.close(ax.figure)
        frames.append(Image.open(buffer).convert('RGB'))
    _save_animation(frames, path, fps)
    return path


if __name__ == '__main__':

    data = SimulationData('output/ocean/deepeq_P_10_dT_30.0_D_1.nc')

    horizontal_circulation(data, style='quiver', projection='mollweide', colour='T')
    equatorial_section(data, style='quiver', polar=True, colour='T')
    meridional_section(data, style='quiver', colour='u', landscape=True)
    horizontal_circulation(data, style='quiver', projection='orthographic', colour='u', centre=(30, 60))
    # rotating_globe(data, 'figures/globe.mp4', colour='T', substellar=False, n_frames=240, fps=24) 
    
    plt.show()
