import xarray as xr
import numpy as np


def rebin(arr, factors):
    """
    Rebins an array of any dimensionality by the given factors.
    factors: A tuple of integers representing the reduction factor for each axis.
             e.g., (1, 2, 2) reduces a 3D array's last two dimensions by half.
    """
    arr = np.asarray(arr)

    if len(factors) != arr.ndim:
        raise ValueError(f"Factors {factors} must match array dimensions {arr.ndim}")

    intermediate_shape = []
    for dim_size, f in zip(arr.shape, factors):
        if dim_size % f != 0:
            raise ValueError(f"Dimension size {dim_size} is not divisible by factor {f}")
        intermediate_shape.extend([dim_size // f, f])

    reshaped_arr = arr.reshape(intermediate_shape)
    reduce_axes = tuple(range(1, len(intermediate_shape), 2))
    return reshaped_arr.mean(axis=reduce_axes)


def _shapiro_1d(f, axis, order, periodic):
    """One Shapiro pass of half-order `order` along a single axis (see shapiro_filter)."""
    mode = 'wrap' if periodic else 'edge'
    pad_width = [(1, 1) if i == axis else (0, 0) for i in range(f.ndim)]
    lo = tuple(slice(0, -2) if i == axis else slice(None) for i in range(f.ndim))
    hi = tuple(slice(2, None) if i == axis else slice(None) for i in range(f.ndim))

    def second_diff_over_4(x):
        xp = np.pad(x, pad_width, mode=mode)
        return (xp[hi] - 2.0 * x + xp[lo]) / 4.0

    # S = δ²/4 has Fourier response -sin²(kΔ/2); applying it `order` times and forming
    # f - (-1)**order · Sᵒʳᵈᵉʳ f gives response (1 - sin(kΔ/2)**(2·order)): the 2Δ mode
    # (kΔ = π) is removed exactly, and larger `order` leaves resolved scales less damped.
    Sp = f
    for _ in range(order):
        Sp = second_diff_over_4(Sp)
    return f - ((-1) ** order) * Sp


def shapiro_filter(arr, order=2, passes=1, axes=(-2, -1), periodic=(-1,)):
    """
    Shapiro filter: removes 2-grid-length (2Δ) noise while preserving the resolved
    scales, without coarsening the grid (unlike `rebin`).

    The filter response is R(k) = 1 - sin(kΔ/2)**(2*order): it zeroes the 2Δ checkerboard
    mode exactly, and higher `order` gives a sharper cutoff that leaves well-resolved
    scales progressively less damped. `order=1` is the classic 1-2-1 filter.

    arr      : array to filter, e.g. an ocean field with dims (z, lat, lon).
    order    : half-order of the filter (>=1); response ~ 1 - sin**(2*order).
    passes   : number of times to apply the whole filter (stronger smoothing).
    axes     : axes to filter along; default = the last two (lat, lon).
    periodic : axes treated as periodic (wrap); the rest use replicate (edge)
               boundaries. Default: the last axis (longitude) is periodic.
    """
    out = np.array(arr, dtype=float)
    axes = [a % out.ndim for a in axes]
    periodic = {a % out.ndim for a in periodic}
    for _ in range(passes):
        for ax in axes:
            out = _shapiro_1d(out, ax, order, ax in periodic)
    return out


def _is_atmosphere(ds):
    return 'temp' in ds and 'layer' in ds.dims


def _select_time(da, time_sel):
    """Select a single snapshot (int index) or average over a range (slice)."""
    if isinstance(time_sel, (int, np.integer)):
        return da.isel(time=time_sel)
    return da.isel(time=time_sel).mean(dim='time')


class SimulationData:
    def __init__(self, filename, lat_bin=False, time=slice(-1, None)):
        """
        filename: path to the NetCDF output file.
        time: which timestep(s) to load. Either an int (a single snapshot,
              e.g. -1 for the last one, 0 for the first) or a slice
              (a range of snapshots that will be averaged, e.g. slice(-10, None)).
        """
        try:
            ds = xr.open_dataset(filename, decode_times=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find '{filename}'.")

        time_units = ds['time'].attrs.get('units', 'seconds')
        seconds_per_unit = 3600 if time_units.startswith('hours') else 1
        times = np.array(ds['time']) * seconds_per_unit
        print(f"t_final = {times[-1] / (3600 * 24)} days")

        if isinstance(time, (int, np.integer)):
            t_sel = times[time]
            print(f"Loading snapshot at index {time} (t = {t_sel / (3600 * 24 * 365.25):.4f} yrs)")
        else:
            t_sel = times[time]
            print(f"Averaging {len(t_sel)} snapshots from t = {t_sel[0] / (3600 * 24 * 365.25):.4f} "
                  f"to {t_sel[-1] / (3600 * 24 * 365.25):.4f} yrs")

        self.time_sel = time

        if _is_atmosphere(ds):
            self._load_atmosphere(ds, lat_bin)
        else:
            self._load_ocean(ds, lat_bin)

    def _load_ocean(self, ds, lat_bin):
        self.sim_type = 'ocean'

        T_raw = _select_time(ds['T'], self.time_sel).squeeze()
        u_raw = _select_time(ds['u'], self.time_sel).squeeze()

        lon = ds[T_raw.dims[-1]]
        lat = ds[T_raw.dims[-2]]

        z_center_dim = [d for d in ds.dims if d.startswith('z') and 'c' in d][0]
        z = ds[z_center_dim].values

        if 'v' not in ds or 'w' not in ds:
            raise ValueError("Ocean output is missing 'v' or 'w'.")

        v_raw = _select_time(ds['v'], self.time_sel).squeeze()
        w_raw = _select_time(ds['w'], self.time_sel).squeeze()

        # Interpolate v from y-faces to y-centers if needed
        if v_raw.shape[1] > lat.shape[0]:
            v = 0.5 * (v_raw.values[:, :-1, :] + v_raw.values[:, 1:, :])
        else:
            v = v_raw.values

        # Interpolate w from z-faces to z-centers if needed
        if w_raw.shape[0] > z.shape[0]:
            w = 0.5 * (w_raw.values[:-1, :, :] + w_raw.values[1:, :, :])
        else:
            w = w_raw.values

        self.lon  = np.array(lon)
        self.lat  = np.array(lat)
        self.z    = np.array(z)
        self.radius = float(ds['Δy_cca'].values.flat[0] / np.radians(np.diff(self.lat[:2])[0]))  # planet radius from grid metrics
        self.u    = np.array(u_raw)
        self.v    = v
        self.w    = w
        self.T    = np.array(T_raw)
        self.mslp = None
        self.vor  = None

        print('---- SHAPES ----')
        self._print_shapes()

        if lat_bin:
            self.lon = rebin(self.lon, [1])
            self.lat = rebin(self.lat, [2])
            self.z   = rebin(self.z,   [1])
            self.u   = rebin(self.u,   [1, 2, 1])
            self.v   = rebin(self.v,   [1, 2, 1])
            self.w   = rebin(self.w,   [1, 2, 1])
            self.T   = rebin(self.T,   [1, 2, 1])

            print('-- NEW SHAPES --')
            self._print_shapes()

    def _load_atmosphere(self, ds, lat_bin):
        self.sim_type = 'atmosphere'

        T_raw = _select_time(ds['temp'], self.time_sel).squeeze()
        u_raw = _select_time(ds['u'], self.time_sel).squeeze()
        v_raw = _select_time(ds['v'], self.time_sel).squeeze()

        # SpeedyWeather outputs on a regular lat/lon grid — no face interpolation needed
        self.lon  = np.array(ds['lon'])
        self.lat  = np.array(ds['lat'])
        self.z    = np.array(ds['layer'])   # sigma pressure levels (0–1)
        self.u    = np.array(u_raw)
        self.v    = np.array(v_raw)
        self.T    = np.array(T_raw)
        self.mslp = np.array(_select_time(ds['mslp'], self.time_sel).squeeze()) if 'mslp' in ds else None
        self.vor  = np.array(_select_time(ds['vor'], self.time_sel).squeeze())  if 'vor'  in ds else None

        # Derive vertical velocity in sigma coordinates from horizontal divergence
        self.w = self._compute_sigma_dot()

        print('---- SHAPES ----')
        self._print_shapes()

        if lat_bin:
            self.lon = rebin(self.lon, [1])
            self.lat = rebin(self.lat, [2])
            self.z   = rebin(self.z,   [1])
            self.u   = rebin(self.u,   [1, 2, 1])
            self.v   = rebin(self.v,   [1, 2, 1])
            self.w   = rebin(self.w,   [1, 2, 1])
            self.T   = rebin(self.T,   [1, 2, 1])
            if self.mslp is not None:
                self.mslp = rebin(self.mslp, [2, 1])
            if self.vor is not None:
                self.vor = rebin(self.vor, [1, 2, 1])

            print('-- NEW SHAPES --')
            self._print_shapes()

    def _compute_sigma_dot(self):
        """Estimate vertical velocity in sigma coordinates from the continuity equation.

        sigma_dot(sigma) = -integral_0^sigma div_h dsigma'
        div_h = (1 / (R cos phi)) * (du/dlambda + d(v cos phi)/dphi)

        Returns w with sign convention: positive = upward (decreasing sigma),
        negative = downward, matching the ocean w convention used by the visualisation.
        Shape: (n_layer, n_lat, n_lon).
        """
        R = 6.371e6  # Earth radius, m
        lon_r = np.radians(self.lon)   # (n_lon,)  uniform spacing assumed
        lat_r = np.radians(self.lat)   # (n_lat,)  N→S (decreasing)
        cos_lat = np.cos(lat_r)        # (n_lat,)

        # Zonal derivative with periodic longitude wrap
        dlon = lon_r[1] - lon_r[0]
        du_dlon = (np.roll(self.u, -1, axis=2) - np.roll(self.u, 1, axis=2)) / (2 * dlon)

        # Meridional derivative of (v * cos lat); np.gradient handles non-uniform spacing
        v_cos = self.v * cos_lat[np.newaxis, :, np.newaxis]
        dv_cos_dlat = np.gradient(v_cos, lat_r, axis=1)

        # Horizontal divergence; guard against cos→0 at exact poles
        cos_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
        div_h = (du_dlon + dv_cos_dlat) / (R * cos_safe[np.newaxis, :, np.newaxis])

        # Integrate downward from the top (sigma=0) using the midpoint rule
        sigma = self.z                   # (n_layer,) increasing 0→1
        d_sigma = np.gradient(sigma)     # layer widths in sigma

        sigma_dot = np.zeros_like(div_h)
        cumsum = np.zeros(div_h.shape[1:])
        for k in range(len(sigma)):
            sigma_dot[k] = -(cumsum + 0.5 * div_h[k] * d_sigma[k])
            cumsum = cumsum + div_h[k] * d_sigma[k]

        # Negate so that positive w = upward (decreasing sigma), matching ocean convention
        return -sigma_dot

    def _print_shapes(self):
        print(f'sim_type: {self.sim_type}')
        print(f'lon: {self.lon.shape}')
        print(f'lat: {self.lat.shape}')
        print(f'z:   {self.z.shape}')
        print(f'u:   {self.u.shape}')
        print(f'v:   {self.v.shape}')
        print(f'w:   {self.w.shape if self.w is not None else "N/A"}')
        print(f'T:   {self.T.shape}')
        if self.mslp is not None:
            print(f'mslp:{self.mslp.shape}')
        if self.vor is not None:
            print(f'vor: {self.vor.shape}')
