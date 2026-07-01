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


def _is_atmosphere(ds):
    return 'temp' in ds and 'layer' in ds.dims


class SimulationData:
    def __init__(self, filename, lat_bin=False):
        try:
            ds = xr.open_dataset(filename, decode_times=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find '{filename}'.")

        print(f"t_final = {np.array(ds['time'])[-1] / (3600 * 24 * 365.25)} yrs")

        if _is_atmosphere(ds):
            self._load_atmosphere(ds, lat_bin)
        else:
            self._load_ocean(ds, lat_bin)

    def _load_ocean(self, ds, lat_bin):
        self.sim_type = 'ocean'

        T_raw = ds['T'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        u_raw = ds['u'].isel(time=slice(-10, None)).mean(dim='time').squeeze()

        lon = ds[T_raw.dims[-1]]
        lat = ds[T_raw.dims[-2]]

        z_center_dim = [d for d in ds.dims if d.startswith('z') and 'c' in d][0]
        z = ds[z_center_dim].values

        if 'v' not in ds or 'w' not in ds:
            raise ValueError("Ocean output is missing 'v' or 'w'.")

        v_raw = ds['v'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        w_raw = ds['w'].isel(time=slice(-10, None)).mean(dim='time').squeeze()

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

        T_raw = ds['temp'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        u_raw = ds['u'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        v_raw = ds['v'].isel(time=slice(-10, None)).mean(dim='time').squeeze()

        # SpeedyWeather outputs on a regular lat/lon grid — no face interpolation needed
        self.lon  = np.array(ds['lon'])
        self.lat  = np.array(ds['lat'])
        self.z    = np.array(ds['layer'])   # sigma pressure levels (0–1)
        self.u    = np.array(u_raw)
        self.v    = np.array(v_raw)
        self.w    = None                    # not output by SpeedyWeather
        self.T    = np.array(T_raw)
        self.mslp = np.array(ds['mslp'].isel(time=slice(-10, None)).mean(dim='time').squeeze()) if 'mslp' in ds else None
        self.vor  = np.array(ds['vor'].isel(time=slice(-10, None)).mean(dim='time').squeeze())  if 'vor'  in ds else None

        print('---- SHAPES ----')
        self._print_shapes()

        if lat_bin:
            self.lon = rebin(self.lon, [1])
            self.lat = rebin(self.lat, [2])
            self.z   = rebin(self.z,   [1])
            self.u   = rebin(self.u,   [1, 2, 1])
            self.v   = rebin(self.v,   [1, 2, 1])
            self.T   = rebin(self.T,   [1, 2, 1])
            if self.mslp is not None:
                self.mslp = rebin(self.mslp, [2, 1])
            if self.vor is not None:
                self.vor = rebin(self.vor, [1, 2, 1])

            print('-- NEW SHAPES --')
            self._print_shapes()

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
