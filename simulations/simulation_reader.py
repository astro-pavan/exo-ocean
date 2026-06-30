import xarray as xr
import numpy as np

def rebin(arr, factors):
    """
    Rebins an array of any dimensionality by the given factors.
    factors: A tuple of integers representing the reduction factor for each axis.
             e.g., (1, 2, 2) reduces a 3D array's last two dimensions by half.
    """
    # Ensure the input is a NumPy array
    arr = np.asarray(arr)
    
    if len(factors) != arr.ndim:
        raise ValueError(f"Factors {factors} must match array dimensions {arr.ndim}")

    intermediate_shape = []
    for dim_size, f in zip(arr.shape, factors):
        # Prevent reshaping errors by enforcing perfect divisibility
        if dim_size % f != 0:
            raise ValueError(f"Dimension size {dim_size} is not divisible by factor {f}")
        
        intermediate_shape.extend([dim_size // f, f])
    
    # Reshape into the expanded form
    reshaped_arr = arr.reshape(intermediate_shape)
    
    # Identify axes to average (1, 3, 5...)
    reduce_axes = tuple(range(1, len(intermediate_shape), 2))
    
    return reshaped_arr.mean(axis=reduce_axes)


def import_data(filename, lat_bin=False):
    try:
        ds = xr.open_dataset(filename, decode_times=False)
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'.")
        exit()
    
    print(f"t_final = {np.array(ds['time'])[-1] / (3600 * 24 * 365.25)} yrs")

    # T = ds['T'].isel(time=-1).squeeze()
    T = ds['T'].isel(time=slice(-10, None)).mean(dim='time').squeeze()

    # u is already matching T's shape (20, 40, 90) due to periodic longitude
    # u = ds['u'].isel(time=-1).squeeze().values
    u = ds['u'].isel(time=slice(-10, None)).mean(dim='time').squeeze()

    # Get coordinates based on T (Cell Centers)
    lon = ds[T.dims[-1]]
    lat = ds[T.dims[-2]]

    z_center_dim = [d for d in ds.dims if d.startswith('z') and 'c' in d][0]
    z = ds[z_center_dim].values

    if 'v' in ds and 'w' in ds:
        # v_raw = ds['v'].isel(time=-1).squeeze()
        # w_raw = ds['w'].isel(time=-1).squeeze()
        
        v_raw = ds['v'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        w_raw = ds['w'].isel(time=slice(-10, None)).mean(dim='time').squeeze()
        
        # FIX v: Interpolate from y-Faces (41) to y-Centers (40)
        # v_raw shape is (z, lat, lon). The lat dimension is index 1.
        if v_raw.shape[1] > lat.shape[0]:
            v = 0.5 * (v_raw.values[:, :-1, :] + v_raw.values[:, 1:, :])
        else:
            v = v_raw.values

        # FIX w: Interpolate from z-Faces (21) to z-Centers (20)
        # w_raw shape is (z, lat, lon). The z dimension is index 0.
        if w_raw.shape[0] > z.shape[0]:
            w = 0.5 * (w_raw.values[:-1, :, :] + w_raw.values[1:, :, :])
        else:
            w = w_raw.values

    else:
        print("Error: Variables 'v' or 'w' missing.")
        raise ValueError()
    
    print('---- SHAPES ----')
    print(f'lon: {lon.shape}')
    print(f'lat: {lat.shape}')
    print(f'z: {z.shape}')
    print(f'u: {u.shape}')
    print(f'v: {v.shape}')
    print(f'w: {w.shape}')
    print(f'T: {T.shape}')

    lon, lat, z, u, v, w, T = np.array(lon), np.array(lat), np.array(z), np.array(u), np.array(v), np.array(w), np.array(T)

    # lat_3d, _, _ = np.meshgrid(lat, z, lon)
    # v = np.sign(lat_3d) * v

    if lat_bin:
        lon = rebin(lon, [1])
        lat = rebin(lat, [2])
        z = rebin(z, [1])
        u = rebin(u, [1, 2, 1])
        v = rebin(v, [1, 2, 1])
        w = rebin(w, [1, 2, 1])
        T = rebin(T, [1, 2, 1])

        print('-- NEW SHAPES --')
        print(f'lon: {lon.shape}')
        print(f'lat: {lat.shape}')
        print(f'z: {z.shape}')
        print(f'u: {u.shape}')
        print(f'v: {v.shape}')
        print(f'w: {w.shape}')
        print(f'T: {T.shape}')
    
    return lon, lat, z, u, v, w, T

    