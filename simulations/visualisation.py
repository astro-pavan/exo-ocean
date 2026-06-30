import reader

import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Span, LinearColorMapper, ColorBar, Select
from bokeh.layouts import gridplot, column, row
from bokeh.io import curdoc
from bokeh.palettes import Magma256, RdBu11, PiYG11

lon, lat, z, u, v, w, T = reader.import_data('oceananigans/runs/aquaplanet_15_1000_2.nc')

skip_xy = 3
skip_z = 1 # Z only has 20 points, so we might not need to skip

lon_sub = lon[::skip_xy]
lat_sub = lat[::skip_xy]
z_sub = z[::skip_z]

# Create 2D coordinate grids for all three planes
Lon_xy, Lat_xy = np.meshgrid(lon_sub, lat_sub)
Lon_xz, Z_xz = np.meshgrid(lon_sub, z_sub)
Lat_yz, Z_yz = np.meshgrid(lat_sub, z_sub)

min_lon, max_lon = np.min(lon), np.max(lon)
min_lat, max_lat = np.min(lat), np.max(lat)
min_z, max_z = np.min(z), np.max(z)

# --- NEW: Setup dictionaries for the background variables and their limits ---
bg_options = {
    "Temperature (T)": T,
    "Zonal Velocity (u)": u,
    "Meridional Velocity (v)": v,
    "Vertical Velocity (w)": w
}

# Pre-calculate min/max for all variables.
# Velocities get symmetric limits (-max to +max) so 0 is exactly in the middle.
bg_limits = {}
for name, arr in bg_options.items():
    if name == "Temperature (T)":
        bg_limits[name] = (np.nanmin(arr), np.nanmax(arr))
    elif name == "Vertical Velocity (w)":
        max_abs = np.nanmax(np.abs(arr))
        bg_limits[name] = (-max_abs / 20, max_abs / 20)
    else:
        max_abs = np.nanmax(np.abs(arr))
        bg_limits[name] = (-max_abs, max_abs)

# Initialize Color Mapper with default Temperature limits
init_bg_name = "Temperature (T)"
color_mapper = LinearColorMapper(palette=Magma256, low=bg_limits[init_bg_name][0], high=bg_limits[init_bg_name][1])

# --- Modified Slice Functions (Now accept bg_array) ---
def get_xy_slice(z_idx, bg_array):
    """Lat-Lon plane (Main Plot)"""
    u_sub = u[z_idx, ::skip_xy, ::skip_xy].flatten()
    v_sub = v[z_idx, ::skip_xy, ::skip_xy].flatten()
    x = Lon_xy.flatten()
    y = Lat_xy.flatten()

    scale = 5 / (np.sqrt(u_sub ** 2 + v_sub ** 2) + 1e-10)
    
    x1 = np.nan_to_num(x + (u_sub * scale), nan=x, posinf=x, neginf=x)
    y1 = np.nan_to_num(y + (v_sub * scale), nan=y, posinf=y, neginf=y)
    
    # Extract image from whatever array was passed in
    bg_img = bg_array[z_idx, :, :]
    
    return dict(x0=x, y0=y, x1=x1, y1=y1), dict(img=[bg_img])

def get_xz_slice(lat_idx, bg_array):
    """Lon-Z plane (Bottom Plot)"""
    u_sub = u[::skip_z, lat_idx, ::skip_xy].flatten()
    w_sub = w[::skip_z, lat_idx, ::skip_xy].flatten()
    x = Lon_xz.flatten()
    y = Z_xz.flatten()

    w_scale_multiplier = 20 * np.mean(np.abs(u_sub)) / np.mean(np.abs(w_sub))
    scale = 20 / (np.sqrt((u_sub ** 2 + (w_sub * w_scale_multiplier) ** 2)) + 1e-10)
    
    x1 = np.nan_to_num(x + (u_sub * scale), nan=x, posinf=x, neginf=x)
    y1 = np.nan_to_num(y + (w_sub * scale * w_scale_multiplier), nan=y, posinf=y, neginf=y)
    
    bg_img = bg_array[:, lat_idx, :]
    
    return dict(x0=x, y0=y, x1=x1, y1=y1), dict(img=[bg_img])

def get_yz_slice(lon_idx, bg_array):
    """Lat-Z plane (Right Plot) - Flipped Axes"""
    v_sub = v[::skip_z, ::skip_xy, lon_idx].flatten()
    w_sub = w[::skip_z, ::skip_xy, lon_idx].flatten()
    
    x = Z_yz.flatten()
    y = Lat_yz.flatten()

    w_scale_multiplier = 20 * np.mean(np.abs(v_sub)) / np.mean(np.abs(w_sub))
    scale = 20 / (np.sqrt((v_sub ** 2 + (w_sub * w_scale_multiplier) ** 2)) + 1e-10)
    
    x1 = np.nan_to_num(x + (w_sub * scale * w_scale_multiplier), nan=x, posinf=x, neginf=x)
    y1 = np.nan_to_num(y + (v_sub * scale), nan=y, posinf=y, neginf=y)
    
    bg_img = bg_array[:, :, lon_idx].T
    
    return dict(x0=x, y0=y, x1=x1, y1=y1), dict(img=[bg_img])

# --- Initialize Data Sources ---
current_bg_array = bg_options[init_bg_name]

vec_xy, img_xy = get_xy_slice(len(z)-1, current_bg_array)
source_xy_vec = ColumnDataSource(data=vec_xy)
source_xy_img = ColumnDataSource(data=img_xy)

vec_xz, img_xz = get_xz_slice(len(lat) // 2, current_bg_array)
source_xz_vec = ColumnDataSource(data=vec_xz)
source_xz_img = ColumnDataSource(data=img_xz)

vec_yz, img_yz = get_yz_slice(len(lon) // 2, current_bg_array)
source_yz_vec = ColumnDataSource(data=vec_yz)
source_yz_img = ColumnDataSource(data=img_yz)

# --- Build Figures & Add Images ---
p_xy = figure(width=900, height=600, match_aspect=True, x_axis_location='above',
              x_range=(min_lon, max_lon), y_range=(min_lat, max_lat), min_border_bottom=0)

p_xy.image(image='img', x=min_lon, y=min_lat, dw=max_lon-min_lon, dh=max_lat-min_lat, source=source_xy_img, color_mapper=color_mapper, alpha=0.6)
p_xy.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=source_xy_vec, color="black", line_width=1.0) 
p_xy.scatter(x='x0', y='y0', source=source_xy_vec, size=3, color="black", alpha=0.5)

color_bar = ColorBar(color_mapper=color_mapper, title="Temp", location=(0,0))
p_xy.add_layout(color_bar, 'left')

p_xz = figure(x_axis_label="Longitude", y_axis_label="Depth (z)", 
              width=900, height=450, 
              x_range=p_xy.x_range, y_range=(min_z, max_z), min_border_top=0)

p_xz.image(image='img', x=min_lon, y=min_z, dw=max_lon-min_lon, dh=max_z-min_z, source=source_xz_img, color_mapper=color_mapper, alpha=0.6)
p_xz.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=source_xz_vec, color="black", line_width=1.0)
p_xz.scatter(x='x0', y='y0', source=source_xz_vec, size=3, color="black", alpha=0.5)

p_yz = figure(x_axis_label="Depth (z)", y_axis_label="Latitude", y_axis_location="right", 
              width=450, height=600, 
              x_range=(max_z, min_z), y_range=p_xy.y_range)

p_yz.image(image='img', x=min_z, y=min_lat, dw=max_z-min_z, dh=max_lat-min_lat, source=source_yz_img, color_mapper=color_mapper, alpha=0.6)
p_yz.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=source_yz_vec, color="black", line_width=1.0)
p_yz.scatter(x='x0', y='y0', source=source_yz_vec, size=3, color="black", alpha=0.5)

# --- Spans ---
init_z = z[-1]
init_lat = lat[len(lat) // 2]
init_lon = lon[len(lon) // 2]

span_xy_lon = Span(location=init_lon, dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
span_xy_lat = Span(location=init_lat, dimension='width', line_color='black', line_dash='dashed', line_width=1.5)
p_xy.add_layout(span_xy_lon)
p_xy.add_layout(span_xy_lat)

span_xz_lon = Span(location=init_lon, dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
span_xz_z = Span(location=init_z, dimension='width', line_color='black', line_dash='dashed', line_width=1.5)
p_xz.add_layout(span_xz_lon)
p_xz.add_layout(span_xz_z)

span_yz_z = Span(location=init_z, dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
span_yz_lat = Span(location=init_lat, dimension='width', line_color='black', line_dash='dashed', line_width=1.5)
p_yz.add_layout(span_yz_z)
p_yz.add_layout(span_yz_lat)

# --- Sliders and Callbacks ---
select_bg = Select(title="Background Variable:", value=init_bg_name, options=list(bg_options.keys()))

slider_z = Slider(start=0, end=len(z)-1, value=len(z)-1, step=1, title="Depth Index (for Lat-Lon plot)")
slider_lat = Slider(start=0, end=len(lat)-1, value=len(lat) // 2, step=1, title="Latitude Index (for Lon-Z plot)")
slider_lon = Slider(start=0, end=len(lon)-1, value=len(lon) // 2, step=1, title="Longitude Index (for Lat-Z plot)")

def update_plots(attr, old, new):
    try:
        z_idx = int(slider_z.value)
        lat_idx = int(slider_lat.value)
        lon_idx = int(slider_lon.value)
        
        # Determine which array to use based on dropdown
        bg_name = select_bg.value
        active_bg_array = bg_options[bg_name]
        
        # --- NEW: Dynamically swap colormap palettes ---
        if bg_name == "Temperature (T)":
            color_mapper.palette = Magma256
        else:
            # Use RdBu11 for velocities. 
            # Use RdBu11[::-1] if you prefer positive velocities to be Red instead of Blue
            color_mapper.palette = PiYG11[::-1] 
            
        # Dynamically update the color mapper limits and ColorBar title
        color_mapper.low, color_mapper.high = bg_limits[bg_name]
        color_bar.title = bg_name.split(" ")[0] 
        
        # Pass the active array into the slice functions
        vec_xy, img_xy = get_xy_slice(z_idx, active_bg_array)
        
        # Pass the active array into the slice functions
        vec_xy, img_xy = get_xy_slice(z_idx, active_bg_array)
        vec_xz, img_xz = get_xz_slice(lat_idx, active_bg_array)
        vec_yz, img_yz = get_yz_slice(lon_idx, active_bg_array)
        
        source_xy_vec.data = vec_xy
        source_xz_vec.data = vec_xz
        source_yz_vec.data = vec_yz
        
        source_xy_img.data = img_xy
        source_xz_img.data = img_xz
        source_yz_img.data = img_yz
        
        current_z, current_lat, current_lon = z[z_idx], lat[lat_idx], lon[lon_idx]
        span_xy_lon.location = current_lon
        span_xy_lat.location = current_lat
        span_xz_lon.location = current_lon
        span_xz_z.location = current_z
        span_yz_z.location = current_z
        span_yz_lat.location = current_lat
        
    except Exception as e:
        print(f"Callback Error: {e}")

plot_grid = gridplot([
    [p_xy, p_yz],
    [p_xz, None]
], toolbar_location="right")

layout = column(
    row(select_bg, slider_z, slider_lat, slider_lon), 
    plot_grid
)

# Connect all widgets to the callback function
select_bg.on_change('value', update_plots)
slider_z.on_change('value', update_plots)
slider_lat.on_change('value', update_plots)
slider_lon.on_change('value', update_plots)

curdoc().add_root(layout)
curdoc().title = "3D Ocean Quiver Plots"