import sys

from analysis import simulation_reader
from analysis.paths import OCEAN_DIR

import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Slider, Span, LinearColorMapper, ColorBar, Select,
    BasicTickFormatter, FixedTicker,
)
from bokeh.layouts import gridplot, column, row
from bokeh.io import curdoc
from bokeh.palettes import Magma256, PiYG11, Turbo256


def _expand_palette(hex_colors, n):
    """Smoothly upsample a small hex palette to n colors via per-channel interpolation."""
    stops = np.linspace(0, 1, len(hex_colors))
    targets = np.linspace(0, 1, n)
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in hex_colors])
    out = np.stack([np.interp(targets, stops, rgb[:, ch]) for ch in range(3)], axis=1)
    out = out.round().astype(int).clip(0, 255)
    return tuple(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in out)


# Diverging (pink-white-green) scale for velocity/vorticity fields, smoothly
# interpolated from the 11-stop Brewer palette up to 256 levels so nearby
# values are distinguishable instead of being quantized into ~10 bins.
VELOCITY_PALETTE = _expand_palette(PiYG11[::-1], 256)


class InteractivePlot:
    FIG_W_XY, FIG_H_XY = 900, 600
    FIG_W_XZ, FIG_H_XZ = 900, 450
    FIG_W_YZ, FIG_H_YZ = 450, 600

    # Fields shown on a zero-centered symmetric-log color scale rather than
    # linearly, so small-magnitude motion isn't washed out by a few extreme
    # values. Temperature/pressure stay linear since they aren't sign-symmetric.
    DIVERGING_FIELDS = {
        "Zonal Velocity (u)", "Meridional Velocity (v)",
        "Vertical Velocity (w)", "Vorticity (vor)", "Radial Velocity (v_r)",
    }

    # Both sim setups (ocean_sim.jl, atmosphere_sim.jl) place the substellar
    # point at lon=0, lat=0 (insolation ~ cos(lon)*cos(lat)); radial velocity
    # is defined relative to that point.
    SUBSTELLAR_LON, SUBSTELLAR_LAT = 0.0, 0.0

    def __init__(self, sim_path, time=slice(-10, None), w_shapiro_passes=1):
        sim = simulation_reader.SimulationData(sim_path, time=time)
        self.lon, self.lat, self.z = sim.lon, sim.lat, sim.z
        self.u, self.v, self.T = sim.u, sim.v, sim.T
        # The diagnostic vertical velocity carries grid-scale (2Δ) checkerboard noise
        # from the horizontal-divergence operator, strongest at the equatorial/substellar
        # convergence zones (u, v, T stay clean). Filter only w, and only for display —
        # sim.w is left untouched for any quantitative use. The Shapiro filter removes the
        # 2Δ mode while keeping full resolution, unlike rebin. Set w_shapiro_passes=0 to disable.
        if sim.w is not None and w_shapiro_passes > 0:
            self.w = simulation_reader.shapiro_filter(sim.w, order=2, passes=w_shapiro_passes)
        else:
            self.w = sim.w
        self.sim = sim

        self.is_atmosphere = sim.sim_type == 'atmosphere'
        self.z_label = "Pressure Level (σ)" if self.is_atmosphere else "Depth (m)"

        self.skip_xy = max(len(sim.lon) // 90, 1)
        self.skip_z = 1

        self._build_grids()
        self._build_bg_options()
        self._build_bg_limits()
        self._init_sources()
        self._build_figures()
        self._build_spans()
        self._build_controls()
        self._wire_callbacks()

    def _build_grids(self):
        lon_sub = self.lon[::self.skip_xy]
        lat_sub = self.lat[::self.skip_xy]
        z_sub   = self.z[::self.skip_z]

        self.Lon_xy, self.Lat_xy = np.meshgrid(lon_sub, lat_sub)
        self.Lon_xz, self.Z_xz  = np.meshgrid(lon_sub, z_sub)
        self.Lat_yz, self.Z_yz  = np.meshgrid(lat_sub, z_sub)

        self.min_lon, self.max_lon = np.min(self.lon), np.max(self.lon)
        self.min_lat, self.max_lat = np.min(self.lat), np.max(self.lat)
        self.min_z,   self.max_z   = np.min(self.z),   np.max(self.z)

        # p_xz and p_yz aren't drawn with match_aspect=True, and their axes
        # mix units of very different scale (e.g. longitude vs. sigma level),
        # so a data-space arrow angle doesn't match the on-screen line slope.
        # Correct for it using each subplot's pixel-per-data-unit ratio.
        self.angle_scale_xz = (
            self.FIG_W_XZ / (self.max_lon - self.min_lon),
            self.FIG_H_XZ / (self.max_z - self.min_z),
        )
        self.angle_scale_yz = (
            self.FIG_W_YZ / (self.min_z - self.max_z),
            self.FIG_H_YZ / (self.max_lat - self.min_lat),
        )

    def _symlog(self, name, arr):
        """Sign-preserving log transform: linear near zero, log beyond linthresh."""
        abs_arr = np.abs(arr)
        nonzero = abs_arr[abs_arr > 0]
        linthresh = np.nanpercentile(nonzero, 5) if nonzero.size else 1.0
        linthresh = max(linthresh, 1e-12)
        self.bg_linthresh[name] = linthresh
        self.bg_max_abs[name] = float(np.nanmax(abs_arr)) if abs_arr.size else 0.0
        return np.sign(arr) * np.log1p(abs_arr / linthresh)

    @staticmethod
    def _decade_tick_values(max_abs, linthresh, max_per_side=4):
        """Nice round physical values (powers of ten, plus zero) spanning a symlog range."""
        if max_abs <= 0:
            return [0.0]
        start_pow = np.floor(np.log10(max(linthresh, max_abs * 1e-6)))
        end_pow = np.ceil(np.log10(max_abs))
        decades = 10.0 ** np.arange(start_pow, end_pow + 1)
        decades = decades[decades <= max_abs * 1.0001]
        if len(decades) > max_per_side:
            # keep the decades closest to the outer edge (most informative range)
            decades = decades[-max_per_side:]
        return sorted(set([0.0] + list(decades) + [-d for d in decades]))

    @staticmethod
    def _format_tick_value(v):
        if v == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(v))))
        return f"{v:g}" if -3 <= exponent <= 3 else f"{v:.0e}"

    def _symlog_ticks_overrides(self, max_abs, linthresh):
        """Build a (tick_positions, label_overrides) pair placing ticks at nice
        round physical values, positioned via the forward symlog transform. This
        makes the log-scaled nature of the color bar visible (ticks bunch up away
        from zero) while the labels report the true physical values.

        The positions are fed into the one persistent FixedTicker (see
        _update_plots) and the labels via major_label_overrides (a pure-Python
        position→label map). Both the CustomJSTickFormatter path and swapping in
        a fresh ticker object failed to re-render on an already-drawn ColorBar
        over the Bokeh server — the bar kept its original auto ticker and showed
        the raw transformed axis positions (plain integers). Mutating the
        existing ticker's .ticks in place rebinds reliably."""
        values = self._decade_tick_values(max_abs, linthresh)
        positions = [float(np.sign(v) * np.log1p(abs(v) / linthresh)) for v in values]
        overrides = {pos: self._format_tick_value(v) for pos, v in zip(positions, values)}
        return positions, overrides

    @staticmethod
    def _linear_ticks(lo, hi, n=7):
        """Nice round evenly-spaced tick positions spanning [lo, hi], for the
        linear (temperature / pressure) color bar. Fed into the same persistent
        FixedTicker; the default BasicTickFormatter renders them as-is since
        these fields aren't transformed."""
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return [float(lo)] if np.isfinite(lo) else [0.0]
        raw_step = (hi - lo) / (n - 1)
        mag = 10.0 ** np.floor(np.log10(raw_step))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw_step <= m * mag)
        start = np.ceil(lo / step) * step
        return [float(t) for t in np.arange(start, hi + step * 0.5, step)]

    def _compute_radial_velocity(self):
        """Component of the horizontal velocity pointing directly away from
        the substellar point: (u, v) projected onto the exact unit vector,
        tangent to the sphere, along the great circle away from
        (SUBSTELLAR_LON, SUBSTELLAR_LAT). Positive = outflow, negative = inflow.
        """
        lon0 = np.radians(self.SUBSTELLAR_LON)
        lat0 = np.radians(self.SUBSTELLAR_LAT)
        lon = np.radians(self.lon)[np.newaxis, :]
        lat = np.radians(self.lat)[:, np.newaxis]
        dlon = lon - lon0

        cos_rho = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
        sin_rho = np.sqrt(np.clip(1 - cos_rho ** 2, 0, None))
        # Undefined exactly at the substellar/antistellar point (no preferred direction)
        sin_rho = np.where(sin_rho < 1e-9, np.nan, sin_rho)

        e_east  = np.cos(lat0) * np.sin(dlon) / sin_rho
        e_north = (np.sin(lat) * np.cos(lat0) * np.cos(dlon) - np.sin(lat0) * np.cos(lat)) / sin_rho

        return self.u * e_east + self.v * e_north

    def _build_bg_options(self):
        self.bg_linthresh = {}
        self.bg_max_abs = {}
        self.bg_options = {
            "Temperature (T)":       self.T,
            "Zonal Velocity (u)":    self._symlog("Zonal Velocity (u)", self.u),
            "Meridional Velocity (v)": self._symlog("Meridional Velocity (v)", self.v),
        }
        if self.w is not None:
            self.bg_options["Vertical Velocity (w)"] = self._symlog("Vertical Velocity (w)", self.w)
        self.bg_options["Radial Velocity (v_r)"] = self._symlog(
            "Radial Velocity (v_r)", self._compute_radial_velocity()
        )
        if self.sim.mslp is not None:
            self.bg_options["Mean SL Pressure"] = np.tile(
                self.sim.mslp[np.newaxis, :, :], (len(self.z), 1, 1)
            )
        if self.sim.vor is not None:
            self.bg_options["Vorticity (vor)"] = self._symlog("Vorticity (vor)", self.sim.vor)

    def _build_bg_limits(self):
        self.bg_limits = {}
        for name, arr in self.bg_options.items():
            if name == "Temperature (T)" and self.is_atmosphere:
                # Fixed colorscale across the whole simulation (all z-levels)
                # rather than per-level, so colors stay comparable as the
                # z-slider moves and across the full 3D pressure profile.
                self.bg_limits[name] = (np.nanmin(arr), np.nanmax(arr))
            elif name == "Temperature (T)":
                self.bg_limits[name] = [
                    (np.nanmin(arr[k]), np.nanmax(arr[k])) for k in range(len(self.z))
                ]
            elif name == "Mean SL Pressure":
                self.bg_limits[name] = (np.nanmin(arr), np.nanmax(arr))
            else:
                # already symlog-transformed and zero-centered
                max_abs = np.nanmax(np.abs(arr))
                self.bg_limits[name] = (-max_abs, max_abs)

    def _get_color_limits(self, bg_name, z_idx):
        lims = self.bg_limits[bg_name]
        if bg_name == "Temperature (T)" and not self.is_atmosphere:
            return lims[z_idx]
        return lims

    def _palette_for(self, bg_name):
        """Rainbow palette for atmosphere temperature so nearby values are
        visually distinguishable; Magma elsewhere for linear-scale fields."""
        if bg_name == "Temperature (T)" and self.is_atmosphere:
            return Turbo256
        return Magma256

    def _get_xy_slice(self, z_idx, bg_array):
        u_sub = self.u[z_idx, ::self.skip_xy, ::self.skip_xy].flatten()
        v_sub = self.v[z_idx, ::self.skip_xy, ::self.skip_xy].flatten()
        x = self.Lon_xy.flatten()
        y = self.Lat_xy.flatten()
        scale = 5 / (np.sqrt(u_sub ** 2 + v_sub ** 2) + 1e-10)
        x1 = np.nan_to_num(x + u_sub * scale, nan=x, posinf=x, neginf=x)
        y1 = np.nan_to_num(y + v_sub * scale, nan=y, posinf=y, neginf=y)
        angle = np.arctan2(-(x1 - x), y1 - y)
        return dict(x0=x, y0=y, x1=x1, y1=y1, angle=angle), dict(img=[bg_array[z_idx, :, :]])

    def _get_xz_slice(self, lat_idx, bg_array):
        u_sub = self.u[::self.skip_z, lat_idx, ::self.skip_xy].flatten()
        x = self.Lon_xz.flatten()
        y = self.Z_xz.flatten()
        if self.w is not None:
            w_sub   = self.w[::self.skip_z, lat_idx, ::self.skip_xy].flatten()
            w_scale = 20 * np.mean(np.abs(u_sub)) / (np.mean(np.abs(w_sub)) + 1e-10)
            scale   = 20 / (np.sqrt(u_sub ** 2 + (w_sub * w_scale) ** 2) + 1e-10)
            x1 = np.nan_to_num(x + u_sub * scale,           nan=x, posinf=x, neginf=x)
            y1 = np.nan_to_num(y + w_sub * scale * w_scale, nan=y, posinf=y, neginf=y)
        else:
            scale = 5 / (np.abs(u_sub) + 1e-10)
            x1 = np.nan_to_num(x + u_sub * scale, nan=x, posinf=x, neginf=x)
            y1 = y
        sx, sy = self.angle_scale_xz
        angle = np.arctan2(-(x1 - x) * sx, (y1 - y) * sy)
        return dict(x0=x, y0=y, x1=x1, y1=y1, angle=angle), dict(img=[bg_array[:, lat_idx, :]])

    def _get_yz_slice(self, lon_idx, bg_array):
        v_sub = self.v[::self.skip_z, ::self.skip_xy, lon_idx].flatten()
        x = self.Z_yz.flatten()
        y = self.Lat_yz.flatten()
        if self.w is not None:
            w_sub   = self.w[::self.skip_z, ::self.skip_xy, lon_idx].flatten()
            w_scale = 20 * np.mean(np.abs(v_sub)) / (np.mean(np.abs(w_sub)) + 1e-10)
            scale   = 20 / (np.sqrt(v_sub ** 2 + (w_sub * w_scale) ** 2) + 1e-10)
            x1 = np.nan_to_num(x + w_sub * scale * w_scale, nan=x, posinf=x, neginf=x)
            y1 = np.nan_to_num(y + v_sub * scale,           nan=y, posinf=y, neginf=y)
        else:
            scale = 5 / (np.abs(v_sub) + 1e-10)
            x1 = x
            y1 = np.nan_to_num(y + v_sub * scale, nan=y, posinf=y, neginf=y)
        sx, sy = self.angle_scale_yz
        angle = np.arctan2(-(x1 - x) * sx, (y1 - y) * sy)
        return dict(x0=x, y0=y, x1=x1, y1=y1, angle=angle), dict(img=[bg_array[:, :, lon_idx].T])

    def _init_sources(self):
        init_bg_name = "Temperature (T)"
        init_z_idx   = len(self.z) - 1
        lo, hi = self._get_color_limits(init_bg_name, init_z_idx)
        self.color_mapper = LinearColorMapper(palette=self._palette_for(init_bg_name), low=lo, high=hi)

        bg = self.bg_options[init_bg_name]
        vec_xy, img_xy = self._get_xy_slice(init_z_idx, bg)
        vec_xz, img_xz = self._get_xz_slice(len(self.lat) // 2, bg)
        vec_yz, img_yz = self._get_yz_slice(len(self.lon) // 2, bg)

        self.source_xy_vec = ColumnDataSource(data=vec_xy)
        self.source_xy_img = ColumnDataSource(data=img_xy)
        self.source_xz_vec = ColumnDataSource(data=vec_xz)
        self.source_xz_img = ColumnDataSource(data=img_xz)
        self.source_yz_vec = ColumnDataSource(data=vec_yz)
        self.source_yz_img = ColumnDataSource(data=img_yz)

    def _build_figures(self):
        ml, xl = self.min_lon, self.max_lon
        mla, xla = self.min_lat, self.max_lat
        mz, xz = self.min_z, self.max_z
        cm = self.color_mapper

        self.p_xy = figure(
            width=self.FIG_W_XY, height=self.FIG_H_XY, match_aspect=True, x_axis_location='above',
            x_range=(ml, xl), y_range=(mla, xla), min_border_bottom=0,
        )
        self.p_xy.image(image='img', x=ml, y=mla, dw=xl-ml, dh=xla-mla,
                        source=self.source_xy_img, color_mapper=cm, alpha=0.6)
        self.p_xy.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                          source=self.source_xy_vec, color="black", line_width=1.0)
        self.p_xy.scatter(x='x1', y='y1', source=self.source_xy_vec, marker="triangle",
                          angle='angle', size=6, color="black", alpha=0.5)
        # One persistent ticker whose .ticks we mutate in place on every field
        # switch. Replacing the ticker object (or using a CustomJSTickFormatter)
        # did not re-render on the live ColorBar; mutating this model's props does.
        init_lo, init_hi = self._get_color_limits("Temperature (T)", len(self.z) - 1)
        self.cbar_ticker = FixedTicker(ticks=self._linear_ticks(init_lo, init_hi))
        self.color_bar = ColorBar(color_mapper=cm, title="Temp", location=(0, 0),
                                   ticker=self.cbar_ticker, formatter=BasicTickFormatter())
        self.p_xy.add_layout(self.color_bar, 'left')

        self.p_xz = figure(
            x_axis_label="Longitude", y_axis_label=self.z_label,
            width=self.FIG_W_XZ, height=self.FIG_H_XZ,
            x_range=self.p_xy.x_range, y_range=(mz, xz), min_border_top=0,
        )
        self.p_xz.image(image='img', x=ml, y=mz, dw=xl-ml, dh=xz-mz,
                        source=self.source_xz_img, color_mapper=cm, alpha=0.6)
        self.p_xz.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                          source=self.source_xz_vec, color="black", line_width=1.0)
        self.p_xz.scatter(x='x1', y='y1', source=self.source_xz_vec, marker="triangle",
                          angle='angle', size=6, color="black", alpha=0.5)

        self.p_yz = figure(
            x_axis_label=self.z_label, y_axis_label="Latitude", y_axis_location="right",
            width=self.FIG_W_YZ, height=self.FIG_H_YZ,
            x_range=(xz, mz), y_range=self.p_xy.y_range,
        )
        self.p_yz.image(image='img', x=mz, y=mla, dw=xz-mz, dh=xla-mla,
                        source=self.source_yz_img, color_mapper=cm, alpha=0.6)
        self.p_yz.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                          source=self.source_yz_vec, color="black", line_width=1.0)
        self.p_yz.scatter(x='x1', y='y1', source=self.source_yz_vec, marker="triangle",
                          angle='angle', size=6, color="black", alpha=0.5)

    def _build_spans(self):
        init_z   = self.z[-1]
        init_lat = self.lat[len(self.lat) // 2]
        init_lon = self.lon[len(self.lon) // 2]

        self.span_xy_lon = Span(location=init_lon, dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
        self.span_xy_lat = Span(location=init_lat, dimension='width',  line_color='black', line_dash='dashed', line_width=1.5)
        self.p_xy.add_layout(self.span_xy_lon)
        self.p_xy.add_layout(self.span_xy_lat)

        self.span_xz_lon = Span(location=init_lon, dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
        self.span_xz_z   = Span(location=init_z,   dimension='width',  line_color='black', line_dash='dashed', line_width=1.5)
        self.p_xz.add_layout(self.span_xz_lon)
        self.p_xz.add_layout(self.span_xz_z)

        self.span_yz_z   = Span(location=init_z,   dimension='height', line_color='black', line_dash='dashed', line_width=1.5)
        self.span_yz_lat = Span(location=init_lat, dimension='width',  line_color='black', line_dash='dashed', line_width=1.5)
        self.p_yz.add_layout(self.span_yz_z)
        self.p_yz.add_layout(self.span_yz_lat)

    def _build_controls(self):
        self.select_bg = Select(
            title="Background Variable:",
            value="Temperature (T)",
            options=list(self.bg_options.keys()),
        )
        self.slider_z   = Slider(start=0, end=len(self.z)-1,   value=len(self.z)-1,      step=1, title=f"{self.z_label} Index (for Lat-Lon plot)")
        self.slider_lat = Slider(start=0, end=len(self.lat)-1, value=len(self.lat) // 2, step=1, title="Latitude Index (for Lon-Z plot)")
        self.slider_lon = Slider(start=0, end=len(self.lon)-1, value=len(self.lon) // 2, step=1, title="Longitude Index (for Lat-Z plot)")

    def _wire_callbacks(self):
        for widget in (self.select_bg, self.slider_z, self.slider_lat, self.slider_lon):
            widget.on_change('value', self._update_plots)

    def _update_plots(self, attr, old, new):
        try:
            z_idx   = int(self.slider_z.value)
            lat_idx = int(self.slider_lat.value)
            lon_idx = int(self.slider_lon.value)

            bg_name  = self.select_bg.value
            bg_array = self.bg_options[bg_name]

            lo, hi = self._get_color_limits(bg_name, z_idx)
            self.color_mapper.low, self.color_mapper.high = lo, hi

            if bg_name in ("Temperature (T)", "Mean SL Pressure"):
                self.color_mapper.palette = self._palette_for(bg_name)
                self.cbar_ticker.ticks = self._linear_ticks(lo, hi)
                self.color_bar.major_label_overrides = {}
            else:
                self.color_mapper.palette = VELOCITY_PALETTE
                positions, overrides = self._symlog_ticks_overrides(
                    self.bg_max_abs[bg_name], self.bg_linthresh[bg_name]
                )
                self.cbar_ticker.ticks = positions
                self.color_bar.major_label_overrides = overrides
            self.color_bar.title = bg_name.split(" ")[0]

            vec_xy, img_xy = self._get_xy_slice(z_idx, bg_array)
            vec_xz, img_xz = self._get_xz_slice(lat_idx, bg_array)
            vec_yz, img_yz = self._get_yz_slice(lon_idx, bg_array)

            self.source_xy_vec.data = vec_xy
            self.source_xz_vec.data = vec_xz
            self.source_yz_vec.data = vec_yz
            self.source_xy_img.data = img_xy
            self.source_xz_img.data = img_xz
            self.source_yz_img.data = img_yz

            self.span_xy_lon.location = self.lon[lon_idx]
            self.span_xy_lat.location = self.lat[lat_idx]
            self.span_xz_lon.location = self.lon[lon_idx]
            self.span_xz_z.location   = self.z[z_idx]
            self.span_yz_z.location   = self.z[z_idx]
            self.span_yz_lat.location = self.lat[lat_idx]

        except Exception as e:
            print(f"Callback Error: {e}")

    def get_layout(self):
        plot_grid = gridplot(
            [[self.p_xy, self.p_yz], [self.p_xz, None]],
            toolbar_location="right",
        )
        return column(
            row(self.select_bg, self.slider_z, self.slider_lat, self.slider_lon),
            plot_grid,
        )


# Bokeh app: bokeh serve --show analysis/visualisation.py --args <output .nc file>  (bokeh names the app module bokeh_app_*)
if __name__.startswith("bokeh_app_"):
    sim_file = sys.argv[1] if len(sys.argv) > 1 else str(OCEAN_DIR / "coriolis_P_10_dT_0.1_D_1.nc")
    interactive_plot = InteractivePlot(sim_file)
    curdoc().add_root(interactive_plot.get_layout())
    curdoc().title = "3D Quiver Plots"
