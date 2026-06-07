import re
import json
import os
import sympy as sp
from OpenGL.GL import *
import numpy as np
from util import parse_to_numpy, default_expr, default_expr_3d
from plot import DynamicLinePlot, GridGeometry, BackgroundQuad
from plot3d import SurfacePlot3D
from plot3d_parametric import ParametricSurfacePlot
from plot3d_curve import SpaceCurve3D
from plot_slot import PlotSlot
import time
from shaders import *
from shaders3d import *
from plot_slot import PLOT_EQUATION, PLOT_SCATTER, PLOT_LINE_DATA, PLOT_HISTOGRAM, PLOT_KDE
from OpenGL.GL.shaders import compileProgram, compileShader
from imgui_bundle import imgui

_HISTORY_FILE = os.path.expanduser("~/.pygraph_history.json")
_MAX_HISTORY  = 20


class AppState:
    # ── sampling constants ────────────────────────────────────────────────────
    OVERDRAW        = 2.0
    MIN_VISIBLE_PTS = 2000
    MAX_TOTAL_PTS   = 400_000
    SAMPLES_PER_PX  = 3

    # ── default slot colours ──────────────────────────────────────────────────
    _SLOT_COLORS = [
        [0.0, 0.5, 0.8], [1.0, 0.4, 0.1], [0.1, 0.8, 0.3],
        [0.8, 0.1, 0.8], [0.9, 0.8, 0.1], [0.1, 0.8, 0.9],
        [0.8, 0.3, 0.3], [0.5, 0.5, 0.5],
    ]

    def __init__(self):
        # ── geometry ─────────────────────────────────────────────────────────
        self.grid_geo    = None
        self.surface_geo = None
        self.param_geo   = None
        self.bg_geo      = None
        self.curve_geo   = None

        # ── shaders ──────────────────────────────────────────────────────────
        self.plot_solid_shader  = None
        self.plot_effect_shader = None
        self.grid_shader        = None
        self.surface_shader     = None
        self.wire_shader        = None
        self.bg_shader          = None

        # ── 2-D viewport (data-space) ─────────────────────────────────────────
        self.min_x, self.max_x = -10.0, 10.0
        self.min_y, self.max_y = -10.0, 10.0

        # ── last plot-rect in pixels ──────────────────────────────────────────
        self._last_plot_x = 0.0
        self._last_plot_y = 0.0
        self._last_plot_w = 1280.0
        self._last_plot_h = 720.0

        # ── 2-D appearance (global defaults, also used by session save) ────────
        self.graph_color     = [0.0, 0.5, 0.8]
        self.line_thickness  = 4.0
        self.resolution      = 20000
        self.last_resolution = 20000
        self.effect_mode     = 0
        self.start_time      = time.time()

        self.show_axis_grid = True
        self.show_numbers   = True
        self.connect_lines  = True

        # ── multi-plot slots ──────────────────────────────────────────────────
        self.plots: list          = []   # filled in init_gl
        self.active_plot_idx: int = 0

        # ── animation ─────────────────────────────────────────────────────────
        self.anim_time    = 0.0
        self.anim_playing = False
        self.anim_speed   = 1.0
        self.anim_enabled = False

        # ── log-scale ─────────────────────────────────────────────────────────
        self.log_scale_x = False
        self.log_scale_y = False

        # ── 3-D state ─────────────────────────────────────────────────────────
        self.mode_3d          = False
        self.mode_parametric  = False
        self.mode_space_curve = False

        # Parametric surface expressions
        self.param_expr_x    = 'u'
        self.param_expr_y    = 'f_u * cos(v)'
        self.param_expr_z    = 'f_u * sin(v)'
        self.param_f_expr    = '(sqrt(pi*u) + sin(pi*u)) / pi'
        self.param_u_range   = (0.0, 5.0)
        self.param_v_range   = (0.0, 6.2832)
        self.param_res_u     = 80
        self.param_res_v     = 80
        self.math_data_needs_update_param = False

        self.input_buf_px = 'u'
        self.input_buf_py = 'f_u * cos(v)'
        self.input_buf_pz = 'f_u * sin(v)'
        self.input_buf_pf = '(sqrt(pi*u) + sin(pi*u)) / pi'

        # ── space curve state ─────────────────────────────────────────────────
        _tx = '(2 + cos(1.5*t)) * cos(t)'
        _ty = '(2 + cos(1.5*t)) * sin(t)'
        _tz = 'sin(1.5*t)'
        self.curve_expr_x      = _tx
        self.curve_expr_y      = _ty
        self.curve_expr_z      = _tz
        self.input_buf_cx      = _tx
        self.input_buf_cy      = _ty
        self.input_buf_cz      = _tz
        self.curve_t_range     = (0.0, 12.566)
        self.curve_resolution  = 500
        self.curve_tube_radius = 0.035
        self.curve_tube_sides  = 14
        self._curve_bounds_x   = (-3.0, 3.0)
        self._curve_bounds_y   = (-3.0, 3.0)
        self._curve_bounds_z   = (-1.0, 1.0)
        self.math_data_needs_update_curve = False

        # ── 3D overlay options ────────────────────────────────────────────────
        self.show_3d_box    = True
        self.show_3d_grid   = True
        self.show_3d_labels = True
        self.resolution_3d  = 80
        self.colormap       = 0
        self.surface_alpha  = 1.0
        self.show_wireframe = True
        self.wire_alpha     = 0.25
        self.circular_mask  = False
        self.circular_max_r = 4.0

        self.cam_theta = 0.6
        self.cam_phi   = 0.8
        self.cam_dist  = 3.5
        self.cam_fov   = 45.0

        self.range3d_x = (-5.0, 5.0)
        self.range3d_y = (-5.0, 5.0)

        # ── sampled x range (for resample detection) ──────────────────────────
        self._sampled_x_start = -1e9
        self._sampled_x_end   =  1e9
        self._sampled_zoom_w  = 20.0

        # ── misc ──────────────────────────────────────────────────────────────
        self.panel_open  = True
        self.panel_width = 280

        self.math_data_needs_update   = False
        self.math_data_needs_update3d = False
        self.initialized  = False
        self.large_font   = None

        self.current_expr    = default_expr
        self.current_expr_3d = default_expr_3d

        self.input_buf_2d = default_expr
        self.input_buf_3d = default_expr_3d

        # ── error display ─────────────────────────────────────────────────────
        self.last_error: str = ""

        # ── equation history ──────────────────────────────────────────────────
        self.equation_history: list = []

        # ── ui / data helpers ─────────────────────────────────────────────────
        self._menubar_h: float = 0.0
        self._pending_file: str = ""
        self._open_file_requested: bool = False
        self.plot_fill_shader  = None
        self.plot_point_shader = None

    # ================================================================
    #  OpenGL init
    # ================================================================
    def init_gl(self):
        tmp_vao = glGenVertexArrays(1)
        glBindVertexArray(tmp_vao)

        self.plot_solid_shader = compileProgram(
            compileShader(SOLID_VERTEX_SHADER,    GL_VERTEX_SHADER),
            compileShader(SOLID_GEOMETRY_SHADER,  GL_GEOMETRY_SHADER),
            compileShader(SOLID_FRAGMENT_SHADER,  GL_FRAGMENT_SHADER),
        )
        self.plot_effect_shader = compileProgram(
            compileShader(EFFECT_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(EFFECT_GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
            compileShader(EFFECT_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.grid_shader = compileProgram(
            compileShader(GRID_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(GRID_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.surface_shader = compileProgram(
            compileShader(SURFACE_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(SURFACE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.wire_shader = compileProgram(
            compileShader(WIRE_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(WIRE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.bg_shader = compileProgram(
            compileShader(BG_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(BG_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.plot_fill_shader = compileProgram(
            compileShader(FILL_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(FILL_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.plot_point_shader = compileProgram(
            compileShader(POINT_VERTEX_SHADER,   GL_VERTEX_SHADER),
            compileShader(POINT_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        glEnable(GL_PROGRAM_POINT_SIZE)

        glBindVertexArray(0)
        glDeleteVertexArrays(1, [tmp_vao])

        # Create first default 2D plot slot
        first_slot = PlotSlot(
            expr=self.current_expr,
            input_buf=self.input_buf_2d,
            color=list(self.graph_color),
            effect_mode=self.effect_mode,
            line_thickness=self.line_thickness,
            connect_lines=self.connect_lines,
            geometry=DynamicLinePlot(),
        )
        self.plots = [first_slot]
        self.active_plot_idx = 0

        self.grid_geo    = GridGeometry()
        self.surface_geo = SurfacePlot3D()
        self.param_geo   = ParametricSurfacePlot()
        self.curve_geo   = SpaceCurve3D()
        self.bg_geo      = BackgroundQuad()

        self.generate_math_data()
        self.generate_math_data_3d(self.current_expr_3d)
        self.generate_parametric_data()
        self.generate_curve_data()
        self.initialized = True

    # ================================================================
    #  Sampling helpers
    # ================================================================
    def _compute_sample_count(self):
        plot_px      = max(self._last_plot_w, 1.0)
        visible_pts  = int(plot_px * self.SAMPLES_PER_PX)
        total_factor = 1.0 + 2.0 * self.OVERDRAW
        total_pts    = int(visible_pts * total_factor)
        min_pts      = int(self.MIN_VISIBLE_PTS * total_factor)
        return max(min_pts, min(total_pts, self.MAX_TOTAL_PTS))

    def _sample_range(self):
        """Visible range expanded by OVERDRAW (log-aware on X when enabled)."""
        if self.log_scale_x:
            lx0 = max(self.min_x, 1e-10)
            lx1 = max(self.max_x, lx0 * 1.001)
            log_half = (np.log(lx1) - np.log(lx0)) * self.OVERDRAW
            return np.exp(np.log(lx0) - log_half), np.exp(np.log(lx1) + log_half)
        half = (self.max_x - self.min_x) * self.OVERDRAW
        return self.min_x - half, self.max_x + half

    def _build_x_array(self, x_start, x_end, n):
        if self.log_scale_x:
            return np.exp(
                np.linspace(np.log(max(x_start, 1e-10)),
                            np.log(max(x_end,   1e-10)), n)
            ).astype(np.float32)
        return np.linspace(x_start, x_end, n, dtype=np.float32)

    # ================================================================
    #  2-D data generation (multi-slot)
    # ================================================================
    def generate_math_data(self, expr=None):
        """Generate data for all visible 2D slots. If expr given, updates active slot."""
        if expr is not None:
            slot = self.plots[self.active_plot_idx]
            slot.expr     = expr
            slot.input_buf = expr
            self.current_expr = expr  # keep in sync for session save

        x_start, x_end = self._sample_range()
        n = self._compute_sample_count()

        self._sampled_x_start = x_start
        self._sampled_x_end   = x_end
        self._sampled_zoom_w  = self.max_x - self.min_x

        x = self._build_x_array(x_start, x_end, n)

        for slot in self.plots:
            if not slot.visible or slot.plot_type != PLOT_EQUATION:
                continue
            if slot.geometry is None or not slot.expr:
                continue

            # Rebuild cached fn only when expression or anim mode changes
            want_anim = self.anim_enabled and 't' in slot.expr
            if (slot._cached_fn is None
                    or slot._cached_expr  != slot.expr
                    or slot._cached_anim  != want_anim):
                try:
                    var_str = 'x t' if want_anim else 'x'
                    slot._cached_fn   = parse_to_numpy(slot.expr, variables_str=var_str)
                    slot._cached_expr = slot.expr
                    slot._cached_anim = want_anim
                    slot.last_error   = ""
                except Exception as e:
                    slot.last_error   = str(e)
                    slot._cached_fn   = None
                    slot._cached_expr = slot.expr

            if slot._cached_fn is None:
                slot.geometry.update_data(x, np.zeros_like(x))
                continue

            try:
                y = slot._cached_fn(x, self.anim_time) if slot._cached_anim else slot._cached_fn(x)
                if self.log_scale_y:
                    y = np.where(y > 0, y, np.nan)
                y = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
                slot.last_error = ""
                slot._sampled_y = y
            except Exception as e:
                slot.last_error = str(e)
                y = np.zeros_like(x)
                slot._sampled_y = None
            slot.geometry.update_data(x, y)

        # Expose active slot error at top level for UI
        if self.plots:
            self.last_error = self.plots[self.active_plot_idx].last_error
        self.math_data_needs_update = False

    def needs_resample(self):
        """Resample when panned within 50% of edge or zoom changed >20%."""
        margin  = (self.max_x - self.min_x) * 0.5
        panned  = (self.min_x - margin < self._sampled_x_start or
                   self.max_x + margin > self._sampled_x_end)
        cur_w   = self.max_x - self.min_x
        zoomed  = abs(cur_w - self._sampled_zoom_w) / max(self._sampled_zoom_w, 1e-9) > 0.20
        return panned or zoomed

    # ================================================================
    #  3-D data generation
    # ================================================================
    def generate_math_data_3d(self, expr):
        self.last_error = ""
        self.current_expr_3d = expr
        n  = self.resolution_3d
        x  = np.linspace(self.range3d_x[0], self.range3d_x[1], n, dtype=np.float32)
        y  = np.linspace(self.range3d_y[0], self.range3d_y[1], n, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        try:
            fn = parse_to_numpy(expr, variables_str="x y")
            Z  = fn(X, Y).astype(np.float32)
        except Exception as e:
            self.last_error = str(e)
            Z = np.zeros_like(X, dtype=np.float32)
        max_r = self.circular_max_r if self.circular_mask else None
        self.surface_geo.update_data(x, y, Z,
                                     circular_mask=self.circular_mask,
                                     max_r=max_r)
        self.math_data_needs_update3d = False

    # ================================================================
    #  Parametric surface generation
    # ================================================================
    def generate_parametric_data(self):
        self.last_error = ""
        nu     = self.param_res_u
        nv     = self.param_res_v
        u_vals = np.linspace(self.param_u_range[0], self.param_u_range[1], nu, dtype=np.float32)
        v_vals = np.linspace(self.param_v_range[0], self.param_v_range[1], nv, dtype=np.float32)
        U, V = np.meshgrid(u_vals, v_vals)

        try:
            f_u_fn = parse_to_numpy(self.param_f_expr.strip(), variables_str="u")
            F_U    = f_u_fn(U).astype(np.float32)

            def _eval_expr(expr_str):
                s = re.sub(r'\bf_u\b', '_fw', expr_str)
                u_sym, v_sym, w_sym = sp.symbols("u v _fw")
                sym_expr = sp.sympify(s)
                fn = sp.lambdify((u_sym, v_sym, w_sym), sym_expr, modules="numpy")
                return fn(U, V, F_U)

            X = _eval_expr(self.param_expr_x).astype(np.float32)
            Y = _eval_expr(self.param_expr_y).astype(np.float32)
            Z = _eval_expr(self.param_expr_z).astype(np.float32)
        except Exception as e:
            self.last_error = str(e)
            X = U; Y = np.zeros_like(U); Z = np.zeros_like(U)

        self.param_geo.update_data(X, Y, Z)
        self.math_data_needs_update_param = False

    # ================================================================
    #  Space curve data generation
    # ================================================================
    def generate_curve_data(self):
        self.last_error = ""
        t_vals = np.linspace(self.curve_t_range[0], self.curve_t_range[1],
                             self.curve_resolution, dtype=np.float32)
        try:
            X = parse_to_numpy(self.curve_expr_x, variables_str='t')(t_vals).astype(np.float32)
            Y = parse_to_numpy(self.curve_expr_y, variables_str='t')(t_vals).astype(np.float32)
            Z = parse_to_numpy(self.curve_expr_z, variables_str='t')(t_vals).astype(np.float32)
            X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
            Y = np.where(np.isfinite(Y), Y, 0.0).astype(np.float32)
            Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)
        except Exception as e:
            self.last_error = str(e)
            X = np.cos(t_vals)
            Y = np.sin(t_vals)
            Z = np.zeros_like(t_vals)

        self._curve_bounds_x = (float(np.min(X)), float(np.max(X)))
        self._curve_bounds_y = (float(np.min(Y)), float(np.max(Y)))
        self._curve_bounds_z = (float(np.min(Z)), float(np.max(Z)))

        extent = max(
            self._curve_bounds_x[1] - self._curve_bounds_x[0],
            self._curve_bounds_y[1] - self._curve_bounds_y[0],
            self._curve_bounds_z[1] - self._curve_bounds_z[0],
            1e-9,
        )
        self.curve_geo.update_data(X, Y, Z,
                                   tube_radius=self.curve_tube_radius * extent,
                                   tube_sides=self.curve_tube_sides)
        self.math_data_needs_update_curve = False

    # ================================================================
    #  Multi-plot slot management
    # ================================================================
    def add_plot(self):
        if len(self.plots) >= 8:
            return
        idx  = len(self.plots)
        slot = PlotSlot(
            expr="",
            input_buf="",
            color=list(self._SLOT_COLORS[idx % len(self._SLOT_COLORS)]),
            geometry=DynamicLinePlot(),
        )
        self.plots.append(slot)
        self.active_plot_idx = idx

    def remove_plot(self, idx):
        if len(self.plots) <= 1:
            return
        self.plots.pop(idx)
        self.active_plot_idx = max(0, min(self.active_plot_idx, len(self.plots) - 1))

    # ================================================================
    #  Data file loading
    # ================================================================
    def load_data_file(self, path):
        from data_loader import load_file
        from plot_hist import HistogramPlot
        try:
            col_names, data = load_file(path)
        except Exception as e:
            self.last_error = f"Load error: {e}"
            return

        if len(self.plots) >= 8:
            self.last_error = "Max 8 plots – remove one first"
            return

        idx  = len(self.plots)
        slot = PlotSlot(
            expr="",
            input_buf="",
            color=list(self._SLOT_COLORS[idx % len(self._SLOT_COLORS)]),
            geometry=DynamicLinePlot(),
            hist_geo=HistogramPlot(),
            plot_type=PLOT_SCATTER,
            raw_data=data,
            col_names=col_names,
            col_x=0,
            col_y=min(1, data.shape[1] - 1),
            source_file=os.path.basename(path),
        )
        self.plots.append(slot)
        self.active_plot_idx = idx
        self.mode_3d = False
        self._update_data_slot(slot)
        self.last_error = ""

    def _update_data_slot(self, slot):
        """Recompute geometry for a data slot after column/type/bins change."""
        if slot.raw_data is None:
            return
        data   = slot.raw_data
        n_cols = data.shape[1]

        if slot.plot_type == PLOT_HISTOGRAM:
            col    = min(slot.col_hist, n_cols - 1)
            values = data[:, col]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return
            counts, edges = np.histogram(values, bins=max(2, slot.hist_bins))
            slot.hist_geo.update_data(edges, counts.astype(np.float32))
            slot._sampled_y = counts.astype(np.float32)
            max_count = float(counts.max()) if len(counts) else 1.0
            self.min_x = float(edges[0])
            self.max_x = float(edges[-1])
            self.min_y = 0.0
            self.max_y = max_count * 1.15

        elif slot.plot_type == PLOT_KDE:
            col    = min(slot.col_hist, n_cols - 1)
            values = data[:, col]
            values = values[np.isfinite(values)]
            if len(values) < 2:
                return
            # Silverman's bandwidth
            h  = 1.06 * values.std() * len(values) ** (-0.2)
            h  = max(h, 1e-9)
            lo = values.min() - 3 * h
            hi = values.max() + 3 * h
            x  = np.linspace(lo, hi, 500, dtype=np.float32)
            diff = (x[:, None] - values[None, :]) / h
            kde  = np.exp(-0.5 * diff ** 2).sum(axis=1) / (len(values) * h * np.sqrt(2 * np.pi))
            kde  = kde.astype(np.float32)
            slot.geometry.update_data(x, kde)
            slot.hist_geo.update_fill_strip(x, kde)
            slot._sampled_y = kde
            pad_x = (hi - lo) * 0.05
            self.min_x = float(lo) - pad_x
            self.max_x = float(hi) + pad_x
            self.min_y = 0.0
            self.max_y = float(kde.max()) * 1.2

        else:  # SCATTER or LINE_DATA
            cx = min(slot.col_x, n_cols - 1)
            cy = min(slot.col_y, n_cols - 1)
            x  = data[:, cx].astype(np.float32)
            y  = data[:, cy].astype(np.float32)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) == 0:
                return
            slot.geometry.update_data(x, y)
            slot._sampled_y = y
            rx = x.max() - x.min()
            ry = y.max() - y.min()
            pad_x = rx * 0.05 + 0.5
            pad_y = ry * 0.05 + 0.5
            self.min_x = float(x.min()) - pad_x
            self.max_x = float(x.max()) + pad_x
            self.min_y = float(y.min()) - pad_y
            self.max_y = float(y.max()) + pad_y

    # ================================================================
    #  Zoom-to-fit (2D, uses sampled y from all visible slots)
    # ================================================================
    def zoom_to_fit_2d(self):
        extremes = []
        for slot in self.plots:
            if slot._sampled_y is not None and slot.visible:
                finite = slot._sampled_y[np.isfinite(slot._sampled_y)]
                if len(finite) > 0:
                    extremes += [float(np.min(finite)), float(np.max(finite))]
        if not extremes:
            return
        y_lo, y_hi = min(extremes), max(extremes)
        margin_y = max((y_hi - y_lo) * 0.1, 0.5)
        x_start, x_end = self._sample_range()
        margin_x = (x_end - x_start) * 0.05
        self.min_x = x_start + margin_x
        self.max_x = x_end   - margin_x
        self.min_y = y_lo - margin_y
        self.max_y = y_hi + margin_y

    # ================================================================
    #  Equation history
    # ================================================================
    def load_history(self):
        try:
            with open(_HISTORY_FILE) as f:
                self.equation_history = json.load(f)
        except Exception:
            pass

    def add_to_history(self, expr: str):
        if expr in self.equation_history:
            self.equation_history.remove(expr)
        self.equation_history.insert(0, expr)
        self.equation_history = self.equation_history[:_MAX_HISTORY]
        try:
            with open(_HISTORY_FILE, 'w') as f:
                json.dump(self.equation_history, f)
        except Exception:
            pass

    # ================================================================
    #  Bounding box for current surface (used by overlay3d)
    # ================================================================
    def get_scene_bounds(self):
        if self.mode_space_curve:
            return self._curve_bounds_x, self._curve_bounds_y, self._curve_bounds_z
        if self.mode_parametric:
            ru = self.param_u_range[1]
            xr = (self.param_u_range[0], self.param_u_range[1])
            yr = (-ru, ru)
            zr = (-ru, ru)
        else:
            xr = (self.range3d_x[0], self.range3d_x[1])
            yr = (self.range3d_y[0], self.range3d_y[1])
            zr = (-5.0, 5.0)
        return xr, yr, zr

    # ================================================================
    #  Grid spacing
    # ================================================================
    def calculate_grid_spacing(self):
        range_x  = self.max_x - self.min_x
        exponent = np.floor(np.log10(max(abs(range_x), 1e-10)))
        base     = 10 ** exponent
        ratio    = range_x / base
        if   ratio < 2.0: spacing_x = base * 0.2
        elif ratio < 5.0: spacing_x = base * 0.5
        else:             spacing_x = base * 1.0
        return spacing_x, spacing_x

    # ================================================================
    #  Resize
    # ================================================================
    def handle_resize(self, px, py, pw, ph):
        self._last_plot_x = px
        self._last_plot_y = py
        self._last_plot_w = pw
        self._last_plot_h = ph

    # ================================================================
    #  Input processing
    # ================================================================
    def process_interactions(self, plot_rect):
        io = imgui.get_io()
        mx = io.mouse_pos.x
        my = io.mouse_pos.y
        px = plot_rect["x"];  py = plot_rect["y"]
        pw = plot_rect["w"];  ph = plot_rect["h"]

        mouse_in_plot = (px <= mx <= px + pw) and (py <= my <= py + ph)

        if io.want_capture_mouse and not mouse_in_plot:
            return

        if self.mode_3d:
            self._process_3d_interactions(io, mouse_in_plot)
        else:
            self._process_2d_interactions(io, mouse_in_plot, px, py, pw, ph)

    def _process_2d_interactions(self, io, mouse_in_plot, px, py, pw, ph):
        if mouse_in_plot and (
                imgui.is_mouse_dragging(imgui.MouseButton_.left) or
                imgui.is_mouse_dragging(imgui.MouseButton_.right)):
            dx = io.mouse_delta.x
            dy = io.mouse_delta.y
            fx = (self.max_x - self.min_x) / pw
            fy = (self.max_y - self.min_y) / ph
            self.min_x -= dx * fx;  self.max_x -= dx * fx
            self.min_y += dy * fy;  self.max_y += dy * fy

        if mouse_in_plot and io.mouse_wheel != 0.0:
            zf = 1.0 - io.mouse_wheel * 0.08
            # Zoom around mouse cursor position in data space
            mx_frac = (io.mouse_pos.x - px) / max(pw, 1.0)
            my_frac = 1.0 - (io.mouse_pos.y - py) / max(ph, 1.0)
            mx_data = self.min_x + mx_frac * (self.max_x - self.min_x)
            my_data = self.min_y + my_frac * (self.max_y - self.min_y)
            self.min_x = mx_data + (self.min_x - mx_data) * zf
            self.max_x = mx_data + (self.max_x - mx_data) * zf
            self.min_y = my_data + (self.min_y - my_data) * zf
            self.max_y = my_data + (self.max_y - my_data) * zf

    def _process_3d_interactions(self, io, mouse_in_plot):
        if mouse_in_plot and imgui.is_mouse_dragging(imgui.MouseButton_.left):
            self.cam_phi   += io.mouse_delta.x * 0.005
            self.cam_theta  = float(np.clip(
                self.cam_theta + io.mouse_delta.y * 0.005, 0.05, np.pi - 0.05))
        if mouse_in_plot and io.mouse_wheel != 0.0:
            self.cam_dist = float(np.clip(
                self.cam_dist * (1.0 - io.mouse_wheel * 0.08), 0.3, 30.0))

    # ================================================================
    #  3-D MVP
    # ================================================================
    def get_mvp(self, width, height):
        r    = self.cam_dist
        st   = np.sin(self.cam_theta);  ct  = np.cos(self.cam_theta)
        sphi = np.sin(self.cam_phi);    cphi = np.cos(self.cam_phi)
        eye    = np.array([r*st*cphi, r*st*sphi, r*ct], dtype=np.float64)
        target = np.zeros(3, dtype=np.float64)
        up     = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        f  = target - eye;  f  /= np.linalg.norm(f)
        rv = np.cross(f, up); rv /= np.linalg.norm(rv)
        uv = np.cross(rv, f)

        V = np.eye(4, dtype=np.float32)
        V[0, :3] =  rv;  V[0, 3] = -np.dot(rv, eye)
        V[1, :3] =  uv;  V[1, 3] = -np.dot(uv, eye)
        V[2, :3] = -f;   V[2, 3] =  np.dot(f,  eye)

        fov_r  = np.radians(self.cam_fov)
        aspect = width / max(height, 1)
        n, fa  = 0.01, 100.0
        fv     = 1.0 / np.tan(fov_r * 0.5)
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = fv/aspect;  P[1, 1] = fv
        P[2, 2] = (fa+n)/(n-fa);  P[2, 3] = 2*fa*n/(n-fa);  P[3, 2] = -1.0

        xr, yr, zr = self.get_scene_bounds()
        cx = (xr[0]+xr[1])*0.5;  cy = (yr[0]+yr[1])*0.5;  cz = (zr[0]+zr[1])*0.5
        rx = max(xr[1]-xr[0], 1e-9)
        ry = max(yr[1]-yr[0], 1e-9)
        rz = max(zr[1]-zr[0], 1e-9)
        sx, sy, sz = 2.0/rx, 2.0/ry, 2.0/rz
        M = np.array([
            [sx,  0,  0, -cx*sx],
            [ 0, sy,  0, -cy*sy],
            [ 0,  0, sz, -cz*sz],
            [ 0,  0,  0,      1],
        ], dtype=np.float32)

        eye_h     = np.array([eye[0], eye[1], eye[2], 1.0], dtype=np.float32)
        cam_model = (M @ eye_h)[:3].astype(np.float32)

        MVP = (P @ V @ M).astype(np.float32)
        return MVP, M, cam_model
