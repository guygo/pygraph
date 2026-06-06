from OpenGL.GL import *
import numpy as np
from util import parse_to_numpy, default_expr, default_expr_3d
from plot import DynamicLinePlot, GridGeometry
from plot3d import SurfacePlot3D
from plot3d_parametric import ParametricSurfacePlot
import time
from shaders import *
from shaders3d import *
from OpenGL.GL.shaders import compileProgram, compileShader
from imgui_bundle import imgui


class AppState:
    def __init__(self):
        # ── geometry ─────────────────────────────────────────────────────
        self.plot_geo    = None
        self.grid_geo    = None
        self.surface_geo = None
        self.param_geo   = None

        # ── shaders ──────────────────────────────────────────────────────
        self.plot_solid_shader  = None
        self.plot_effect_shader = None
        self.grid_shader        = None
        self.surface_shader     = None
        self.wire_shader        = None

        # ── 2-D viewport (data-space) ────────────────────────────────────
        self.min_x, self.max_x = -10.0, 10.0
        self.min_y, self.max_y = -10.0, 10.0

        # ── last plot-rect in pixels (set by main every frame) ────────────
        # Used for correct resize, pan, and zoom — NOT the full display size.
        self._last_plot_x = 0.0
        self._last_plot_y = 0.0
        self._last_plot_w = 1280.0
        self._last_plot_h = 720.0

        # ── 2-D appearance ───────────────────────────────────────────────
        self.graph_color     = [0.0, 0.5, 0.8]
        self.line_thickness  = 4.0
        self.resolution      = 20000
        self.last_resolution = 20000
        self.effect_mode     = 0
        self.start_time      = time.time()

        self.show_axis_grid = True
        self.show_numbers   = True
        self.connect_lines  = True

        # ── 3-D state ────────────────────────────────────────────────────
        self.mode_3d        = False
        self.mode_parametric = False   # sub-mode of 3D: parametric vs z=f(x,y)

        # Parametric surface expressions
        self.param_expr_x    = 'u'
        self.param_expr_y    = 'f_u * cos(v)'
        self.param_expr_z    = 'f_u * sin(v)'
        self.param_f_expr    = '(sqrt(pi*u) + sin(pi*u)) / pi'   # f(u) for SoR
        self.param_u_range   = (0.0, 5.0)
        self.param_v_range   = (0.0, 6.2832)
        self.param_res_u     = 80
        self.param_res_v     = 80
        self.math_data_needs_update_param = False

        # Input buffers for parametric
        self.input_buf_px = 'u'
        self.input_buf_py = 'f_u * cos(v)'
        self.input_buf_pz = 'f_u * sin(v)'
        self.input_buf_pf = '(sqrt(pi*u) + sin(pi*u)) / pi'

        # 3D overlay options
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

        # ── sampled x range (for resample detection) ─────────────────────
        self._sampled_x_start = -1e9
        self._sampled_x_end   =  1e9
        self._sampled_zoom_w  = 20.0   # visible width at last sample

        # ── misc ─────────────────────────────────────────────────────────
        self.math_data_needs_update   = False
        self.math_data_needs_update3d = False
        self.initialized  = False
        self.large_font   = None

        self.current_expr    = default_expr
        self.current_expr_3d = default_expr_3d

        # ── equation input buffers (persistent across frames) ────────────
        self.input_buf_2d = default_expr
        self.input_buf_3d = default_expr_3d

    # ================================================================
    #  OpenGL init
    # ================================================================
    def init_gl(self):
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

        self.plot_geo    = DynamicLinePlot()
        self.grid_geo    = GridGeometry()
        self.surface_geo = SurfacePlot3D()
        self.param_geo   = ParametricSurfacePlot()

        self.generate_math_data()   # uses viewport-derived range
        self.generate_math_data_3d(self.current_expr_3d)
        self.generate_parametric_data()
        self.initialized = True

    # ================================================================
    #  Data generation
    # ================================================================
    # Overdraw: buffer this many visible-widths of data off each edge
    # so panning never reveals an end.
    OVERDRAW        = 2.0
    # Minimum samples across the visible range (when zoomed way out).
    MIN_VISIBLE_PTS = 2000
    # Maximum total sample count (caps memory / GPU upload time).
    MAX_TOTAL_PTS   = 400_000
    # Samples per pixel of visible width (drives quality when zoomed in).
    SAMPLES_PER_PX  = 3

    def _compute_sample_count(self):
        """
        How many points to use for the FULL sampled range?

        Strategy:
          - We want SAMPLES_PER_PX samples for every pixel of the VISIBLE
            plot width.  That keeps sin(x) resolved even when zoomed in.
          - The full range is (1 + 2*OVERDRAW) × visible width, so the
            total count scales proportionally.
          - Clamp to [MIN_VISIBLE_PTS*(1+2*OD), MAX_TOTAL_PTS] so we never
            under-sample (zoom-out aliasing) or over-allocate (zoom-in).
        """
        plot_px   = max(self._last_plot_w, 1.0)
        visible_pts = int(plot_px * self.SAMPLES_PER_PX)
        # Scale up for the overdraw region
        total_factor = 1.0 + 2.0 * self.OVERDRAW
        total_pts = int(visible_pts * total_factor)
        min_pts   = int(self.MIN_VISIBLE_PTS * total_factor)
        return max(min_pts, min(total_pts, self.MAX_TOTAL_PTS))

    def _sample_range(self):
        """Visible range expanded by OVERDRAW on each side."""
        half = (self.max_x - self.min_x) * self.OVERDRAW
        return self.min_x - half, self.max_x + half

    def generate_math_data(self, expr=None):
        if expr is not None:
            self.current_expr = expr

        x_start, x_end = self._sample_range()
        n = self._compute_sample_count()

        # Remember sampled extent so needs_resample() can check it
        self._sampled_x_start  = x_start
        self._sampled_x_end    = x_end
        self._sampled_zoom_w   = self.max_x - self.min_x  # zoom level at sample time

        x = np.linspace(x_start, x_end, n, dtype=np.float32)
        try:
            y = parse_to_numpy(self.current_expr)(x)
            y = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
        except Exception as e:
            print(f"Eval error: {e}")
            y = np.zeros_like(x)
        self.plot_geo.update_data(x, y)
        self.math_data_needs_update = False

    def needs_resample(self):
        """
        Resample when:
          (a) viewport has panned so we're within 50% of the sampled edge, OR
          (b) zoom level changed by more than 20% (zoomed in or out)
              — so sample density stays correct after zoom.
        """
        margin   = (self.max_x - self.min_x) * 0.5
        panned   = (self.min_x - margin < self._sampled_x_start or
                    self.max_x + margin > self._sampled_x_end)
        cur_w    = self.max_x - self.min_x
        zoomed   = abs(cur_w - self._sampled_zoom_w) / max(self._sampled_zoom_w, 1e-9) > 0.20
        return panned or zoomed

    def generate_math_data_3d(self, expr):
        self.current_expr_3d = expr
        n  = self.resolution_3d
        x  = np.linspace(self.range3d_x[0], self.range3d_x[1], n, dtype=np.float32)
        y  = np.linspace(self.range3d_y[0], self.range3d_y[1], n, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        try:
            fn = parse_to_numpy(expr, variables_str="x y")
            Z  = fn(X, Y).astype(np.float32)
        except Exception as e:
            print(f"3D parse error: {e}")
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
        import sympy as sp
        from util import parse_to_numpy
        nu = self.param_res_u
        nv = self.param_res_v
        u_vals = np.linspace(self.param_u_range[0], self.param_u_range[1], nu, dtype=np.float32)
        v_vals = np.linspace(self.param_v_range[0], self.param_v_range[1], nv, dtype=np.float32)
        U, V = np.meshgrid(u_vals, v_vals)  # (nv, nu)

        try:
            # Evaluate f(u) first if referenced
            f_u_expr = self.param_f_expr.strip()
            f_u_fn = parse_to_numpy(f_u_expr, variables_str="u")
            F_U = f_u_fn(U).astype(np.float32)  # (nv, nu)

            # Replace f_u in xyz expressions by substituting numerically
            # We evaluate each expression with u, v, f_u as available symbols
            def eval_expr(expr_str):
                # Replace f_u token with a placeholder, then lambdify with u,v
                # Strategy: sympify with local dict won't work for f_u
                # Instead: do string substitution of f_u -> a third variable w
                s = expr_str.replace("f_u", "_fw")
                u_sym, v_sym, w_sym = sp.symbols("u v _fw")
                expr = sp.sympify(s)
                fn = sp.lambdify((u_sym, v_sym, w_sym), expr, modules="numpy")
                return fn(U, V, F_U)

            X = eval_expr(self.param_expr_x).astype(np.float32)
            Y = eval_expr(self.param_expr_y).astype(np.float32)
            Z = eval_expr(self.param_expr_z).astype(np.float32)
        except Exception as e:
            print(f"Parametric eval error: {e}")
            X = U; Y = np.zeros_like(U); Z = np.zeros_like(U)

        self.param_geo.update_data(X, Y, Z)
        self.math_data_needs_update_param = False

    # ================================================================
    #  Bounding box for current surface (used by overlay3d)
    # ================================================================
    def get_scene_bounds(self):
        """Return (x_range, y_range, z_range) as (min,max) tuples for 3D overlay."""
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
        range_x = self.max_x - self.min_x
        exponent = np.floor(np.log10(max(abs(range_x), 1e-10)))
        base  = 10 ** exponent
        ratio = range_x / base
        if   ratio < 2.0: spacing_x = base * 0.2
        elif ratio < 5.0: spacing_x = base * 0.5
        else:             spacing_x = base * 1.0
        return spacing_x, spacing_x

    # ================================================================
    #  Resize – called every frame with the current plot rect
    # ================================================================
    def handle_resize(self, px, py, pw, ph):
        """
        Rescale viewport so that the centre stays fixed when the
        PLOT area (not the full window) changes size.
        """
        ow = self._last_plot_w
        oh = self._last_plot_h

        if ow > 0 and oh > 0 and (abs(pw - ow) > 0.5 or abs(ph - oh) > 0.5):
            cx = (self.min_x + self.max_x) * 0.5
            cy = (self.min_y + self.max_y) * 0.5
            rx = (self.max_x - self.min_x) * (pw / ow)
            ry = (self.max_y - self.min_y) * (ph / oh)
            self.min_x = cx - rx * 0.5;  self.max_x = cx + rx * 0.5
            self.min_y = cy - ry * 0.5;  self.max_y = cy + ry * 0.5

        self._last_plot_x = px
        self._last_plot_y = py
        self._last_plot_w = pw
        self._last_plot_h = ph

    # ================================================================
    #  Input processing – pan/zoom use the plot rect, not full screen
    # ================================================================
    def process_interactions(self, plot_rect):
        io = imgui.get_io()
        mx = io.mouse_pos.x
        my = io.mouse_pos.y
        px = plot_rect["x"];  py = plot_rect["y"]
        pw = plot_rect["w"];  ph = plot_rect["h"]

        # Only interact when cursor is inside the plot area
        mouse_in_plot = (px <= mx <= px + pw) and (py <= my <= py + ph)

        if io.want_capture_mouse and not mouse_in_plot:
            return

        if self.mode_3d:
            self._process_3d_interactions(io, mouse_in_plot)
        else:
            self._process_2d_interactions(io, mouse_in_plot, pw, ph)

    def _process_2d_interactions(self, io, mouse_in_plot, pw, ph):
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
            cx = (self.min_x + self.max_x) * 0.5
            cy = (self.min_y + self.max_y) * 0.5
            rx = max((self.max_x - self.min_x) * zf, 1e-6)
            ry = max((self.max_y - self.min_y) * zf, 1e-6)
            self.min_x = cx - rx*0.5;  self.max_x = cx + rx*0.5
            self.min_y = cy - ry*0.5;  self.max_y = cy + ry*0.5

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
        r  = self.cam_dist
        st = np.sin(self.cam_theta); ct = np.cos(self.cam_theta)
        sp = np.sin(self.cam_phi);   cp = np.cos(self.cam_phi)
        eye    = np.array([r*st*cp, r*st*sp, r*ct], dtype=np.float64)
        target = np.zeros(3, dtype=np.float64)
        up     = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        f = target - eye;  f /= np.linalg.norm(f)
        rv = np.cross(f, up); rv /= np.linalg.norm(rv)
        uv = np.cross(rv, f)

        V = np.eye(4, dtype=np.float32)
        V[0,:3] =  rv;  V[0,3] = -np.dot(rv, eye)
        V[1,:3] =  uv;  V[1,3] = -np.dot(uv, eye)
        V[2,:3] = -f;   V[2,3] =  np.dot(f,  eye)

        fov_r  = np.radians(self.cam_fov)
        aspect = width / max(height, 1)
        n, fa  = 0.01, 100.0
        fv     = 1.0 / np.tan(fov_r * 0.5)
        P = np.zeros((4,4), dtype=np.float32)
        P[0,0] = fv/aspect; P[1,1] = fv
        P[2,2] = (fa+n)/(n-fa); P[2,3] = 2*fa*n/(n-fa); P[3,2] = -1.0

        xr, yr, zr = self.get_scene_bounds()
        cx = (xr[0]+xr[1])*0.5
        cy = (yr[0]+yr[1])*0.5
        cz = (zr[0]+zr[1])*0.5
        rx = max(xr[1]-xr[0], 1e-9)
        ry = max(yr[1]-yr[0], 1e-9)
        rz = max(zr[1]-zr[0], 1e-9)
        s  = 2.0 / max(rx, ry, rz)   # uniform scale so shape is not distorted
        sx, sy, sz = 2.0/rx, 2.0/ry, 2.0/rz
        M = np.array([
            [sx,  0,  0, -cx*sx],
            [ 0, sy,  0, -cy*sy],
            [ 0,  0, sz, -cz*sz],
            [ 0,  0,  0,      1],
        ], dtype=np.float32)

        MVP = (P @ V @ M).astype(np.float32)
        return MVP, M