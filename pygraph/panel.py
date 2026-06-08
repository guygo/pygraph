import numpy as np
from imgui_bundle import imgui
from .plot_slot import (PLOT_EQUATION, PLOT_SCATTER, PLOT_LINE_DATA,
                       PLOT_HISTOGRAM, PLOT_KDE, PLOT_HEATMAP2D, PLOT_VIOLIN,
                       PLOT_TYPE_NAMES)

_INPUT_DARK = imgui.ImVec4(0.90, 0.90, 0.90, 1.0)   # match the app's light-gray input theme
_INPUT_TEXT = imgui.ImVec4(0.05, 0.05, 0.05, 1.0)  # dark text on light background
_ERR_COLOR  = imgui.ImVec4(0.9, 0.2, 0.2, 1.0)

_COLORMAP_NAMES = [
    "Viridis", "Hot", "Cool", "Grayscale",
    "HSV Height", "HSV Angle", "Plasma", "Inferno", "Turbo",
]

_CAM_THETA_DEFAULT = 0.6
_CAM_PHI_DEFAULT   = 0.8
_CAM_DIST_DEFAULT  = 3.5


def _tip(text):
    """Show a tooltip for the most recently rendered item."""
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_short):
        imgui.set_tooltip(text)


def _input_text(label, value, max_size=256):
    imgui.push_style_color(imgui.Col_.frame_bg,         _INPUT_DARK)
    imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.12, 0.18, 0.38, 1.0))
    imgui.push_style_color(imgui.Col_.frame_bg_active,  imgui.ImVec4(0.16, 0.24, 0.48, 1.0))
    imgui.push_style_color(imgui.Col_.text,             _INPUT_TEXT)
    changed, new_val = imgui.input_text(label, value, max_size)
    imgui.pop_style_color(4)
    return changed, new_val


def _show_error(err_msg):
    if err_msg:
        imgui.text_colored(_ERR_COLOR, err_msg[:80] + ("..." if len(err_msg) > 80 else ""))


def _colormap_combo(state, suffix=""):
    _, state.colormap = imgui.combo(
        f"Colormap##{suffix}", state.colormap, _COLORMAP_NAMES)


def _camera_section(state, io, suffix=""):
    imgui.text(f"theta={np.degrees(state.cam_theta):.1f}  "
               f"phi={np.degrees(state.cam_phi):.1f}")
    imgui.text(f"dist={state.cam_dist:.2f}  FPS {io.framerate:.0f}")
    if imgui.button(f"Reset Camera##{suffix}", imgui.ImVec2(-1, 0)):
        state.cam_theta = _CAM_THETA_DEFAULT
        state.cam_phi   = _CAM_PHI_DEFAULT
        state.cam_dist  = _CAM_DIST_DEFAULT


# ================================================================
#  Plot list (shared for equation and data slots)
# ================================================================
def _draw_plot_list(state):
    imgui.text("Plots")
    i = 0
    while i < len(state.plots):
        slot      = state.plots[i]
        is_active = (i == state.active_plot_idx)

        # Delete first so selectable can't block it
        if len(state.plots) > 1:
            if imgui.small_button(f"x##{i}"):
                state.remove_plot(i)
                state.math_data_needs_update = True
                return
            imgui.same_line()

        _, slot.visible = imgui.checkbox(f"##vis{i}", slot.visible)
        imgui.same_line()

        r, g, b = slot.color
        imgui.color_button(
            f"##col{i}", imgui.ImVec4(r, g, b, 1.0),
            imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14))
        imgui.same_line()

        result = imgui.selectable(f"{slot.display_label()}##sel{i}", is_active)
        clicked = result[0] if isinstance(result, tuple) else result
        if clicked:
            state.active_plot_idx = i

        i += 1

    if len(state.plots) < 8:
        if imgui.button("+ Equation", imgui.ImVec2(-1, 0)):
            state.add_plot()
        _tip("Add another equation curve to the plot (up to 8)")
    imgui.separator()


# ================================================================
#  Data slot controls
# ================================================================
def _draw_data_controls(state, io, slot):
    imgui.text_colored(imgui.ImVec4(0.1, 0.5, 0.85, 1.0),
                       f"Data: {slot.source_file or '(unknown)'}")
    
    # Data-type constants are 1-based (SCATTER=1…VIOLIN=6); combo is 0-based.
    display_idx = max(0, slot.plot_type - 1)
    ch_t, new_t = imgui.combo("Type##dt", display_idx,
                              ["Scatter", "Line", "Histogram", "KDE", "Heatmap", "Violin"])
    if ch_t:
        slot.plot_type = new_t + 1   # 0→SCATTER=1, …, 4→HEATMAP2D=5, 5→VIOLIN=6
        state._update_data_slot(slot, fit_view=False)

    col_names = slot.col_names or [f"col{i}" for i in range(
        slot.raw_data.shape[1] if slot.raw_data is not None else 1)]

    if slot.plot_type in (PLOT_SCATTER, PLOT_LINE_DATA, PLOT_HEATMAP2D):
        imgui.push_item_width(-1)
        ch_x, new_x = imgui.combo("X##cx", slot.col_x, col_names)
        if ch_x:
            slot.col_x = new_x
            state._update_data_slot(slot, fit_view=False)
        ch_y, new_y = imgui.combo("Y##cy", slot.col_y, col_names)
        if ch_y:
            slot.col_y = new_y
            state._update_data_slot(slot, fit_view=False)
        imgui.pop_item_width()

        if slot.plot_type == PLOT_SCATTER:
            imgui.push_item_width(-1)
            _, slot.point_size = imgui.slider_float(
                "##pts", slot.point_size, 2.0, 20.0,
                format=f"Point size  {slot.point_size:.0f}")
            imgui.pop_item_width()
        else:
            _, slot.connect_lines = imgui.checkbox("Connect Lines##dl", slot.connect_lines)

    else:  # HISTOGRAM, KDE, or VIOLIN
        imgui.push_item_width(-1)
        ch_c, new_c = imgui.combo("Column##dhc", slot.col_hist, col_names)
        if ch_c:
            slot.col_hist = new_c
            state._update_data_slot(slot, fit_view=False)
        imgui.pop_item_width()

        if slot.plot_type == PLOT_HISTOGRAM:
            ch_b, new_b = imgui.input_int("Bins##db", slot.hist_bins, 5, 20)
            if ch_b:
                slot.hist_bins = max(2, min(500, new_b))
                state._update_data_slot(slot, fit_view=False)
            # Ensure alpha is in valid range (line_thickness defaults to 4.0 for equations)
            slot.line_thickness = max(0.1, min(1.0, slot.line_thickness))

    imgui.separator()
    _, slot.color = imgui.color_edit3(f"Color##dc_{state.active_plot_idx}", slot.color)

    if slot.plot_type == PLOT_HISTOGRAM:
        imgui.push_item_width(-1)
        cur_alpha = max(0.1, min(1.0, slot.line_thickness))
        ch_a, new_a = imgui.slider_float(
            "##baro", cur_alpha, 0.1, 1.0,
            format=f"Fill alpha  {cur_alpha:.2f}")
        if ch_a:
            slot.line_thickness = new_a
        imgui.pop_item_width()

    n = slot.raw_data.shape[0] if slot.raw_data is not None else 0
    imgui.text_disabled(f"{n:,} rows  |  {len(col_names)} cols")

    # ── Column statistics ────────────────────────────────────────────────────
    if slot.raw_data is not None and imgui.collapsing_header("Stats##colstats"):
        col_idx = (slot.col_hist if slot.plot_type in
                   (PLOT_HISTOGRAM, PLOT_KDE, PLOT_VIOLIN) else slot.col_x)
        col_idx = min(col_idx, slot.raw_data.shape[1] - 1)
        col_data = slot.raw_data[:, col_idx]
        finite   = col_data[np.isfinite(col_data)]
        if len(finite):
            q25, q50, q75 = np.percentile(finite, [25, 50, 75])
            rows = [
                ("Min",    f"{float(np.min(finite)):.6g}"),
                ("Max",    f"{float(np.max(finite)):.6g}"),
                ("Mean",   f"{float(np.mean(finite)):.6g}"),
                ("Std",    f"{float(np.std(finite)):.6g}"),
                ("Median", f"{float(q50):.6g}"),
                ("Q1/Q3",  f"{float(q25):.4g} / {float(q75):.4g}"),
            ]
            for lbl, val in rows:
                imgui.text_disabled(f"{lbl:7s}")
                imgui.same_line(60)
                imgui.text(val)

    imgui.separator()
    _, state.show_axis_grid = imgui.checkbox("Axis Lines", state.show_axis_grid)
    imgui.same_line()
    _, state.show_numbers = imgui.checkbox("Grid", state.show_numbers)

    imgui.text(f"X [{state.min_x:.3g} : {state.max_x:.3g}]")
    imgui.text(f"Y [{state.min_y:.3g} : {state.max_y:.3g}]")
    if imgui.button("Fit View##dfit", imgui.ImVec2(-1, 0)):
        state._update_data_slot(slot)


# ================================================================
#  Equation 2-D controls
# ================================================================
def _draw_2d_controls(state, io):
    _draw_plot_list(state)

    if not state.plots:
        return

    slot = state.plots[state.active_plot_idx]

    # Route to data controls if this is a data slot
    if slot.plot_type != PLOT_EQUATION:
        _draw_data_controls(state, io, slot)
        return

    _show_error(slot.last_error)

    imgui.text("f(x) =")
    imgui.push_item_width(-1)
    changed_eq, new_eq = _input_text(f"##expr2d_{state.active_plot_idx}", slot.input_buf, 256)
    imgui.pop_item_width()
    if changed_eq:
        slot.input_buf = new_eq
    if imgui.is_item_deactivated_after_edit():
        expr = slot.input_buf.strip()
        if expr and expr != slot.expr:
            slot._cached_fn = None
            state.generate_math_data(expr)
            if not slot.last_error:
                state.add_to_history(expr)

    if imgui.small_button("?##help2d"):
        imgui.open_popup("##expr_help")
    _tip("Show supported functions and syntax")
    imgui.same_line()
    if imgui.begin_popup("##expr_help"):
        imgui.text_colored(imgui.ImVec4(0.2, 0.45, 0.8, 1.0), "Supported functions")
        imgui.separator()
        _HELP = [
            ("Trig",        "sin cos tan asin acos atan atan2(y,x)"),
            ("Hyp",         "sinh cosh tanh"),
            ("Exp / Log",   "exp log log10 log2 sqrt"),
            ("Rounding",    "abs floor ceil sign"),
            ("Constants",   "pi E"),
            ("Animation",   "'t' — enable Animation checkbox first"),
            ("Examples",    "sin(x)/x   |   x**2 + 2*x   |   exp(-x**2)"),
        ]
        for cat, txt in _HELP:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), f"{cat}:")
            imgui.same_line(70)
            imgui.text(txt)
        imgui.end_popup()

    if imgui.button("Plot##2d", imgui.ImVec2(-1, 0)):
        expr = slot.input_buf.strip()
        if expr:
            slot._cached_fn = None
            state.generate_math_data(expr)
            if not slot.last_error:
                state.add_to_history(expr)
    _tip("Evaluate and draw the expression (also triggered by Enter)")

    if state.equation_history and imgui.collapsing_header("Recent##hist"):
        for h_expr in state.equation_history[:12]:
            lbl = (h_expr[:38] + "...") if len(h_expr) > 38 else h_expr
            if imgui.selectable(f"{lbl}##h", False):
                slot.input_buf = h_expr
                slot._cached_fn = None
                state.generate_math_data(h_expr)

    imgui.separator()

    _, state.anim_enabled = imgui.checkbox("Animation (use 't')", state.anim_enabled)
    _tip("Enable time variable 't' in the expression, e.g. sin(x - t)")
    if state.anim_enabled:
        imgui.push_item_width(-1)
        ch_t, new_t = imgui.slider_float(
            "##anim_t", state.anim_time, 0.0, 20.0,
            format=f"t = {state.anim_time:.2f}")
        imgui.pop_item_width()
        if ch_t:
            state.anim_time = new_t
            state.math_data_needs_update = True

        play_lbl = "Pause##ap" if state.anim_playing else "Play##ap"
        if imgui.button(play_lbl, imgui.ImVec2(-85, 0)):
            state.anim_playing = not state.anim_playing
        imgui.same_line()
        if imgui.button("Reset##ar", imgui.ImVec2(-1, 0)):
            state.anim_time = 0.0
            state.anim_playing = False
            state.math_data_needs_update = True
        _, state.anim_speed = imgui.slider_float("Speed##as", state.anim_speed, 0.1, 10.0)

    imgui.separator()
    imgui.text("Active curve style:")
    imgui.push_item_width(-1)
    ch_lbl, new_lbl = _input_text(f"##slotlabel_{state.active_plot_idx}", slot.label, 64)
    imgui.pop_item_width()
    _tip("Custom label shown in the plot list (leave empty to auto-generate)")
    if ch_lbl:
        slot.label = new_lbl
    _, slot.color = imgui.color_edit3(f"Color##sc_{state.active_plot_idx}", slot.color)
    _, slot.effect_mode = imgui.combo(
        f"Effect##em_{state.active_plot_idx}", slot.effect_mode,
        ["Solid Color", "Neon Glow", "Plasma", "Electric"])
    _tip("Visual effect applied to this curve")
    imgui.push_item_width(-1)
    _, slot.line_thickness = imgui.slider_float(
        f"##thick_{state.active_plot_idx}", slot.line_thickness, 1.0, 15.0,
        format=f"Thickness  {slot.line_thickness:.1f}")
    imgui.pop_item_width()
    _tip("Line width in pixels")
    _, slot.connect_lines = imgui.checkbox(
        f"Connect Lines##{state.active_plot_idx}", slot.connect_lines)
    _tip("Draw a continuous line; uncheck to show individual sample dots")

    imgui.separator()
    _, state.show_axis_grid = imgui.checkbox("Axis Lines", state.show_axis_grid)
    _tip("Show the X=0 and Y=0 axis lines")
    imgui.same_line()
    _, state.show_numbers   = imgui.checkbox("Grid", state.show_numbers)
    _tip("Show background grid lines and tick labels")
    imgui.same_line()
    ch_dm, state.dark_mode = imgui.checkbox("Dark", state.dark_mode)
    _tip("Toggle dark / light UI theme")
    if ch_dm:
        state._dark_mode_dirty = True

    ch_lx, state.log_scale_x = imgui.checkbox("Log X", state.log_scale_x)
    _tip("Switch X axis to logarithmic scale")
    imgui.same_line()
    ch_ly, state.log_scale_y = imgui.checkbox("Log Y", state.log_scale_y)
    _tip("Switch Y axis to logarithmic scale")
    if ch_lx or ch_ly:
        state.math_data_needs_update = True

    changed_res, new_res = imgui.input_int("Resolution##2d", state.resolution, 1000, 10000)
    _tip("Number of sample points computed along X\nHigher = smoother curves, slower updates")
    if changed_res:
        state.resolution = max(100, min(200_000, new_res))
        state.math_data_needs_update = True

    imgui.separator()
    imgui.text("Viewport:")
    imgui.text(f"  X [{state.min_x:.2f} : {state.max_x:.2f}]")
    imgui.text(f"  Y [{state.min_y:.2f} : {state.max_y:.2f}]")
    zoom = 20.0 / max(state.max_x - state.min_x, 1e-9)
    imgui.text(f"  Zoom {zoom:.2f}x  FPS {io.framerate:.0f}")
    if imgui.button("Fit View##fit", imgui.ImVec2(-1, 0)):
        state.zoom_to_fit_2d()
    _tip("Auto-scale viewport to show the full curve (R)")


# ================================================================
#  3-D controls
# ================================================================
def _draw_3d_controls(state, io):
    _show_error(state.last_error)

    imgui.text("f(x, y) =")
    imgui.push_item_width(-1)
    changed_eq3, new_eq3 = _input_text("##expr3d", state.input_buf_3d, 256)
    imgui.pop_item_width()
    if changed_eq3:
        state.input_buf_3d = new_eq3
    if imgui.is_item_deactivated_after_edit():
        expr = state.input_buf_3d.strip()
        if expr and expr != state.current_expr_3d:
            state.generate_math_data_3d(expr)

    if imgui.button("Plot##3d", imgui.ImVec2(-1, 0)):
        expr = state.input_buf_3d.strip()
        if expr:
            state.generate_math_data_3d(expr)

    imgui.separator()
    changed_res, new_res3 = imgui.input_int("Grid Res##3d", state.resolution_3d, 10, 50)
    if changed_res:
        state.resolution_3d = max(10, min(300, new_res3))
        state.math_data_needs_update3d = True

    _colormap_combo(state, "3d")
    imgui.push_item_width(-1)
    _, state.surface_alpha = imgui.slider_float(
        "##alpha3d", state.surface_alpha, 0.1, 1.0,
        format=f"Opacity  {state.surface_alpha:.2f}")
    imgui.pop_item_width()

    _, state.show_wireframe = imgui.checkbox("Wireframe", state.show_wireframe)
    if state.show_wireframe:
        imgui.push_item_width(-1)
        _, state.wire_alpha = imgui.slider_float(
            "##wirealpha", state.wire_alpha, 0.0, 1.0,
            format=f"Wire alpha  {state.wire_alpha:.2f}")
        imgui.pop_item_width()

    imgui.separator()
    changed_cm, state.circular_mask = imgui.checkbox("Circular Domain", state.circular_mask)
    if changed_cm:
        state.math_data_needs_update3d = True
    if state.circular_mask:
        imgui.push_item_width(-1)
        changed_mr, new_mr = imgui.input_float(
            "##maxr", state.circular_max_r, 0.5, 2.0,
            format=f"Max r  {state.circular_max_r:.2f}")
        imgui.pop_item_width()
        if changed_mr:
            state.circular_max_r = max(0.1, new_mr)
            state.math_data_needs_update3d = True

    imgui.separator()
    imgui.text("X range:")
    imgui.push_item_width(-1)
    cx_r, rx_vals = imgui.input_float2("##xrange3d", list(state.range3d_x))
    imgui.pop_item_width()
    if cx_r and rx_vals[0] < rx_vals[1]:
        state.range3d_x = (rx_vals[0], rx_vals[1])
        state.math_data_needs_update3d = True

    imgui.text("Y range:")
    imgui.push_item_width(-1)
    cy_r, ry_vals = imgui.input_float2("##yrange3d", list(state.range3d_y))
    imgui.pop_item_width()
    if cy_r and ry_vals[0] < ry_vals[1]:
        state.range3d_y = (ry_vals[0], ry_vals[1])
        state.math_data_needs_update3d = True

    imgui.separator()
    _, state.show_3d_box    = imgui.checkbox("Bounding Box",  state.show_3d_box)
    _, state.show_3d_grid   = imgui.checkbox("Floor Grid",    state.show_3d_grid)
    _, state.show_3d_labels = imgui.checkbox("Axis Labels",   state.show_3d_labels)
    imgui.separator()
    _camera_section(state, io, "3d")


# ================================================================
#  Space curve controls
# ================================================================
def _draw_space_curve_controls(state, io):
    imgui.text_colored(imgui.ImVec4(0.4, 0.75, 1.0, 1.0), "Space Curve  (x(t), y(t), z(t))")
    imgui.separator()
    _show_error(state.last_error)

    for axis, buf_attr, expr_attr in [
        ("x", "input_buf_cx", "curve_expr_x"),
        ("y", "input_buf_cy", "curve_expr_y"),
        ("z", "input_buf_cz", "curve_expr_z"),
    ]:
        imgui.text(f"{axis}(t) =")
        imgui.push_item_width(-1)
        ch, new_v = _input_text(f"##c{axis}", getattr(state, buf_attr), 256)
        imgui.pop_item_width()
        if ch:
            setattr(state, buf_attr, new_v)
        if imgui.is_item_deactivated_after_edit():
            setattr(state, expr_attr, getattr(state, buf_attr).strip())
            state.math_data_needs_update_curve = True

    if imgui.button("Plot##curve", imgui.ImVec2(-1, 0)):
        state.curve_expr_x = state.input_buf_cx.strip()
        state.curve_expr_y = state.input_buf_cy.strip()
        state.curve_expr_z = state.input_buf_cz.strip()
        state.math_data_needs_update_curve = True

    imgui.separator()
    imgui.text("t range:")
    imgui.push_item_width(-1)
    ct, tvals = imgui.input_float2("##trange_c", list(state.curve_t_range))
    imgui.pop_item_width()
    if ct and tvals[0] < tvals[1]:
        state.curve_t_range = (tvals[0], tvals[1])
        state.math_data_needs_update_curve = True

    changed_n, new_n = imgui.input_int("Resolution##c", state.curve_resolution, 50, 200)
    if changed_n:
        state.curve_resolution = max(20, min(2000, new_n))
        state.math_data_needs_update_curve = True

    imgui.separator()
    imgui.push_item_width(-1)
    ch_r, new_r = imgui.slider_float(
        "##tuberadius", state.curve_tube_radius, 0.005, 0.15,
        format=f"Tube Radius  {state.curve_tube_radius:.3f}")
    imgui.pop_item_width()
    if ch_r:
        state.curve_tube_radius = max(0.001, new_r)
        state.math_data_needs_update_curve = True

    _colormap_combo(state, "c")
    imgui.push_item_width(-1)
    _, state.surface_alpha = imgui.slider_float(
        "##alpha_c", state.surface_alpha, 0.1, 1.0,
        format=f"Opacity  {state.surface_alpha:.2f}")
    imgui.pop_item_width()

    imgui.separator()
    _camera_section(state, io, "c")


# ================================================================
#  Parametric controls
# ================================================================
def _draw_parametric_controls(state, io):
    imgui.text_colored(imgui.ImVec4(0.2, 0.6, 0.3, 1.0), "Surface of Revolution")
    imgui.separator()
    _show_error(state.last_error)

    for label, buf_attr, expr_attr in [
        ("f(u) =",    "input_buf_pf", "param_f_expr"),
        ("x(u,v) =",  "input_buf_px", "param_expr_x"),
        ("y(u,v) =",  "input_buf_py", "param_expr_y"),
        ("z(u,v) =",  "input_buf_pz", "param_expr_z"),
    ]:
        imgui.text(label)
        imgui.push_item_width(-1)
        ch, new_v = _input_text(f"##p_{buf_attr}", getattr(state, buf_attr), 256)
        imgui.pop_item_width()
        if ch:
            setattr(state, buf_attr, new_v)
        if imgui.is_item_deactivated_after_edit():
            setattr(state, expr_attr, getattr(state, buf_attr).strip())
            state.math_data_needs_update_param = True

    if imgui.button("Plot##param", imgui.ImVec2(-1, 0)):
        state.param_f_expr = state.input_buf_pf.strip()
        state.param_expr_x = state.input_buf_px.strip()
        state.param_expr_y = state.input_buf_py.strip()
        state.param_expr_z = state.input_buf_pz.strip()
        state.math_data_needs_update_param = True

    imgui.separator()
    imgui.text("u range:")
    imgui.push_item_width(-1)
    cu, uvals = imgui.input_float2("##urange", list(state.param_u_range))
    imgui.pop_item_width()
    if cu and uvals[0] < uvals[1]:
        state.param_u_range = (uvals[0], uvals[1])
        state.math_data_needs_update_param = True

    imgui.text("v range:")
    imgui.push_item_width(-1)
    cv, vvals = imgui.input_float2("##vrange", list(state.param_v_range))
    imgui.pop_item_width()
    if cv and vvals[0] < vvals[1]:
        state.param_v_range = (vvals[0], vvals[1])
        state.math_data_needs_update_param = True

    imgui.separator()
    changed_res_u, new_ru = imgui.input_int("Res U##p", state.param_res_u, 10, 50)
    if changed_res_u:
        state.param_res_u = max(10, min(200, new_ru))
        state.math_data_needs_update_param = True
    changed_res_v, new_rv = imgui.input_int("Res V##p", state.param_res_v, 10, 50)
    if changed_res_v:
        state.param_res_v = max(10, min(200, new_rv))
        state.math_data_needs_update_param = True

    imgui.separator()
    _colormap_combo(state, "p")
    imgui.push_item_width(-1)
    _, state.surface_alpha = imgui.slider_float(
        "##alpha_p", state.surface_alpha, 0.1, 1.0,
        format=f"Opacity  {state.surface_alpha:.2f}")
    imgui.pop_item_width()
    _, state.show_wireframe = imgui.checkbox("Wireframe##p", state.show_wireframe)
    if state.show_wireframe:
        imgui.push_item_width(-1)
        _, state.wire_alpha = imgui.slider_float(
            "##wirealpha_p", state.wire_alpha, 0.0, 1.0,
            format=f"Wire alpha  {state.wire_alpha:.2f}")
        imgui.pop_item_width()

    imgui.separator()
    _camera_section(state, io, "param")


# ================================================================
#  Main panel entry point
# ================================================================
def draw_panel(state, plot_rect):
    io  = imgui.get_io()
    W   = io.display_size.x
    H   = io.display_size.y
    mh  = state._menubar_h          # menu bar height offset

    panel_w = state.panel_width if state.panel_open else 24
    px = float(panel_w)
    pw = max(W - panel_w, 1.0)
    ph = max(H - mh, 1.0)

    plot_rect["x"] = px
    plot_rect["y"] = mh
    plot_rect["w"] = pw
    plot_rect["h"] = ph

    state.handle_resize(px, mh, pw, ph)

    imgui.set_next_window_pos(imgui.ImVec2(0, mh))
    imgui.set_next_window_size(imgui.ImVec2(panel_w, H - mh))
    flags = (imgui.WindowFlags_.no_resize   | imgui.WindowFlags_.no_move    |
             imgui.WindowFlags_.no_collapse  | imgui.WindowFlags_.no_title_bar |
             imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##panel", None, flags)

    if not state.panel_open:
        if imgui.arrow_button("##expand", imgui.Dir.right):
            state.panel_open = True
    else:
        if imgui.arrow_button("##collapse", imgui.Dir.left):
            state.panel_open = False
        imgui.same_line()
        imgui.text_colored(imgui.ImVec4(0.2, 0.4, 0.8, 1.0), "OpenGL Plotter")
        imgui.text("Drag: Pan  |  Scroll: Zoom  |  Shift+Drag: Zoom Box")

        # ── Global error banner ──────────────────────────────────────────────
        if state.last_error:
            imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0.95, 0.20, 0.20, 0.18))
            imgui.begin_child("##errbanner", imgui.ImVec2(-1, 0), False,
                              imgui.WindowFlags_.no_scrollbar)
            imgui.text_colored(imgui.ImVec4(0.85, 0.10, 0.10, 1.0), "⚠ " + state.last_error[:72])
            imgui.same_line()
            if imgui.small_button("✕##errdismiss"):
                state.last_error = ""
            imgui.end_child()

        if imgui.small_button("Open File..."):
            state._open_file_requested = True
        _tip("Load a CSV / TSV data file")
        imgui.same_line()
        if imgui.small_button("Save PNG"):
            state._save_screenshot_requested = True
        _tip("Export the current plot as a PNG image (Ctrl+S)")
        imgui.separator()

        _is_surf  = state.mode_3d and not state.mode_parametric and not state.mode_space_curve
        _is_param = state.mode_3d and state.mode_parametric
        _is_curve = state.mode_3d and state.mode_space_curve

        if imgui.radio_button("2D##mode", not state.mode_3d):
            state.mode_3d = False; state.mode_parametric = False; state.mode_space_curve = False
        imgui.same_line()
        if imgui.radio_button("3D f(x,y)##mode", _is_surf):
            state.mode_3d = True; state.mode_parametric = False; state.mode_space_curve = False
        if imgui.radio_button("Parametric##mode", _is_param):
            state.mode_3d = True; state.mode_parametric = True; state.mode_space_curve = False
        imgui.same_line()
        if imgui.radio_button("Curve##mode", _is_curve):
            state.mode_3d = True; state.mode_parametric = False; state.mode_space_curve = True
        imgui.separator()

        if not state.mode_3d:
            _draw_2d_controls(state, io)
        elif _is_surf:
            _draw_3d_controls(state, io)
        elif _is_param:
            _draw_parametric_controls(state, io)
        else:
            _draw_space_curve_controls(state, io)

    imgui.end()

    # Resize handle
    if state.panel_open:
        handle_x = state.panel_width - 3
        imgui.set_next_window_pos(imgui.ImVec2(handle_x, mh))
        imgui.set_next_window_size(imgui.ImVec2(6, H - mh))
        imgui.set_next_window_bg_alpha(0.0)
        rflags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
                  imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_move   |
                  imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_nav)
        imgui.begin("##resize_handle", None, rflags)
        imgui.invisible_button("##drag", imgui.ImVec2(6, H - mh))
        if imgui.is_item_active():
            delta = imgui.get_io().mouse_delta.x
            state.panel_width = max(200, min(520, int(state.panel_width + delta)))
        if imgui.is_item_hovered() or imgui.is_item_active():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        imgui.end()
