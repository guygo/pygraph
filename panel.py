import numpy as np
from imgui_bundle import imgui
from plot_slot import (PLOT_EQUATION, PLOT_SCATTER, PLOT_LINE_DATA,
                       PLOT_HISTOGRAM, PLOT_KDE, PLOT_TYPE_NAMES)

_INPUT_DARK = imgui.ImVec4(0.08, 0.12, 0.28, 1.0)
_INPUT_TEXT = imgui.ImVec4(1.00, 1.00, 1.00, 1.0)
_ERR_COLOR  = imgui.ImVec4(0.9, 0.2, 0.2, 1.0)

_COLORMAP_NAMES = [
    "Viridis", "Hot", "Cool", "Grayscale",
    "HSV Height", "HSV Angle", "Plasma", "Inferno", "Turbo",
]

_CAM_THETA_DEFAULT = 0.6
_CAM_PHI_DEFAULT   = 0.8
_CAM_DIST_DEFAULT  = 3.5


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

        if slot.plot_type == PLOT_EQUATION:
            lbl = slot.expr[:22] + ("..." if len(slot.expr) > 22 else "") if slot.expr else "(empty)"
        else:
            type_tag = ["", "scat", "line", "hist", "kde"][slot.plot_type]
            lbl = f"{type_tag}:{slot.source_file[:18]}"

        result = imgui.selectable(f"{lbl}##sel{i}", is_active)
        clicked = result[0] if isinstance(result, tuple) else result
        if clicked:
            state.active_plot_idx = i

        i += 1

    if len(state.plots) < 8:
        if imgui.button("+ Equation", imgui.ImVec2(-1, 0)):
            state.add_plot()
    imgui.separator()


# ================================================================
#  Data slot controls
# ================================================================
def _draw_data_controls(state, io, slot):
    imgui.text_colored(imgui.ImVec4(0.1, 0.5, 0.85, 1.0),
                       f"Data: {slot.source_file or '(unknown)'}")
    
    # Data-type constants are 1-based (SCATTER=1…KDE=4); combo is 0-based.
    display_idx = max(0, slot.plot_type - 1)
    ch_t, new_t = imgui.combo("Type##dt", display_idx,
                              ["Scatter", "Line", "Histogram", "KDE"])
    if ch_t:
        slot.plot_type = new_t + 1   # 0→SCATTER=1, 1→LINE=2, 2→HIST=3, 3→KDE=4
        state._update_data_slot(slot)

    col_names = slot.col_names or [f"col{i}" for i in range(
        slot.raw_data.shape[1] if slot.raw_data is not None else 1)]

    if slot.plot_type in (PLOT_SCATTER, PLOT_LINE_DATA):
        imgui.push_item_width(-1)
        ch_x, new_x = imgui.combo("X##cx", slot.col_x, col_names)
        if ch_x:
            slot.col_x = new_x
            state._update_data_slot(slot)
        ch_y, new_y = imgui.combo("Y##cy", slot.col_y, col_names)
        if ch_y:
            slot.col_y = new_y
            state._update_data_slot(slot)
        imgui.pop_item_width()

        if slot.plot_type == PLOT_SCATTER:
            imgui.push_item_width(-1)
            _, slot.point_size = imgui.slider_float(
                "##pts", slot.point_size, 2.0, 20.0,
                format=f"Point size  {slot.point_size:.0f}")
            imgui.pop_item_width()
        else:
            _, slot.connect_lines = imgui.checkbox("Connect Lines##dl", slot.connect_lines)

    else:  # HISTOGRAM or KDE
        imgui.push_item_width(-1)
        ch_c, new_c = imgui.combo("Column##dhc", slot.col_hist, col_names)
        if ch_c:
            slot.col_hist = new_c
            state._update_data_slot(slot)
        imgui.pop_item_width()

        if slot.plot_type == PLOT_HISTOGRAM:
            ch_b, new_b = imgui.input_int("Bins##db", slot.hist_bins, 5, 20)
            if ch_b:
                slot.hist_bins = max(2, min(500, new_b))
                state._update_data_slot(slot)

    imgui.separator()
    _, slot.color = imgui.color_edit3(f"Color##dc_{state.active_plot_idx}", slot.color)

    if slot.plot_type == PLOT_HISTOGRAM:
        imgui.push_item_width(-1)
        _, slot.line_thickness = imgui.slider_float(
            "##baro", slot.line_thickness, 0.3, 1.0,
            format=f"Fill alpha  {slot.line_thickness:.2f}")
        imgui.pop_item_width()

    n = slot.raw_data.shape[0] if slot.raw_data is not None else 0
    imgui.text_disabled(f"{n:,} rows  |  {len(col_names)} cols")

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

    if imgui.button("Plot##2d", imgui.ImVec2(-1, 0)):
        expr = slot.input_buf.strip()
        if expr:
            slot._cached_fn = None
            state.generate_math_data(expr)
            if not slot.last_error:
                state.add_to_history(expr)

    if state.equation_history and imgui.collapsing_header("Recent##hist"):
        for h_expr in state.equation_history[:12]:
            lbl = (h_expr[:38] + "...") if len(h_expr) > 38 else h_expr
            if imgui.selectable(f"{lbl}##h", False):
                slot.input_buf = h_expr
                slot._cached_fn = None
                state.generate_math_data(h_expr)

    imgui.separator()

    _, state.anim_enabled = imgui.checkbox("Animation (use 't')", state.anim_enabled)
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
    _, slot.color = imgui.color_edit3(f"Color##sc_{state.active_plot_idx}", slot.color)
    _, slot.effect_mode = imgui.combo(
        f"Effect##em_{state.active_plot_idx}", slot.effect_mode,
        ["Solid Color", "Neon Glow", "Plasma", "Electric"])
    imgui.push_item_width(-1)
    _, slot.line_thickness = imgui.slider_float(
        f"##thick_{state.active_plot_idx}", slot.line_thickness, 1.0, 15.0,
        format=f"Thickness  {slot.line_thickness:.1f}")
    imgui.pop_item_width()
    _, slot.connect_lines = imgui.checkbox(
        f"Connect Lines##{state.active_plot_idx}", slot.connect_lines)

    imgui.separator()
    _, state.show_axis_grid = imgui.checkbox("Axis Lines", state.show_axis_grid)
    imgui.same_line()
    _, state.show_numbers   = imgui.checkbox("Grid", state.show_numbers)

    ch_lx, state.log_scale_x = imgui.checkbox("Log X", state.log_scale_x)
    imgui.same_line()
    ch_ly, state.log_scale_y = imgui.checkbox("Log Y", state.log_scale_y)
    if ch_lx or ch_ly:
        state.math_data_needs_update = True

    changed_res, new_res = imgui.input_int("Resolution##2d", state.resolution, 1000, 10000)
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
        imgui.text("Drag: Pan  |  Scroll: Zoom")
        if imgui.small_button("Open File..."):
            state._open_file_requested = True
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
