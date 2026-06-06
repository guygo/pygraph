import numpy as np
import ctypes
from OpenGL.GL import *
from imgui_bundle import imgui, hello_imgui
import time
from shaders import *
from shaders3d import *
from OpenGL.GL.shaders import compileProgram, compileShader
from gui import draw_axis_labels, draw_grid_lines, draw_axis_lines, state, plot_rect
from overlay3d import draw_3d_overlay
from plot import DynamicLinePlot, GridGeometry


# ── helpers ────────────────────────────────────────────────────────────────

def _upload_mat4(program, name, mat):
    loc = glGetUniformLocation(program, name)
    glUniformMatrix4fv(loc, 1, GL_FALSE, mat.T.flatten())


PANEL_WIDTH = 260   # fixed UI panel width in pixels


# ── callbacks ──────────────────────────────────────────────────────────────

def on_init_callback():
    if not state.initialized:
        state.init_gl()
    fonts = imgui.get_io().fonts
    try:
        state.large_font = fonts.add_font_default(
            imgui.FontConfig(size_pixels=18.0)
        )
        hello_imgui.reload_font_texture()
    except Exception:
        state.large_font = None


def frame_update_callback():
    if not state.initialized:
        return

    io = imgui.get_io()
    W  = io.display_size.x
    H  = io.display_size.y

    # ── compute plot rect (right of the panel, full height) ───────────────
    px = float(PANEL_WIDTH)
    py = 0.0
    pw = max(W - PANEL_WIDTH, 1.0)
    ph = max(H, 1.0)

    # Update shared plot_rect dict used by gui.py draw functions
    plot_rect["x"] = px
    plot_rect["y"] = py
    plot_rect["w"] = pw
    plot_rect["h"] = ph

    # Resize correction — only on plot area changes
    state.handle_resize(px, py, pw, ph)

    # ── UI panel (fixed left side) ─────────────────────────────────────────
    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    imgui.set_next_window_size(imgui.ImVec2(PANEL_WIDTH, H))
    flags = (imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move |
             imgui.WindowFlags_.no_collapse |
             imgui.WindowFlags_.no_title_bar)
    imgui.begin("##panel", None, flags)

    imgui.text_colored(imgui.ImVec4(0.2, 0.4, 0.8, 1.0), "OpenGL Plotter")
    imgui.text("Drag: Pan  |  Scroll: Zoom")
    imgui.separator()

    # Mode toggle
    if imgui.radio_button("2D Plot##mode", not state.mode_3d):
        state.mode_3d = False
        state.mode_parametric = False
    imgui.same_line()
    if imgui.radio_button("3D z=f(x,y)##mode", state.mode_3d and not state.mode_parametric):
        state.mode_3d = True
        state.mode_parametric = False
    imgui.same_line()
    if imgui.radio_button("Parametric##mode", state.mode_3d and state.mode_parametric):
        state.mode_3d = True
        state.mode_parametric = True
    imgui.separator()

    if not state.mode_3d:
        # ── 2-D controls ───────────────────────────────────────────────
        _, state.show_axis_grid = imgui.checkbox("Axis Lines (X=0 / Y=0)", state.show_axis_grid)
        _, state.show_numbers   = imgui.checkbox("Dynamic Grid",             state.show_numbers)
        imgui.separator()

        changed, new_res = imgui.input_int("Resolution##2d", state.resolution, 1000, 10000)
        if changed:
            state.resolution = max(100, min(200_000, new_res))
            state.math_data_needs_update = True

        imgui.separator()
        imgui.text("Visual Style")
        _, state.effect_mode = imgui.combo(
            "Effect##mode", state.effect_mode,
            ["Solid Color", "Neon Rainbow", "Animated Stripes"]
        )
        if state.effect_mode == 0:
            _, state.graph_color = imgui.color_edit3("Color##plot", state.graph_color)

        _, state.connect_lines = imgui.checkbox("Connect Lines", state.connect_lines)

        imgui.push_item_width(-1)
        _, state.line_thickness = imgui.slider_float(
            "##thickness", state.line_thickness, 1.0, 15.0,
            format=f"Thickness  {state.line_thickness:.1f}"
        )
        imgui.pop_item_width()

        imgui.separator()
        imgui.text("Viewport:")
        imgui.text(f"  X [{state.min_x:.2f} : {state.max_x:.2f}]")
        imgui.text(f"  Y [{state.min_y:.2f} : {state.max_y:.2f}]")
        imgui.text(f"  FPS {io.framerate:.0f}")

        imgui.separator()
        imgui.text("f(x) =")
        imgui.push_item_width(-1)
        changed_eq, new_eq = imgui.input_text("##expr2d", state.input_buf_2d, 256)
        imgui.pop_item_width()
        if changed_eq:
            state.input_buf_2d = new_eq
        # Apply on Enter or when editing stops and expression changed
        if imgui.is_item_deactivated_after_edit():
            expr = state.input_buf_2d.strip()
            if expr and expr != state.current_expr:
                try:
                    state.generate_math_data(expr)
                except Exception as e:
                    print(f"2D parse error: {e}")

        if imgui.button("Plot##2d", imgui.ImVec2(-1, 0)):
            expr = state.input_buf_2d.strip()
            if expr:
                try:
                    state.generate_math_data(expr)
                except Exception as e:
                    print(f"2D parse error: {e}")

    elif state.mode_3d and not state.mode_parametric:
        # ── 3-D z=f(x,y) controls ─────────────────────────────────────
        imgui.text("f(x, y) =")
        imgui.push_item_width(-1)
        changed_eq3, new_eq3 = imgui.input_text("##expr3d", state.input_buf_3d, 256)
        imgui.pop_item_width()
        if changed_eq3:
            state.input_buf_3d = new_eq3
        if imgui.is_item_deactivated_after_edit():
            expr = state.input_buf_3d.strip()
            if expr and expr != state.current_expr_3d:
                try:
                    state.generate_math_data_3d(expr)
                except Exception as e:
                    print(f"3D parse error: {e}")

        if imgui.button("Plot##3d", imgui.ImVec2(-1, 0)):
            expr = state.input_buf_3d.strip()
            if expr:
                try:
                    state.generate_math_data_3d(expr)
                except Exception as e:
                    print(f"3D parse error: {e}")

        imgui.separator()
        changed_res, new_res3 = imgui.input_int("Grid Res##3d", state.resolution_3d, 10, 50)
        if changed_res:
            state.resolution_3d = max(10, min(300, new_res3))
            state.math_data_needs_update3d = True

        _, state.colormap = imgui.combo(
            "Colormap##3d", state.colormap,
            ["Viridis", "Hot", "Cool", "Grayscale", "HSV Height", "HSV Angle"]
        )
        imgui.push_item_width(-1)
        _, state.surface_alpha = imgui.slider_float(
            "##alpha3d", state.surface_alpha, 0.1, 1.0,
            format=f"Opacity  {state.surface_alpha:.2f}"
        )
        imgui.pop_item_width()

        _, state.show_wireframe = imgui.checkbox("Wireframe", state.show_wireframe)
        if state.show_wireframe:
            imgui.push_item_width(-1)
            _, state.wire_alpha = imgui.slider_float(
                "##wirealpha", state.wire_alpha, 0.0, 1.0,
                format=f"Wire alpha  {state.wire_alpha:.2f}"
            )
            imgui.pop_item_width()

        imgui.separator()
        changed_cm, state.circular_mask = imgui.checkbox("Circular Domain", state.circular_mask)
        if changed_cm:
            state.math_data_needs_update3d = True
        if state.circular_mask:
            imgui.push_item_width(-1)
            changed_mr, new_mr = imgui.input_float("##maxr", state.circular_max_r, 0.5, 2.0,
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
        if cx_r:
            state.range3d_x = (rx_vals[0], rx_vals[1])
            state.math_data_needs_update3d = True

        imgui.text("Y range:")
        imgui.push_item_width(-1)
        cy_r, ry_vals = imgui.input_float2("##yrange3d", list(state.range3d_y))
        imgui.pop_item_width()
        if cy_r:
            state.range3d_y = (ry_vals[0], ry_vals[1])
            state.math_data_needs_update3d = True

        imgui.separator()
        imgui.text(f"θ={np.degrees(state.cam_theta):.1f}°  "
                   f"φ={np.degrees(state.cam_phi):.1f}°")
        imgui.text(f"dist={state.cam_dist:.2f}  FPS {io.framerate:.0f}")
        imgui.separator()
        _, state.show_3d_box    = imgui.checkbox("Bounding Box",  state.show_3d_box)
        _, state.show_3d_grid   = imgui.checkbox("Floor Grid",    state.show_3d_grid)
        _, state.show_3d_labels = imgui.checkbox("Axis Labels",   state.show_3d_labels)

        if imgui.button("Reset Camera##3d", imgui.ImVec2(-1, 0)):
            state.cam_theta = 0.6; state.cam_phi = 0.8; state.cam_dist = 3.5

    elif state.mode_3d and state.mode_parametric:
        # ── Parametric surface controls ────────────────────────────────
        imgui.text_colored(imgui.ImVec4(0.2, 0.6, 0.3, 1.0), "Surface of Revolution")
        imgui.separator()

        imgui.text("f(u) =")
        imgui.push_item_width(-1)
        chf, newf = imgui.input_text("##pf", state.input_buf_pf, 256)
        imgui.pop_item_width()
        if chf: state.input_buf_pf = newf
        if imgui.is_item_deactivated_after_edit():
            state.param_f_expr = state.input_buf_pf.strip()
            state.math_data_needs_update_param = True

        imgui.separator()
        imgui.text("x(u,v) =")
        imgui.push_item_width(-1)
        chx, newx = imgui.input_text("##px", state.input_buf_px, 256)
        imgui.pop_item_width()
        if chx: state.input_buf_px = newx
        if imgui.is_item_deactivated_after_edit():
            state.param_expr_x = state.input_buf_px.strip()
            state.math_data_needs_update_param = True

        imgui.text("y(u,v) =")
        imgui.push_item_width(-1)
        chy, newy = imgui.input_text("##py", state.input_buf_py, 256)
        imgui.pop_item_width()
        if chy: state.input_buf_py = newy
        if imgui.is_item_deactivated_after_edit():
            state.param_expr_y = state.input_buf_py.strip()
            state.math_data_needs_update_param = True

        imgui.text("z(u,v) =")
        imgui.push_item_width(-1)
        chz, newz = imgui.input_text("##pz", state.input_buf_pz, 256)
        imgui.pop_item_width()
        if chz: state.input_buf_pz = newz
        if imgui.is_item_deactivated_after_edit():
            state.param_expr_z = state.input_buf_pz.strip()
            state.math_data_needs_update_param = True

        if imgui.button("Plot##param", imgui.ImVec2(-1, 0)):
            state.param_f_expr  = state.input_buf_pf.strip()
            state.param_expr_x  = state.input_buf_px.strip()
            state.param_expr_y  = state.input_buf_py.strip()
            state.param_expr_z  = state.input_buf_pz.strip()
            state.math_data_needs_update_param = True

        imgui.separator()
        imgui.text("u range:")
        imgui.push_item_width(-1)
        cu, uvals = imgui.input_float2("##urange", list(state.param_u_range))
        imgui.pop_item_width()
        if cu:
            state.param_u_range = (uvals[0], uvals[1])
            state.math_data_needs_update_param = True

        imgui.text("v range:")
        imgui.push_item_width(-1)
        cv, vvals = imgui.input_float2("##vrange", list(state.param_v_range))
        imgui.pop_item_width()
        if cv:
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
        _, state.colormap = imgui.combo(
            "Colormap##p", state.colormap,
            ["Viridis", "Hot", "Cool", "Grayscale", "HSV Height", "HSV Angle"]
        )
        imgui.push_item_width(-1)
        _, state.surface_alpha = imgui.slider_float(
            "##alpha_p", state.surface_alpha, 0.1, 1.0,
            format=f"Opacity  {state.surface_alpha:.2f}"
        )
        imgui.pop_item_width()
        _, state.show_wireframe = imgui.checkbox("Wireframe##p", state.show_wireframe)
        if state.show_wireframe:
            imgui.push_item_width(-1)
            _, state.wire_alpha = imgui.slider_float(
                "##wirealpha_p", state.wire_alpha, 0.0, 1.0,
                format=f"Wire alpha  {state.wire_alpha:.2f}"
            )
            imgui.pop_item_width()

        imgui.separator()
        imgui.text(f"θ={np.degrees(state.cam_theta):.1f}°  "
                   f"φ={np.degrees(state.cam_phi):.1f}°")
        imgui.text(f"dist={state.cam_dist:.2f}  FPS {io.framerate:.0f}")
        if imgui.button("Reset Camera##param", imgui.ImVec2(-1, 0)):
            state.cam_theta = 0.6; state.cam_phi = 0.8; state.cam_dist = 3.5

    imgui.end()   # ── end panel ──────────────────────────────────────────

    # ── deferred recompute ─────────────────────────────────────────────────
    state.process_interactions(plot_rect)

    if state.math_data_needs_update:
        state.generate_math_data()          # resample around new viewport
    elif not state.mode_3d and state.needs_resample():
        state.generate_math_data()          # viewport drifted — extend sample range

    if state.math_data_needs_update3d:
        state.generate_math_data_3d(state.current_expr_3d)

    if state.math_data_needs_update_param:
        state.generate_parametric_data()

    # ── OpenGL clear (whole window white) ──────────────────────────────────
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLineWidth(1.0)

    if not state.mode_3d:
        # ── 2-D ────────────────────────────────────────────────────────
        # Clip all GL drawing to the plot area
        glEnable(GL_SCISSOR_TEST)
        glScissor(int(px), 0, int(pw), int(ph))

        if state.show_numbers:
            draw_grid_lines()
        if state.show_axis_grid:
            draw_axis_lines()

        # Shader needs NDC relative to the PLOT viewport, not the full window.
        # We set glViewport to the plot rect so NDC [-1,1] maps to the plot area.
        glViewport(int(px), int(py), int(pw), int(ph))

        draw_mode = GL_LINE_STRIP if state.connect_lines else GL_POINTS
        if state.effect_mode == 0:
            prog = state.plot_solid_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"), *state.graph_color)
            glUniform1f(glGetUniformLocation(prog, "uLineWidth"), state.line_thickness)
            glUniform2f(glGetUniformLocation(prog, "uViewport"), pw, ph)
            state.plot_geo.draw(draw_mode)
        else:
            prog = state.plot_effect_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"), *state.graph_color)
            glUniform1i(glGetUniformLocation(prog, "uEffectMode"), state.effect_mode)
            glUniform1f(glGetUniformLocation(prog, "uTime"), time.time() - state.start_time)
            state.plot_geo.draw(draw_mode)

        # Restore full viewport for imgui
        glViewport(0, 0, int(W), int(H))
        glDisable(GL_SCISSOR_TEST)

        if state.show_axis_grid:
            draw_axis_labels()

    else:
        # ── 3-D ────────────────────────────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glViewport(int(px), int(py), int(pw), int(ph))

        MVP, M = state.get_mvp(pw, ph)
        light_dir = np.array([0.6, 0.8, 1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        prog = state.surface_shader
        glUseProgram(prog)
        _upload_mat4(prog, "uMVP",   MVP)
        _upload_mat4(prog, "uModel", M)
        glUniform3fv(glGetUniformLocation(prog, "uLightDir"), 1, light_dir)
        glUniform1f(glGetUniformLocation(prog,  "uAmbient"),  0.25)
        glUniform1i(glGetUniformLocation(prog,  "uColormap"), state.colormap)
        glUniform1f(glGetUniformLocation(prog,  "uAlpha"),    state.surface_alpha)

        if state.surface_alpha < 1.0:
            glDepthMask(GL_FALSE)
        state.surface_geo.draw()
        glDepthMask(GL_TRUE)

        if state.show_wireframe:
            prog_w = state.wire_shader
            glUseProgram(prog_w)
            _upload_mat4(prog_w, "uMVP", MVP)
            glUniform4f(glGetUniformLocation(prog_w, "uWireColor"),
                        0.0, 0.0, 0.0, state.wire_alpha)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            state.surface_geo.draw()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_POLYGON_OFFSET_LINE)

        # ── parametric surface ────────────────────────────────────────
        if state.mode_parametric:
            prog2 = state.surface_shader
            glUseProgram(prog2)
            _upload_mat4(prog2, "uMVP",   MVP)
            _upload_mat4(prog2, "uModel", M)
            glUniform3fv(glGetUniformLocation(prog2, "uLightDir"), 1, light_dir)
            glUniform1f(glGetUniformLocation(prog2,  "uAmbient"),  0.3)
            glUniform1i(glGetUniformLocation(prog2,  "uColormap"), state.colormap)
            glUniform1f(glGetUniformLocation(prog2,  "uAlpha"),    state.surface_alpha)
            if state.surface_alpha < 1.0:
                glDepthMask(GL_FALSE)
            state.param_geo.draw()
            glDepthMask(GL_TRUE)

            if state.show_wireframe:
                prog_w2 = state.wire_shader
                glUseProgram(prog_w2)
                _upload_mat4(prog_w2, "uMVP", MVP)
                glUniform4f(glGetUniformLocation(prog_w2, "uWireColor"),
                            0.0, 0.0, 0.0, state.wire_alpha)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glEnable(GL_POLYGON_OFFSET_LINE)
                glPolygonOffset(-1.0, -1.0)
                state.param_geo.draw()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDisable(GL_POLYGON_OFFSET_LINE)

        glViewport(0, 0, int(W), int(H))
        glDisable(GL_DEPTH_TEST)

        # ── 3D overlay (box, grid, labels) drawn in screen space ──────────
        if state.show_3d_box or state.show_3d_grid or state.show_3d_labels:
            xr, yr, zr = state.get_scene_bounds()
            draw_3d_overlay(
                MVP, (px, py, pw, ph),
                xr, yr, zr,
                show_grid=state.show_3d_grid,
                show_box=state.show_3d_box,
                show_labels=state.show_3d_labels,
                large_font=state.large_font
            )


# ── style ──────────────────────────────────────────────────────────────────

def on_post_init_style():
    s = imgui.get_style()
    white      = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
    light      = imgui.ImVec4(0.93, 0.93, 0.93, 1.0)
    mid        = imgui.ImVec4(0.80, 0.80, 0.80, 1.0)
    dark       = imgui.ImVec4(0.65, 0.65, 0.65, 1.0)
    darker     = imgui.ImVec4(0.50, 0.50, 0.50, 1.0)
    accent     = imgui.ImVec4(0.20, 0.45, 0.80, 1.0)
    accent_h   = imgui.ImVec4(0.15, 0.35, 0.70, 1.0)
    black_text = imgui.ImVec4(0.05, 0.05, 0.05, 1.0)

    s.set_color_(imgui.Col_.window_bg,          white)
    s.set_color_(imgui.Col_.child_bg,           white)
    s.set_color_(imgui.Col_.popup_bg,           white)
    s.set_color_(imgui.Col_.frame_bg,           light)
    s.set_color_(imgui.Col_.frame_bg_hovered,   mid)
    s.set_color_(imgui.Col_.frame_bg_active,    dark)
    s.set_color_(imgui.Col_.title_bg,           mid)
    s.set_color_(imgui.Col_.title_bg_active,    dark)
    s.set_color_(imgui.Col_.button,             mid)
    s.set_color_(imgui.Col_.button_hovered,     dark)
    s.set_color_(imgui.Col_.button_active,      darker)
    s.set_color_(imgui.Col_.slider_grab,        accent)
    s.set_color_(imgui.Col_.slider_grab_active, accent_h)
    s.set_color_(imgui.Col_.check_mark,         accent)
    s.set_color_(imgui.Col_.header,             mid)
    s.set_color_(imgui.Col_.header_hovered,     dark)
    s.set_color_(imgui.Col_.header_active,      darker)
    s.set_color_(imgui.Col_.separator,          dark)
    s.set_color_(imgui.Col_.text,               black_text)
    s.set_color_(imgui.Col_.text_disabled,      imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
    s.set_color_(imgui.Col_.border,             dark)
    s.window_rounding    = 4.0
    s.frame_rounding     = 3.0
    s.grab_rounding      = 3.0
    s.item_spacing       = imgui.ImVec2(6, 5)
    s.frame_padding      = imgui.ImVec2(5, 4)
    s.window_padding     = imgui.ImVec2(8, 8)


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Fast OpenGL Math Plotter (Pan & Zoom Demo)"
    params.app_window_params.window_geometry.size = (1280, 720)
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    params.callbacks.post_init         = on_init_callback
    params.callbacks.setup_imgui_style = on_post_init_style
    params.callbacks.custom_background = frame_update_callback
    hello_imgui.run(params)