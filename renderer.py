import time
import numpy as np
from OpenGL.GL import *
from imgui_bundle import imgui
from gui import (draw_axis_labels, draw_grid_lines, draw_axis_lines,
                 draw_cursor_readout, draw_zoom_box, draw_pinned_points)
from overlay3d import draw_3d_overlay
from plot_slot import PLOT_EQUATION, PLOT_SCATTER, PLOT_LINE_DATA, PLOT_HISTOGRAM, PLOT_KDE, PLOT_HEATMAP2D, PLOT_VIOLIN

_EFFECT_WIDTH_MULT = {1: 2.8, 2: 2.2, 3: 2.8}


def _upload_mat4(program, name, mat):
    loc = glGetUniformLocation(program, name)
    glUniformMatrix4fv(loc, 1, GL_FALSE, mat.T.flatten())


def _draw_surface(state, MVP, M, cam_model, light_dir, geo, ambient):
    prog = state.surface_shader
    glUseProgram(prog)
    _upload_mat4(prog, "uMVP",   MVP)
    _upload_mat4(prog, "uModel", M)
    glUniform3fv(glGetUniformLocation(prog, "uLightDir"), 1, light_dir)
    glUniform3fv(glGetUniformLocation(prog, "uCamPos"),   1, cam_model)
    glUniform1f(glGetUniformLocation(prog,  "uAmbient"),  ambient)
    glUniform1i(glGetUniformLocation(prog,  "uColormap"), state.colormap)
    glUniform1f(glGetUniformLocation(prog,  "uAlpha"),    state.surface_alpha)

    if state.surface_alpha < 1.0:
        glDepthMask(GL_FALSE)
    geo.draw()
    glDepthMask(GL_TRUE)

    if state.show_wireframe:
        prog_w = state.wire_shader
        glUseProgram(prog_w)
        _upload_mat4(prog_w, "uMVP", MVP)
        glUniform4f(glGetUniformLocation(prog_w, "uWireColor"),
                    0.18, 0.28, 0.55, state.wire_alpha)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        geo.draw()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_POLYGON_OFFSET_LINE)


def _set_log_uniforms(prog, state):
    glUniform1i(glGetUniformLocation(prog, "uLogX"), 1 if state.log_scale_x else 0)
    glUniform1i(glGetUniformLocation(prog, "uLogY"), 1 if state.log_scale_y else 0)


def _render_2d(state, fb_px, fb_py, fb_pw, fb_ph, fb_W, fb_H, plot_rect=None):
    glEnable(GL_SCISSOR_TEST)
    glScissor(fb_px, 0, fb_pw, fb_ph)

    if state.show_numbers:
        draw_grid_lines(state)
    if state.show_axis_grid:
        draw_axis_lines(state)

    glViewport(fb_px, fb_py, fb_pw, fb_ph)

    t_now = time.time() - state.start_time

    for slot in state.plots:
        if not slot.visible:
            continue

        pt = slot.plot_type

        # ── Histogram bars ────────────────────────────────────────────────────
        if pt == PLOT_HISTOGRAM:
            if slot.hist_geo is None:
                continue
            prog = state.plot_fill_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog, "uAlpha"),     max(0.3, min(1.0, slot.line_thickness)))
            slot.hist_geo.draw()
            continue

        # ── Scatter points ────────────────────────────────────────────────────
        if pt == PLOT_SCATTER:
            if slot.geometry is None:
                continue
            prog = state.plot_point_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog, "uPointSize"), slot.point_size)
            _set_log_uniforms(prog, state)
            slot.geometry.draw(GL_POINTS)
            continue

        # ── KDE: filled area + curve line ────────────────────────────────────
        if pt == PLOT_KDE:
            if slot.geometry is None:
                continue
            # Semi-transparent fill
            if slot.hist_geo is not None:
                prog_f = state.plot_fill_shader
                glUseProgram(prog_f)
                glUniform2f(glGetUniformLocation(prog_f, "uMinBounds"), state.min_x, state.min_y)
                glUniform2f(glGetUniformLocation(prog_f, "uMaxBounds"), state.max_x, state.max_y)
                glUniform3f(glGetUniformLocation(prog_f, "uColor"),     *slot.color)
                glUniform1f(glGetUniformLocation(prog_f, "uAlpha"),     0.35)
                slot.hist_geo.draw()
            # Solid curve on top
            prog = state.plot_solid_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog, "uLineWidth"), slot.line_thickness)
            glUniform2f(glGetUniformLocation(prog, "uViewport"),  float(fb_pw), float(fb_ph))
            _set_log_uniforms(prog, state)
            slot.geometry.draw(GL_LINE_STRIP)
            continue

        # ── Line from data ────────────────────────────────────────────────────
        if pt == PLOT_LINE_DATA:
            if slot.geometry is None:
                continue
            prog = state.plot_solid_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog, "uLineWidth"), slot.line_thickness)
            glUniform2f(glGetUniformLocation(prog, "uViewport"),  float(fb_pw), float(fb_ph))
            _set_log_uniforms(prog, state)
            slot.geometry.draw(GL_LINE_STRIP)
            continue

        # ── 2D Heatmap ────────────────────────────────────────────────────────
        if pt == PLOT_HEATMAP2D:
            if slot.heatmap_geo is None:
                continue
            prog = state.plot_colored_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform1f(glGetUniformLocation(prog, "uAlpha"),     1.0)
            slot.heatmap_geo.draw()
            continue

        # ── Violin ────────────────────────────────────────────────────────────
        if pt == PLOT_VIOLIN:
            if slot.hist_geo is None:
                continue
            # Filled shape
            prog_f = state.plot_fill_shader
            glUseProgram(prog_f)
            glUniform2f(glGetUniformLocation(prog_f, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog_f, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog_f, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog_f, "uAlpha"),     0.40)
            slot.hist_geo.draw()
            # Upper edge line
            if slot.geometry is not None:
                prog = state.plot_solid_shader
                glUseProgram(prog)
                glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
                glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
                glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
                glUniform1f(glGetUniformLocation(prog, "uLineWidth"), slot.line_thickness)
                glUniform2f(glGetUniformLocation(prog, "uViewport"),  float(fb_pw), float(fb_ph))
                _set_log_uniforms(prog, state)
                slot.geometry.draw(GL_LINE_STRIP)
            # Median/quartile lines (vertical lines at q25, q50, q75)
            if getattr(slot, '_violin_quartiles', None) and plot_rect is not None:
                q25, q50, q75 = slot._violin_quartiles
                mx = getattr(slot, '_violin_max_kde', 0.0)
                draw_list = imgui.get_background_draw_list()
                col_u32 = imgui.get_color_u32(imgui.ImVec4(*slot.color, 0.9))
                px2 = plot_rect["x"];  py2 = plot_rect["y"]
                pw2 = plot_rect["w"];  ph2 = plot_rect["h"]
                rx = state.max_x - state.min_x; ry = state.max_y - state.min_y
                def _sx(xd): return px2 + pw2 * (xd - state.min_x) / rx
                def _sy(yd): return py2 + ph2 * (1.0 - (yd - state.min_y) / ry)
                for xq in [q25, q50, q75]:
                    sx = _sx(xq)
                    thick = 3.0 if xq == q50 else 1.5
                    draw_list.add_line(imgui.ImVec2(sx, _sy(-mx * 0.85)),
                                       imgui.ImVec2(sx, _sy( mx * 0.85)),
                                       col_u32, thick)
            continue

        # ── Equation (solid or effect) ────────────────────────────────────────
        if slot.geometry is None or not slot.expr:
            continue
        draw_mode = GL_LINE_STRIP if slot.connect_lines else GL_POINTS
        if slot.effect_mode == 0:
            prog = state.plot_solid_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"), state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"), state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),     *slot.color)
            glUniform1f(glGetUniformLocation(prog, "uLineWidth"), slot.line_thickness)
            glUniform2f(glGetUniformLocation(prog, "uViewport"),  float(fb_pw), float(fb_ph))
            _set_log_uniforms(prog, state)
            slot.geometry.draw(draw_mode)
        else:
            mult = _EFFECT_WIDTH_MULT.get(slot.effect_mode, 2.0)
            prog = state.plot_effect_shader
            glUseProgram(prog)
            glUniform2f(glGetUniformLocation(prog, "uMinBounds"),  state.min_x, state.min_y)
            glUniform2f(glGetUniformLocation(prog, "uMaxBounds"),  state.max_x, state.max_y)
            glUniform3f(glGetUniformLocation(prog, "uColor"),      *slot.color)
            glUniform1i(glGetUniformLocation(prog, "uEffectMode"), slot.effect_mode)
            glUniform1f(glGetUniformLocation(prog, "uTime"),       t_now)
            glUniform1f(glGetUniformLocation(prog, "uLineWidth"),  slot.line_thickness * mult)
            glUniform2f(glGetUniformLocation(prog, "uViewport"),   float(fb_pw), float(fb_ph))
            _set_log_uniforms(prog, state)
            slot.geometry.draw(GL_LINE_STRIP)

    glViewport(0, 0, fb_W, fb_H)
    glDisable(GL_SCISSOR_TEST)

    if state.show_axis_grid:
        draw_axis_labels(state)
    draw_pinned_points(state)
    draw_zoom_box(state)
    draw_cursor_readout(state)


def _render_3d(state, fb_px, fb_py, fb_pw, fb_ph, fb_W, fb_H, px, py, pw, ph):
    glViewport(fb_px, fb_py, fb_pw, fb_ph)

    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glUseProgram(state.bg_shader)
    state.bg_geo.draw()
    glDepthMask(GL_TRUE)

    glEnable(GL_DEPTH_TEST)

    MVP, M, cam_model = state.get_mvp(pw, ph)
    light_dir = np.array([0.6, 0.8, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    if state.mode_space_curve:
        _draw_surface(state, MVP, M, cam_model, light_dir, state.curve_geo,   ambient=0.18)
    elif not state.mode_parametric:
        _draw_surface(state, MVP, M, cam_model, light_dir, state.surface_geo, ambient=0.20)
    else:
        _draw_surface(state, MVP, M, cam_model, light_dir, state.param_geo,   ambient=0.20)

    glViewport(0, 0, fb_W, fb_H)
    glDisable(GL_DEPTH_TEST)

    if state.show_3d_box or state.show_3d_grid or state.show_3d_labels:
        xr, yr, zr = state.get_scene_bounds()
        draw_3d_overlay(
            MVP, (px, py, pw, ph),
            xr, yr, zr,
            grid_color=(0.60, 0.68, 0.82, 0.45),
            axis_color=(0.15, 0.30, 0.68, 0.85),
            box_color =(0.60, 0.68, 0.82, 0.55),
            label_color=(0.08, 0.12, 0.28, 0.95),
            show_grid=state.show_3d_grid,
            show_box=state.show_3d_box,
            show_labels=state.show_3d_labels,
            large_font=state.large_font
        )

    # Camera info overlay (bottom-left of plot)
    draw_list = imgui.get_background_draw_list()
    th_deg = np.degrees(state.cam_theta)
    ph_deg = np.degrees(state.cam_phi) % 360
    info   = f"θ={th_deg:.0f}°  φ={ph_deg:.0f}°  dist={state.cam_dist:.2f}  ⊙dbl-click reset"
    draw_list.add_text(
        imgui.ImVec2(px + 6, py + ph - 20),
        imgui.get_color_u32(imgui.ImVec4(0.6, 0.6, 0.6, 0.9)),
        info,
    )


def render_frame(state, plot_rect):
    """Clear and render the current frame (2D or 3D)."""
    io    = imgui.get_io()
    W     = io.display_size.x
    H     = io.display_size.y
    fb_sx = io.display_framebuffer_scale.x
    fb_sy = io.display_framebuffer_scale.y

    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    fb_px = int(px * fb_sx)
    # OpenGL Y=0 is at the bottom; convert from imgui top-left to OpenGL bottom-left
    fb_py = int((H - py - ph) * fb_sy)
    fb_pw = int(pw * fb_sx);  fb_ph = int(ph * fb_sy)
    fb_W  = int(W  * fb_sx);  fb_H  = int(H  * fb_sy)

    glClearColor(1.0, 1.0, 1.0, 1.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLineWidth(1.0)

    if not state.mode_3d:
        _render_2d(state, fb_px, fb_py, fb_pw, fb_ph, fb_W, fb_H, plot_rect)
    else:
        _render_3d(state, fb_px, fb_py, fb_pw, fb_ph, fb_W, fb_H, px, py, pw, ph)
