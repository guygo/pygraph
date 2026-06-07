from imgui_bundle import imgui
from util import format_label
import numpy as np
import math


def screen_to_data(state, sx, sy):
    """Convert screen pixel → data-space coordinates."""
    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]
    rx = state.max_x - state.min_x
    ry = state.max_y - state.min_y
    dx = state.min_x + rx * (sx - px) / pw
    dy = state.min_y + ry * (1.0 - (sy - py) / ph)
    return dx, dy

# ── plot viewport rect ───────────────────────────────────────────────────────
# Updated every frame by panel.py before rendering.
plot_rect = {"x": 0.0, "y": 0.0, "w": 1280.0, "h": 720.0}


def data_to_screen(state, dx, dy):
    """Convert data-space (dx, dy) → screen pixel (sx, sy) inside plot_rect."""
    rx = state.max_x - state.min_x
    ry = state.max_y - state.min_y
    px = plot_rect["x"]
    py = plot_rect["y"]
    pw = plot_rect["w"]
    ph = plot_rect["h"]
    sx = px + pw * (dx - state.min_x) / rx
    sy = py + ph * (1.0 - (dy - state.min_y) / ry)
    return sx, sy


def draw_axis_labels(state):
    """Draw axis tick labels using imgui background draw list."""
    text_color = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
    spacing_x, spacing_y = state.calculate_grid_spacing()
    draw_list = imgui.get_background_draw_list()

    px = plot_rect["x"]
    py = plot_rect["y"]
    pw = plot_rect["w"]
    ph = plot_rect["h"]

    use_large = state.large_font is not None
    if use_large:
        imgui.push_font(state.large_font)

    range_x = state.max_x - state.min_x

    if state.min_x <= 0.0 <= state.max_x:
        screen_x_origin = px + pw * ((0.0 - state.min_x) / range_x)
    else:
        screen_x_origin = px + 10.0

    # ── Y-axis tick labels ───────────────────────────────────────────────────
    def _y_label_x(lbl):
        """Place label to the LEFT of the y-axis line, clamped inside the plot."""
        tw = imgui.calc_text_size(lbl).x
        return max(screen_x_origin - tw - 4, px + 2)

    if state.log_scale_y and state.min_y > 0:
        lo_y = max(state.min_y, 1e-10)
        hi_y = max(state.max_y, lo_y * 1.001)
        for e in range(int(math.floor(math.log10(lo_y))),
                       int(math.ceil(math.log10(hi_y))) + 1):
            for m in [1, 2, 5]:
                v = m * (10.0 ** e)
                if lo_y <= v <= hi_y:
                    frac = (np.log(v) - np.log(lo_y)) / (np.log(hi_y) - np.log(lo_y))
                    sy = py + ph * (1.0 - frac)
                    lbl = f"1e{e}" if m == 1 else format_label(v)
                    draw_list.add_text(imgui.ImVec2(_y_label_x(lbl), sy - 10),
                                       text_color, lbl)
    else:
        val = np.floor(state.min_y / spacing_y) * spacing_y
        while val <= state.max_y + spacing_y:
            if abs(val) > spacing_y * 0.01:   # skip y≈0; origin label handles it
                sx, sy = data_to_screen(state, 0.0, val)
                if py - 10 < sy < py + ph + 10:
                    lbl = format_label(val)
                    draw_list.add_text(imgui.ImVec2(_y_label_x(lbl), sy - 10),
                                       text_color, lbl)
            val += spacing_y

    if state.min_y <= 0.0 <= state.max_y:
        _, screen_y_origin = data_to_screen(state, 0.0, 0.0)
    else:
        screen_y_origin = py + ph - 20.0

    # ── X-axis tick labels ───────────────────────────────────────────────────
    if state.log_scale_x and state.min_x > 0:
        lo_x = max(state.min_x, 1e-10)
        hi_x = max(state.max_x, lo_x * 1.001)
        for e in range(int(math.floor(math.log10(lo_x))),
                       int(math.ceil(math.log10(hi_x))) + 1):
            for m in [1, 2, 5]:
                v = m * (10.0 ** e)
                if lo_x <= v <= hi_x:
                    frac = (np.log(v) - np.log(lo_x)) / (np.log(hi_x) - np.log(lo_x))
                    sx = px + frac * pw
                    lbl = f"1e{e}" if m == 1 else format_label(v)
                    draw_list.add_text(imgui.ImVec2(sx - 10, screen_y_origin + 4),
                                       text_color, lbl)
    else:
        val = np.floor(state.min_x / spacing_x) * spacing_x
        while val <= state.max_x + spacing_x:
            if abs(val) > spacing_x * 0.01:   # skip x≈0; origin label handles it
                sx, _ = data_to_screen(state, val, 0.0)
                if px - 10 < sx < px + pw + 10:
                    lbl = format_label(val)
                    tw = imgui.calc_text_size(lbl).x
                    # Centre label on tick; clamp so it never clips at either edge
                    tx = sx - tw / 2
                    tx = max(tx, px + 2)
                    tx = min(tx, px + pw - tw - 2)
                    draw_list.add_text(imgui.ImVec2(tx, screen_y_origin + 4),
                                       text_color, lbl)
            val += spacing_x

    if (state.min_x <= 0.0 <= state.max_x and state.min_y <= 0.0 <= state.max_y
            and not state.log_scale_x and not state.log_scale_y):
        sx, sy = data_to_screen(state, 0.0, 0.0)
        draw_list.add_text(imgui.ImVec2(sx + 4, sy + 4), text_color, "0")

    if use_large:
        imgui.pop_font()


def draw_axis_lines(state):
    """Draw X=0 and Y=0 axis lines clipped to the plot rect."""
    draw_list  = imgui.get_background_draw_list()
    axis_color = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.9))
    thickness  = 1.5

    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    if state.min_y <= 0.0 <= state.max_y:
        _, sy = data_to_screen(state, 0.0, 0.0)
        draw_list.add_line(imgui.ImVec2(px, sy),
                           imgui.ImVec2(px + pw, sy),
                           axis_color, thickness)

    if state.min_x <= 0.0 <= state.max_x:
        sx, _ = data_to_screen(state, 0.0, 0.0)
        draw_list.add_line(imgui.ImVec2(sx, py),
                           imgui.ImVec2(sx, py + ph),
                           axis_color, thickness)


def draw_grid_lines(state):
    """Draw dynamic grid lines clipped to the plot rect."""
    draw_list  = imgui.get_background_draw_list()
    grid_color = imgui.get_color_u32(imgui.ImVec4(0.75, 0.75, 0.75, 0.5))
    spacing_x, spacing_y = state.calculate_grid_spacing()

    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    val = np.floor(state.min_x / spacing_x) * spacing_x
    while val <= state.max_x + spacing_x:
        sx, _ = data_to_screen(state, val, 0.0)
        draw_list.add_line(imgui.ImVec2(sx, py), imgui.ImVec2(sx, py + ph),
                           grid_color, 1.0)
        val += spacing_x

    val = np.floor(state.min_y / spacing_y) * spacing_y
    while val <= state.max_y + spacing_y:
        _, sy = data_to_screen(state, 0.0, val)
        draw_list.add_line(imgui.ImVec2(px, sy), imgui.ImVec2(px + pw, sy),
                           grid_color, 1.0)
        val += spacing_y


def draw_pinned_points(state):
    """Draw double-click pin markers with coordinate labels."""
    if not state.pinned_points:
        return
    draw_list  = imgui.get_background_draw_list()
    dot_col    = imgui.get_color_u32(imgui.ImVec4(0.85, 0.15, 0.15, 1.0))
    text_col   = imgui.get_color_u32(imgui.ImVec4(0.05, 0.05, 0.05, 1.0))
    bg_col     = imgui.get_color_u32(imgui.ImVec4(1.0,  1.0,  1.0,  0.85))

    for px_val, py_val, label in state.pinned_points:
        sx, sy = data_to_screen(state, px_val, py_val)
        draw_list.add_circle_filled(imgui.ImVec2(sx, sy), 5.0, dot_col)
        draw_list.add_circle(imgui.ImVec2(sx, sy), 5.0,
                             imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 1.0)), 0, 1.5)
        tw = imgui.calc_text_size(label).x
        th = imgui.calc_text_size(label).y
        tx, ty = sx + 8, sy - th - 2
        draw_list.add_rect_filled(
            imgui.ImVec2(tx - 2, ty - 1), imgui.ImVec2(tx + tw + 2, ty + th + 1),
            bg_col, 2.0)
        draw_list.add_text(imgui.ImVec2(tx, ty), text_col, label)


def draw_zoom_box(state):
    """Draw the Shift+drag selection rectangle while zooming."""
    if not state._zoom_box_active:
        return
    io = imgui.get_io()
    sx0, sy0 = state._zoom_box_start
    sx1, sy1 = io.mouse_pos.x, io.mouse_pos.y
    draw_list = imgui.get_background_draw_list()
    fill  = imgui.get_color_u32(imgui.ImVec4(0.20, 0.45, 0.80, 0.15))
    border = imgui.get_color_u32(imgui.ImVec4(0.20, 0.45, 0.80, 0.90))
    draw_list.add_rect_filled(imgui.ImVec2(sx0, sy0), imgui.ImVec2(sx1, sy1), fill)
    draw_list.add_rect(imgui.ImVec2(sx0, sy0), imgui.ImVec2(sx1, sy1), border, 0.0, 0, 1.5)


def draw_cursor_readout(state):
    """Show x/y data coordinates at the top-right corner of the plot when hovering."""
    io = imgui.get_io()
    mx, my = io.mouse_pos.x, io.mouse_pos.y
    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    if not (px <= mx <= px + pw and py <= my <= py + ph):
        return

    dx, dy = screen_to_data(state, mx, my)
    label = f"x = {format_label(dx)}    y = {format_label(dy)}"

    draw_list = imgui.get_background_draw_list()
    tw = imgui.calc_text_size(label).x
    th = imgui.calc_text_size(label).y
    pad = 6.0
    bg_x0 = px + pw - tw - pad * 2 - 4
    bg_y0 = py + 4
    # Subtle frosted background
    draw_list.add_rect_filled(
        imgui.ImVec2(bg_x0, bg_y0),
        imgui.ImVec2(bg_x0 + tw + pad * 2, bg_y0 + th + pad),
        imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.75)),
        3.0,
    )
    draw_list.add_text(
        imgui.ImVec2(bg_x0 + pad, bg_y0 + pad / 2),
        imgui.get_color_u32(imgui.ImVec4(0.1, 0.1, 0.1, 1.0)),
        label,
    )
