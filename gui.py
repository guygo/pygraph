from imgui_bundle import imgui
from util import format_label
import numpy as np
import math

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
                    draw_list.add_text(imgui.ImVec2(screen_x_origin + 4, sy - 10),
                                       text_color, lbl)
    else:
        val = np.floor(state.min_y / spacing_y) * spacing_y
        while val <= state.max_y + spacing_y:
            sx, sy = data_to_screen(state, 0.0, val)
            if py - 10 < sy < py + ph + 10:
                draw_list.add_text(imgui.ImVec2(screen_x_origin + 4, sy - 10),
                                   text_color, format_label(val))
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
            if abs(val) > spacing_x * 0.01:
                sx, _ = data_to_screen(state, val, 0.0)
                if px - 10 < sx < px + pw + 10:
                    draw_list.add_text(imgui.ImVec2(sx - 10, screen_y_origin + 4),
                                       text_color, format_label(val))
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
