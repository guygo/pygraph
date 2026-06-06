from OpenGL.GL import *
from imgui_bundle import imgui
from state import AppState
from util import format_label
from plot import DynamicLinePlot, GridGeometry
import numpy as np

# Global state (shared with main.py via import)
state = AppState()

# ── plot viewport rect ───────────────────────────────────────────────────────
# Updated every frame by main.py after the UI panel is drawn.
# All screen↔data conversions use this rect so the panel is never in the way.
plot_rect = {"x": 0.0, "y": 0.0, "w": 1280.0, "h": 720.0}


def data_to_screen(dx, dy):
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


def draw_axis_labels():
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
    range_y = state.max_y - state.min_y

    # Y-axis label X position (at x=0 if visible, else left edge of plot)
    if state.min_x <= 0.0 <= state.max_x:
        screen_x_origin = px + pw * ((0.0 - state.min_x) / range_x)
    else:
        screen_x_origin = px + 10.0

    # Y-axis tick labels
    val = np.floor(state.min_y / spacing_y) * spacing_y
    while val <= state.max_y + spacing_y:
        sx, sy = data_to_screen(0.0, val)
        if py - 10 < sy < py + ph + 10:
            draw_list.add_text(imgui.ImVec2(screen_x_origin + 4, sy - 10),
                               text_color, format_label(val))
        val += spacing_y

    # X-axis label Y position (at y=0 if visible, else bottom of plot)
    if state.min_y <= 0.0 <= state.max_y:
        _, screen_y_origin = data_to_screen(0.0, 0.0)
    else:
        screen_y_origin = py + ph - 20.0

    # X-axis tick labels
    val = np.floor(state.min_x / spacing_x) * spacing_x
    while val <= state.max_x + spacing_x:
        if abs(val) > spacing_x * 0.01:          # skip near-zero
            sx, _ = data_to_screen(val, 0.0)
            if px - 10 < sx < px + pw + 10:
                draw_list.add_text(imgui.ImVec2(sx - 10, screen_y_origin + 4),
                                   text_color, format_label(val))
        val += spacing_x

    # Origin "0"
    if state.min_x <= 0.0 <= state.max_x and state.min_y <= 0.0 <= state.max_y:
        sx, sy = data_to_screen(0.0, 0.0)
        draw_list.add_text(imgui.ImVec2(sx + 4, sy + 4), text_color, "0")

    if use_large:
        imgui.pop_font()


def draw_axis_lines():
    """Draw X=0 and Y=0 axis lines clipped to the plot rect."""
    draw_list  = imgui.get_background_draw_list()
    axis_color = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.9))
    thickness  = 1.5

    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    # Y=0 horizontal line
    if state.min_y <= 0.0 <= state.max_y:
        _, sy = data_to_screen(0.0, 0.0)
        draw_list.add_line(imgui.ImVec2(px, sy),
                           imgui.ImVec2(px + pw, sy),
                           axis_color, thickness)

    # X=0 vertical line
    if state.min_x <= 0.0 <= state.max_x:
        sx, _ = data_to_screen(0.0, 0.0)
        draw_list.add_line(imgui.ImVec2(sx, py),
                           imgui.ImVec2(sx, py + ph),
                           axis_color, thickness)


def draw_grid_lines():
    """Draw dynamic grid lines clipped to the plot rect."""
    draw_list  = imgui.get_background_draw_list()
    grid_color = imgui.get_color_u32(imgui.ImVec4(0.75, 0.75, 0.75, 0.5))
    spacing_x, spacing_y = state.calculate_grid_spacing()

    px = plot_rect["x"];  py = plot_rect["y"]
    pw = plot_rect["w"];  ph = plot_rect["h"]

    # Vertical lines
    val = np.floor(state.min_x / spacing_x) * spacing_x
    while val <= state.max_x + spacing_x:
        sx, _ = data_to_screen(val, 0.0)
        draw_list.add_line(imgui.ImVec2(sx, py), imgui.ImVec2(sx, py + ph),
                           grid_color, 1.0)
        val += spacing_x

    # Horizontal lines
    val = np.floor(state.min_y / spacing_y) * spacing_y
    while val <= state.max_y + spacing_y:
        _, sy = data_to_screen(0.0, val)
        draw_list.add_line(imgui.ImVec2(px, sy), imgui.ImVec2(px + pw, sy),
                           grid_color, 1.0)
        val += spacing_y
