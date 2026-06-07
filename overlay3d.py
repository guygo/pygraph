"""
3-D scene overlays drawn with Dear ImGui draw-list calls:
  - Bounding box (12 edges)
  - Axis lines with tick marks and labels
  - Grid floor (XY plane at z_min)

All coordinates are projected via the same MVP + viewport as the surface.
"""
import numpy as np
from imgui_bundle import imgui
from util import format_label


def _project(pt, MVP, viewport):
    """Project 3-D world point -> screen (sx, sy) using MVP + viewport."""
    x, y, z = pt
    v = MVP @ np.array([x, y, z, 1.0], dtype=np.float32)
    if abs(v[3]) < 1e-9:
        return None
    ndc = v[:3] / v[3]
    px, py, pw, ph = viewport
    sx = px + (ndc[0] + 1.0) * 0.5 * pw
    sy = py + (1.0 - (ndc[1] + 1.0) * 0.5) * ph
    # Cull if outside depth range (behind near or far plane)
    if ndc[2] > 1.0 or ndc[2] < -1.0:
        return None
    return sx, sy


def draw_3d_overlay(MVP, viewport, x_range, y_range, z_range,
                    grid_color=(0.65, 0.65, 0.65, 0.5),
                    axis_color=(0.25, 0.25, 0.25, 1.0),
                    box_color=(0.5, 0.5, 0.5, 0.6),
                    label_color=(0.15, 0.15, 0.15, 1.0),
                    show_grid=True, show_box=True, show_labels=True,
                    large_font=None):
    """
    MVP      – 4×4 float32 array (column-major, already transposed for GL)
    viewport – (px, py, pw, ph) in screen pixels
    *_range  – (min, max) tuples for each axis
    """
    dl = imgui.get_background_draw_list()

    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range

    gc  = imgui.get_color_u32(imgui.ImVec4(*grid_color))
    ac  = imgui.get_color_u32(imgui.ImVec4(*axis_color))
    bc  = imgui.get_color_u32(imgui.ImVec4(*box_color))
    tc  = imgui.get_color_u32(imgui.ImVec4(*label_color))

    def proj(pt):
        return _project(pt, MVP, viewport)

    def line(a, b, col, thick=1.0):
        sa = proj(a); sb = proj(b)
        if sa and sb:
            dl.add_line(imgui.ImVec2(*sa), imgui.ImVec2(*sb), col, thick)

    def label(pt, text, offset=(4, -10)):
        s = proj(pt)
        if s:
            dl.add_text(imgui.ImVec2(s[0] + offset[0], s[1] + offset[1]), tc, text)

    # ── bounding box ────────────────────────────────────────────────────────
    if show_box:
        corners = [(x, y, z)
                   for x in (xmin, xmax)
                   for y in (ymin, ymax)
                   for z in (zmin, zmax)]
        edges = [
            (0,1),(2,3),(4,5),(6,7),   # z edges
            (0,2),(1,3),(4,6),(5,7),   # y edges
            (0,4),(1,5),(2,6),(3,7),   # x edges
        ]
        for i, j in edges:
            line(corners[i], corners[j], bc, 1.0)

    # ── grid floor at z = zmin ───────────────────────────────────────────────
    if show_grid:
        n_lines = 8
        for i in range(n_lines + 1):
            t = i / n_lines
            xv = xmin + t * (xmax - xmin)
            yv = ymin + t * (ymax - ymin)
            line((xv, ymin, zmin), (xv, ymax, zmin), gc, 1.0)
            line((xmin, yv, zmin), (xmax, yv, zmin), gc, 1.0)

    # ── axis lines ──────────────────────────────────────────────────────────
    # X axis (along y=ymin, z=zmin)
    line((xmin, ymin, zmin), (xmax, ymin, zmin), ac, 1.5)
    # Y axis (along x=xmin, z=zmin)
    line((xmin, ymin, zmin), (xmin, ymax, zmin), ac, 1.5)
    # Z axis (along x=xmin, y=ymin)
    line((xmin, ymin, zmin), (xmin, ymin, zmax), ac, 2.0)

    if show_labels:
        if large_font:
            imgui.push_font(large_font)

        # Z axis label
        label((xmin, ymin, zmax), "Z", offset=(-18, -6))

        # Z ticks
        n_z = 4
        for i in range(n_z + 1):
            t = i / n_z
            zv = zmin + t * (zmax - zmin)
            label((xmin, ymin, zv), format_label(zv), offset=(-38, -8))

        # X ticks along bottom
        n_x = 4
        for i in range(n_x + 1):
            t = i / n_x
            xv = xmin + t * (xmax - xmin)
            label((xv, ymin, zmin), format_label(xv), offset=(0, 4))

        # Y ticks along bottom
        n_y = 4
        for i in range(n_y + 1):
            t = i / n_y
            yv = ymin + t * (ymax - ymin)
            label((xmin, yv, zmin), format_label(yv), offset=(4, 4))

        if large_font:
            imgui.pop_font()
