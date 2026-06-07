import traceback
import threading
from OpenGL.GL import *
from imgui_bundle import imgui, hello_imgui
from state import AppState
from gui import plot_rect
from panel import draw_panel
from renderer import render_frame
from session import load_session, save_session

state = AppState()


# ================================================================
#  Exception guard – prevents Python exceptions from reaching
#  nanobind callbacks (which can't translate them on Python 3.14)
# ================================================================
def _guarded(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
    return wrapper


# ================================================================
#  File dialog – runs in a background thread so the render loop
#  keeps running while the user picks a file.
# ================================================================
def _open_file_dialog():
    """Open a native file-picker dialog without blocking the render loop."""
    def _worker():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Open Data File",
                filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All files", "*")],
            )
            root.destroy()
            if path:
                state._pending_file = path
        except Exception as e:
            state.last_error = f"File dialog error: {e}"

    threading.Thread(target=_worker, daemon=True).start()


# ================================================================
#  Menus  (hello_imgui native – called when show_menu_bar = True)
# ================================================================
def _show_menus():
    if imgui.begin_menu("File"):
        r = imgui.menu_item("Open Data File...", "Ctrl+O", False)
        clicked = r[0] if isinstance(r, tuple) else r
        if clicked:
            state._open_file_requested = True

        imgui.separator()

        r2 = imgui.menu_item("Save Session", "", False)
        c2 = r2[0] if isinstance(r2, tuple) else r2
        if c2:
            save_session(state)

        imgui.end_menu()


# ================================================================
#  Callbacks
# ================================================================
def _on_init():
    state.init_gl()
    load_session(state)
    state.load_history()

    io = imgui.get_io()
    io.config_input_text_cursor_blink = False

    state._menubar_h = imgui.get_frame_height()


def _frame_update():
    if not state.initialized:
        return

    io = imgui.get_io()
    state._menubar_h = imgui.get_frame_height()

    # Save session when user closes the window (avoids before_exit crash)
    try:
        if hello_imgui.get_runner_params().app_shall_exit:
            save_session(state)
    except Exception:
        pass

    # Open-file request from panel button or menu bar
    if state._open_file_requested:
        state._open_file_requested = False
        _open_file_dialog()     # blocks via subprocess until dialog closes

    # Handle deferred file-load from dialog
    if state._pending_file:
        state.load_data_file(state._pending_file)
        state._pending_file = ""

    if state.anim_playing and state.anim_enabled and not state.mode_3d:
        state.anim_time += io.delta_time * state.anim_speed
        state.math_data_needs_update = True

    if state._dark_mode_dirty:
        state._dark_mode_dirty = False
        _apply_style(dark=state.dark_mode)

    draw_panel(state, plot_rect)
    state.process_interactions(plot_rect)

    if state.math_data_needs_update:
        state.generate_math_data()
    elif not state.mode_3d and state.needs_resample():
        state.generate_math_data()

    if state.math_data_needs_update3d:
        state.generate_math_data_3d(state.current_expr_3d)

    if state.math_data_needs_update_param:
        state.generate_parametric_data()

    if state.math_data_needs_update_curve:
        state.generate_curve_data()

    render_frame(state, plot_rect)

    # Screenshot – read pixels after render so the framebuffer is fully drawn
    if state._save_screenshot_requested:
        state._save_screenshot_requested = False
        state.save_screenshot()


def _apply_style(dark: bool = False):
    s = imgui.get_style()
    accent   = imgui.ImVec4(0.20, 0.45, 0.80, 1.0)
    accent_h = imgui.ImVec4(0.15, 0.35, 0.70, 1.0)

    if dark:
        win_bg   = imgui.ImVec4(0.13, 0.14, 0.16, 1.0)
        input_bg = imgui.ImVec4(0.20, 0.22, 0.26, 1.0)
        input_hov= imgui.ImVec4(0.26, 0.28, 0.34, 1.0)
        input_act= imgui.ImVec4(0.30, 0.33, 0.40, 1.0)
        btn      = imgui.ImVec4(0.22, 0.24, 0.28, 1.0)
        btn_hov  = imgui.ImVec4(0.30, 0.33, 0.38, 1.0)
        btn_act  = imgui.ImVec4(0.18, 0.20, 0.24, 1.0)
        sep      = imgui.ImVec4(0.35, 0.37, 0.42, 1.0)
        text     = imgui.ImVec4(0.90, 0.90, 0.92, 1.0)
        txt_dis  = imgui.ImVec4(0.50, 0.52, 0.56, 1.0)
        menu_bg  = imgui.ImVec4(0.10, 0.11, 0.13, 1.0)
    else:
        win_bg   = imgui.ImVec4(1.0,  1.0,  1.0,  1.0)
        input_bg = imgui.ImVec4(0.90, 0.90, 0.90, 1.0)
        input_hov= imgui.ImVec4(0.80, 0.80, 0.80, 1.0)
        input_act= imgui.ImVec4(0.70, 0.70, 0.70, 1.0)
        btn      = imgui.ImVec4(0.82, 0.82, 0.82, 1.0)
        btn_hov  = imgui.ImVec4(0.70, 0.70, 0.70, 1.0)
        btn_act  = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)
        sep      = imgui.ImVec4(0.65, 0.65, 0.65, 1.0)
        text     = imgui.ImVec4(0.05, 0.05, 0.05, 1.0)
        txt_dis  = imgui.ImVec4(0.50, 0.50, 0.50, 1.0)
        menu_bg  = imgui.ImVec4(0.95, 0.95, 0.97, 1.0)

    s.set_color_(imgui.Col_.window_bg,          win_bg)
    s.set_color_(imgui.Col_.child_bg,           win_bg)
    s.set_color_(imgui.Col_.popup_bg,           win_bg)
    s.set_color_(imgui.Col_.frame_bg,           input_bg)
    s.set_color_(imgui.Col_.frame_bg_hovered,   input_hov)
    s.set_color_(imgui.Col_.frame_bg_active,    input_act)
    s.set_color_(imgui.Col_.title_bg,           btn)
    s.set_color_(imgui.Col_.title_bg_active,    btn_hov)
    s.set_color_(imgui.Col_.button,             btn)
    s.set_color_(imgui.Col_.button_hovered,     btn_hov)
    s.set_color_(imgui.Col_.button_active,      btn_act)
    s.set_color_(imgui.Col_.slider_grab,        accent)
    s.set_color_(imgui.Col_.slider_grab_active, accent_h)
    s.set_color_(imgui.Col_.check_mark,         accent)
    s.set_color_(imgui.Col_.header,             btn)
    s.set_color_(imgui.Col_.header_hovered,     btn_hov)
    s.set_color_(imgui.Col_.header_active,      btn_act)
    s.set_color_(imgui.Col_.separator,          sep)
    s.set_color_(imgui.Col_.text,               text)
    s.set_color_(imgui.Col_.text_disabled,      txt_dis)
    s.set_color_(imgui.Col_.border,             sep)
    s.set_color_(imgui.Col_.menu_bar_bg,        menu_bg)
    s.window_rounding = 4.0
    s.frame_rounding  = 3.0
    s.grab_rounding   = 3.0
    s.item_spacing    = imgui.ImVec2(6, 5)
    s.frame_padding   = imgui.ImVec2(5, 4)
    s.window_padding  = imgui.ImVec2(8, 8)


def _setup_style():
    _apply_style(dark=False)


if __name__ == "__main__":
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "PyGraph"
    params.app_window_params.window_geometry.size = (1280, 720)
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    params.imgui_window_params.show_menu_bar = True
    try:
        params.imgui_window_params.menu_app_title = ""   # suppress duplicate title in menu bar
    except AttributeError:
        pass

    params.callbacks.post_init         = _guarded(_on_init)
    params.callbacks.setup_imgui_style = _guarded(_setup_style)
    params.callbacks.show_menus        = _guarded(_show_menus)
    params.callbacks.custom_background = _guarded(_frame_update)

    try:
        hello_imgui.run(params)
    except SystemError as e:
        if "nb_func_error_except" in str(e):
            pass   # nanobind/Python-3.14 cleanup race – harmless, app already finished
        else:
            raise
