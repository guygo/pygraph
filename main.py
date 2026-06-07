import sys
import traceback
import subprocess
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
#  File dialog (native per-platform, runs in a background thread
#  so the OpenGL render loop keeps running while the dialog is open)
# ================================================================
def _open_file_dialog():
    """Open a native file-picker dialog without blocking the render loop."""
    def _worker():
        path = ""
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["osascript", "-e",
                     'POSIX path of (choose file with prompt "Open Data File")'],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                elif result.returncode != 1:
                    state.last_error = f"Dialog error: {result.stderr.strip()}"

            elif sys.platform == "win32":
                ps_script = (
                    "Add-Type -AssemblyName System.Windows.Forms;"
                    "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                    "$d.Filter = 'Data files (*.csv;*.tsv;*.txt)|*.csv;*.tsv;*.txt|All files (*.*)|*.*';"
                    "$d.Title = 'Open Data File';"
                    "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    path = result.stdout.strip()

            else:
                # Linux – try zenity, then kdialog, then tkinter as last resort
                for cmd in (
                    ["zenity", "--file-selection", "--title=Open Data File"],
                    ["kdialog", "--getopenfilename", ".", "*.csv *.tsv *.txt"],
                ):
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            path = result.stdout.strip()
                            break
                    except FileNotFoundError:
                        continue
                else:
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

        except Exception as e:
            state.last_error = f"File dialog error: {e}"

        if path:
            state._pending_file = path

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


def _setup_style():
    s = imgui.get_style()
    white      = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
    input_bg   = imgui.ImVec4(0.90, 0.90, 0.90, 1.0)
    input_hov  = imgui.ImVec4(0.80, 0.80, 0.80, 1.0)
    input_act  = imgui.ImVec4(0.70, 0.70, 0.70, 1.0)
    btn        = imgui.ImVec4(0.82, 0.82, 0.82, 1.0)
    btn_hov    = imgui.ImVec4(0.70, 0.70, 0.70, 1.0)
    btn_act    = imgui.ImVec4(0.55, 0.55, 0.55, 1.0)
    sep        = imgui.ImVec4(0.65, 0.65, 0.65, 1.0)
    accent     = imgui.ImVec4(0.20, 0.45, 0.80, 1.0)
    accent_h   = imgui.ImVec4(0.15, 0.35, 0.70, 1.0)
    black_text = imgui.ImVec4(0.05, 0.05, 0.05, 1.0)

    s.set_color_(imgui.Col_.window_bg,          white)
    s.set_color_(imgui.Col_.child_bg,           white)
    s.set_color_(imgui.Col_.popup_bg,           white)
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
    s.set_color_(imgui.Col_.text,               black_text)
    s.set_color_(imgui.Col_.text_disabled,      imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
    s.set_color_(imgui.Col_.border,             sep)
    s.set_color_(imgui.Col_.menu_bar_bg,        imgui.ImVec4(0.95, 0.95, 0.97, 1.0))
    s.window_rounding = 4.0
    s.frame_rounding  = 3.0
    s.grab_rounding   = 3.0
    s.item_spacing    = imgui.ImVec2(6, 5)
    s.frame_padding   = imgui.ImVec2(5, 4)
    s.window_padding  = imgui.ImVec2(8, 8)


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
