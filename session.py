import json
import os

SESSION_FILE = os.path.expanduser("~/.pygraph_session.json")

_FIELDS_2D = [
    'current_expr', 'min_x', 'max_x', 'min_y', 'max_y',
    'resolution', 'effect_mode', 'graph_color', 'line_thickness',
    'connect_lines', 'show_axis_grid', 'show_numbers',
    'log_scale_x', 'log_scale_y',
]
_FIELDS_3D = [
    'current_expr_3d', 'resolution_3d', 'colormap', 'surface_alpha',
    'show_wireframe', 'wire_alpha', 'circular_mask', 'circular_max_r',
    'range3d_x', 'range3d_y', 'cam_theta', 'cam_phi', 'cam_dist',
    'param_expr_x', 'param_expr_y', 'param_expr_z', 'param_f_expr',
    'param_u_range', 'param_v_range', 'param_res_u', 'param_res_v',
    'curve_expr_x', 'curve_expr_y', 'curve_expr_z',
    'curve_t_range', 'curve_resolution', 'curve_tube_radius',
    'show_3d_box', 'show_3d_grid', 'show_3d_labels',
]
_FIELDS_MODE = ['mode_3d', 'mode_parametric', 'mode_space_curve']

_ALL_FIELDS = _FIELDS_2D + _FIELDS_3D + _FIELDS_MODE


def save_session(state):
    data = {}
    for f in _ALL_FIELDS:
        v = getattr(state, f, None)
        if isinstance(v, (list, tuple)):
            v = list(v)
        data[f] = v
    try:
        with open(SESSION_FILE, 'w') as fh:
            json.dump(data, fh, indent=2)
    except Exception as e:
        print(f"Session save error: {e}")


def load_session(state):
    if not os.path.exists(SESSION_FILE):
        return
    try:
        with open(SESSION_FILE) as fh:
            data = json.load(fh)
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        # Sync input buffers from restored state
        state.input_buf_2d = state.current_expr
        state.input_buf_3d = state.current_expr_3d
        state.input_buf_px = state.param_expr_x
        state.input_buf_py = state.param_expr_y
        state.input_buf_pz = state.param_expr_z
        state.input_buf_pf = state.param_f_expr
        state.input_buf_cx = state.curve_expr_x
        state.input_buf_cy = state.curve_expr_y
        state.input_buf_cz = state.curve_expr_z
    except Exception as e:
        print(f"Session load error: {e}")
