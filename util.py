import threading
import sympy as sp
import numpy as np

default_expr    = "sin(x)"
default_expr_3d = "sin(sqrt(x**2 + y**2))"

_EXTRA = {"atan2": np.arctan2, "arctan2": np.arctan2}


def parse_to_numpy(equation_str, variables_str='x', timeout=3.0):
    """
    Converts a math equation string into a NumPy-backed lambda.
    variables_str can be 'x', 'x y', 'x t', etc.
    Raises TimeoutError if sympify/lambdify hangs beyond timeout seconds.
    """
    result = [None]
    error  = [None]

    def _worker():
        try:
            expr     = sp.sympify(equation_str)
            vars_sym = sp.symbols(variables_str)
            result[0] = sp.lambdify(vars_sym, expr, modules=[_EXTRA, 'numpy'])
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Expression timed out (>{timeout}s): {equation_str}")
    if error[0] is not None:
        raise error[0]
    return result[0]


def format_label(val):
    """Format axis label cleanly."""
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val >= 1000 or (abs_val < 0.01 and abs_val > 0):
        return f"{val:.2e}"
    if abs_val >= 10:
        return f"{val:.0f}"
    if abs_val >= 1:
        return f"{val:.1f}"
    return f"{val:.2f}"
