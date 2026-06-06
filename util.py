import sympy as sp
import numpy as np

default_expr    = "sin(x) + exp(-x) + log(x)"
default_expr_3d = "sin(sqrt(x**2 + y**2))"


def parse_to_numpy(equation_str, variables_str='x'):
    """
    Converts a math equation string into a NumPy-backed function.
    Supports both 2-D (variable='x') and 3-D (variables='x y') expressions.
    """
    expr     = sp.sympify(equation_str)
    vars_sym = sp.symbols(variables_str)
    # Extend lambdify with atan2 and other numpy functions not in sympy by default
    extra = {"atan2": np.arctan2, "arctan2": np.arctan2}
    return sp.lambdify(vars_sym, expr, modules=[extra, 'numpy'])


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
