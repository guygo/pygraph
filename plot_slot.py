from dataclasses import dataclass, field

# Plot-type constants
PLOT_EQUATION  = 0   # math expression f(x)
PLOT_SCATTER   = 1   # x,y columns from file (dots)
PLOT_LINE_DATA = 2   # x,y columns from file (connected)
PLOT_HISTOGRAM = 3   # single column, binned bars
PLOT_KDE       = 4   # kernel density estimate (smooth curve)
PLOT_HEATMAP2D = 5   # 2-D binned density (heatmap)
PLOT_VIOLIN    = 6   # mirrored KDE (violin)

PLOT_TYPE_NAMES = ["Equation", "Scatter", "Line", "Histogram", "KDE", "Heatmap", "Violin"]


@dataclass
class PlotSlot:
    # ── equation mode ─────────────────────────────────────────────────────────
    expr: str = "sin(x)"
    input_buf: str = "sin(x)"
    _cached_fn: object = None
    _cached_expr: str = ""
    _cached_anim: bool = False

    # ── appearance ────────────────────────────────────────────────────────────
    color: list = field(default_factory=lambda: [0.0, 0.5, 0.8])
    effect_mode: int = 0
    visible: bool = True
    line_thickness: float = 4.0
    connect_lines: bool = True
    point_size: float = 6.0

    # ── state ─────────────────────────────────────────────────────────────────
    last_error: str = ""
    geometry: object = None       # DynamicLinePlot (equation / scatter / line / KDE)
    hist_geo: object = None       # HistogramPlot (histogram only)
    heatmap_geo: object = None    # ColoredMesh (heatmap)
    _sampled_y: object = None     # for zoom-to-fit
    _violin_quartiles: object = None
    _violin_max_kde: float = 0.0

    # ── data mode ─────────────────────────────────────────────────────────────
    plot_type: int = PLOT_EQUATION
    raw_data: object = None       # float32 ndarray (N, M)
    col_names: list = None        # column header strings
    col_x: int = 0
    col_y: int = 1
    col_hist: int = 0
    hist_bins: int = 30
    source_file: str = ""
