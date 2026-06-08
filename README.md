# PyGraph Plotter

An OpenGL-powered interactive math function plotter with 2D, 3D surface, parametric surface, and space curve modes.

## Install

```bash
pip install pygraph-plotter
```

## Launch

```bash
pygraph                  # open empty plotter
pygraph "sin(x)"         # pre-load an expression
pygraph "x**2 - 2*x"
python -m pygraph
```

## Features

- **2D equations** — plot any expression in `x` with live pan/zoom
- **3D surfaces** — `f(x, y)` rendered with OpenGL shading and 9 colormaps
- **Parametric surfaces** — `(x(u,v), y(u,v), z(u,v))`
- **Space curves** — 3D parametric curves with tube rendering
- **Data files** — load CSV/TSV as scatter, line, histogram, KDE, heatmap, or violin plot
- **Multi-curve overlay** — up to 8 equations plotted simultaneously in 2D
- **Animation** — use `t` in your expression and press Play (`sin(x - t)`)
- **Log-scale axes** — toggle Log X / Log Y independently
- **Zoom-to-fit** — auto-scale viewport to the curve's Y range
- **Cursor readout** — hover shows live x/y coordinates
- **Pinned points** — double-click to drop coordinate markers
- **Zoom box** — Shift+drag to rubber-band zoom a region
- **Dark mode** — toggle in the panel
- **Save PNG** — Ctrl+S or button; copy to clipboard in one click
- **Session persistence** — last state restored on relaunch
- **Equation history** — recent expressions listed for quick reuse

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Scroll | Zoom in/out |
| Left drag | Pan |
| Shift + drag | Zoom box |
| Double-click | Pin coordinate marker |
| R | Reset / zoom to fit |
| Space | Play / pause animation |
| Ctrl+S | Save screenshot |
| ? | Show shortcut help |

## Requirements

- Python ≥ 3.10
- OpenGL 3.3-capable GPU

Dependencies installed automatically: `PyOpenGL`, `imgui_bundle`, `numpy`, `sympy`, `Pillow`.

## License

MIT
