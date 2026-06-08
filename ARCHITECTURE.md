# PyGraph — Architecture & Math Reference

This document explains every subsystem in the codebase: coordinate systems, the
OpenGL pipeline, all GLSL shaders, the 3-D camera mathematics, surface geometry,
lighting, colormaps, 2-D sampling, animation, and the data-visualisation layer.

---

## Table of Contents

1. [High-level architecture](#1-high-level-architecture)
2. [Coordinate systems](#2-coordinate-systems)
3. [2-D rendering pipeline](#3-2-d-rendering-pipeline)
   - [Grid shader](#31-grid-shader)
   - [Solid line shader (geometry-shader anti-aliasing)](#32-solid-line-shader)
   - [Effect shaders — Neon / Plasma / Electric](#33-effect-shaders)
   - [Fill shader (histogram bars)](#34-fill-shader)
   - [Point shader (scatter)](#35-point-shader)
   - [Colored-vertex shader (heatmap)](#36-colored-vertex-shader)
   - [Log-scale normalization](#37-log-scale-normalization)
4. [2-D sampling strategy](#4-2-d-sampling-strategy)
5. [Animation](#5-animation)
6. [3-D rendering pipeline](#6-3-d-rendering-pipeline)
   - [Camera model (spherical coordinates)](#61-camera-model)
   - [View matrix (look-at)](#62-view-matrix-look-at)
   - [Perspective projection matrix](#63-perspective-projection-matrix)
   - [Model matrix (scene normalization)](#64-model-matrix-scene-normalization)
   - [Full MVP](#65-full-mvp)
7. [3-D surface geometry](#7-3-d-surface-geometry)
   - [Mesh construction](#71-mesh-construction)
   - [Central-difference normals](#72-central-difference-normals)
   - [Index array & NaN culling](#73-index-array--nan-culling)
8. [3-D lighting — Blinn-Phong](#8-3-d-lighting--blinn-phong)
9. [Colormaps](#9-colormaps)
10. [3-D overlay projection](#10-3-d-overlay-projection)
11. [Data visualisation types](#11-data-visualisation-types)
12. [File-by-file reference](#12-file-by-file-reference)

---

## 1  High-level architecture

```
main.py          hello_imgui run-loop, callbacks, style
  │
  ├── panel.py        Dear ImGui control panel (left sidebar)
  ├── state.py        AppState — all mutable data + math computation
  │     ├── util.py        SymPy → NumPy expression parser
  │     ├── plot_slot.py   PlotSlot dataclass (one curve/dataset)
  │     ├── session.py     JSON save/restore of session
  │     ├── data_loader.py CSV/TSV reader
  │     └── plot_hist.py   Histogram/KDE/violin geometry builders
  │
  ├── renderer.py     OpenGL draw calls (2-D and 3-D)
  │     ├── shaders.py     2-D GLSL source strings
  │     └── shaders3d.py   3-D GLSL source strings
  │
  ├── gui.py          Axis labels, grid lines, overlay widgets (imgui draw-list)
  │     └── overlay3d.py   3-D bounding box / tick labels
  │
  └── plot*.py        OpenGL VAO/VBO geometry classes
        plot.py            DynamicLinePlot, GridGeometry, …
        plot3d.py          SurfacePlot3D
        plot3d_parametric  ParametricSurfacePlot
        plot3d_curve       SpaceCurve3D
```

The main thread runs the `hello_imgui` loop. Every frame:

1. `draw_panel` builds the Dear ImGui UI.
2. `state.process_interactions` handles mouse/keyboard.
3. `state.generate_math_data*` recomputes geometry when needed.
4. `render_frame` issues OpenGL draw calls.
5. Axis labels and overlays are drawn via imgui's background draw-list
   (which renders *after* OpenGL, so labels always appear on top).

---

## 2  Coordinate systems

Four coordinate systems are active simultaneously.

### Data space
The mathematical domain chosen by the user: `[min_x, max_x] × [min_y, max_y]`.
All equation values live here.

### Screen space (pixels)
Top-left origin, Y increases downward — the standard Dear ImGui / OS convention.
`plot_rect = {x, y, w, h}` in `gui.py` holds the pixel rectangle of the plot area.

### Framebuffer space
Same as screen space but multiplied by `display_framebuffer_scale` (e.g. 2× on
Retina / HiDPI displays). OpenGL scissor, viewport, and `glReadPixels` use this.

### NDC (Normalized Device Coordinates)
OpenGL's clip space: X and Y each in `[−1, +1]`, Y increasing *upward*.

**Conversion formulas (used in every vertex shader):**

```
nx = (data_x − min_x) / (max_x − min_x)   →  [0, 1]
ny = (data_y − min_y) / (max_y − min_y)   →  [0, 1]

ndc_x = nx * 2 − 1
ndc_y = ny * 2 − 1
```

**Screen → data (used in `gui.py::screen_to_data`):**

```
data_x = min_x + (max_x − min_x) * (sx − plot_x) / plot_w
data_y = min_y + (max_y − min_y) * (1 − (sy − plot_y) / plot_h)
```

Note the `1 − ...` on Y: screen Y is downward, data Y is upward.

---

## 3  2-D rendering pipeline

All 2-D shaders share the same **linear normalization** vertex stage.
Log-scale adds an alternative path (see §3.7).

### 3.1  Grid shader

**File:** `shaders.py` — `GRID_VERTEX_SHADER` / `GRID_FRAGMENT_SHADER`

The grid is a single screen-filling quad (two triangles) whose vertices carry
their own data-space coordinates. The fragment shader uses `fract` to draw grid
lines analytically — no per-line geometry is needed.

```glsl
// Fragment shader core
vec2 grid = fract(vGraphPos / uGridSpacing);

if (grid.x < gridThickness || grid.x > (1.0 − gridThickness) ||
    grid.y < gridThickness || grid.y > (1.0 − gridThickness))
    FragColor = vec4(0.8, 0.8, 0.8, 0.5);   // draw line
else
    discard;                                  // transparent fill
```

`fract(p / spacing)` maps each fragment's world position to `[0, 1)` within one
grid cell. Values near 0 or 1 are the grid edges. `gridThickness = 0.015` means
lines are 1.5 % of a grid cell wide.

> **Why use a fragment-shader grid instead of line geometry?**
> A single quad costs two triangles regardless of how many grid lines appear.
> Line geometry would cost O(lines) draw calls and vertex uploads.

---

### 3.2  Solid line shader

**File:** `shaders.py` — `SOLID_VERTEX_SHADER` / `SOLID_GEOMETRY_SHADER` / `SOLID_FRAGMENT_SHADER`

This is a three-stage pipeline that turns a polyline into pixel-perfect,
anti-aliased strokes of arbitrary width.

#### Vertex shader
Maps each 2-D data point to NDC (see §2).

#### Geometry shader
Runs once per line *segment* (two consecutive vertices). It expands each segment
into a screen-aligned rectangle (a quad = two triangles).

```
Segment:  p0 ────────── p1

Quad:
  p0−perp ──────────── p1−perp
  p0+perp ──────────── p1+perp
```

```glsl
vec2 dir  = normalize((p1.xy − p0.xy) * uViewport);   // pixel-space direction
vec2 perp = vec2(−dir.y, dir.x) / uViewport * uLineWidth;
```

Key steps:
1. Compute the segment direction in *pixel space* (multiply by viewport size to
   undo the aspect-ratio distortion that NDC introduces).
2. Rotate 90° to get the perpendicular: `(−dy, dx)`.
3. Divide back by viewport to return to NDC units.
4. Emit four vertices: `p0 ± perp`, `p1 ± perp`.
5. Pass `gDist = ±1` so the fragment shader knows how far from the centre line
   each fragment sits.

#### Fragment shader (anti-aliasing)

```glsl
float d     = abs(gDist);               // 0 = centre, 1 = edge
float alpha = 1.0 − smoothstep(0.60, 1.0, d);
```

`smoothstep(a, b, x)` returns 0 for `x ≤ a`, 1 for `x ≥ b`, and a smooth cubic
in between. Here alpha fades from 1 (at `d = 0.60`) to 0 (at `d = 1.0`), giving
a soft, anti-aliased edge over the outermost 40 % of the line width.

---

### 3.3  Effect shaders

**File:** `shaders.py` — `EFFECT_VERTEX_SHADER` / `EFFECT_GEOMETRY_SHADER` / `EFFECT_FRAGMENT_SHADER`

Same vertex+geometry structure as the solid shader, but the fragment shader adds
animated visual effects driven by `uTime` (seconds since launch).

The helper `hsv2rgb` converts HSV colour to RGB:

```glsl
vec3 hsv2rgb(float h, float s, float v) {
    // h in [0, 360], s and v in [0, 1]
    float c = v * s;                          // chroma
    float x = c * (1 − |mod(h/60, 2) − 1|);  // secondary component
    float m = v − c;                           // match value
    // select sextant, return (r,g,b) + m
}
```

#### Mode 1 — Neon Glow Rainbow

```glsl
float hue  = mod(gNorm.x * 360.0 − uTime * 45.0, 360.0);
vec3  col  = hsv2rgb(hue, 1.0, 1.0);
float core = 1.0 − smoothstep(0.0, 0.30, d);   // bright inner core
float glow = 1.0 − smoothstep(0.30, 1.0, d);   // fading outer glow
```

- The hue sweeps through the colour wheel as a function of X position (`gNorm.x`)
  and scrolls over time (`uTime * 45` degrees/s).
- Two soft-edge layers: a tight bright **core** (inner 30 % of width) and a wider
  **glow** halo with 45 % opacity.
- The core colour is blended toward white with `mix(col, vec3(1), core * 0.55)` to
  simulate an overexposed hot centre.

#### Mode 2 — Plasma

```glsl
float v1 = sin(px * 1.5 + uTime * 0.5);
float v2 = sin(py * 1.5 − px * 0.5 + uTime * 0.3);
float v3 = sin(sqrt(px*px + py*py) * 1.8 − uTime * 0.7);
float plasma = (v1 + v2 + v3) / 3.0;   // −1..+1
float hue    = mod(plasma * 160.0 + uTime * 25.0 + 200.0, 360.0);
```

Three sine waves (linear X, diagonal, and radial) are summed and averaged.
Their interference produces organic swirling patterns. The result drives a hue in
the blue-purple-magenta band (+200° offset). All three waves evolve independently
in time, preventing periodicity from becoming obvious.

#### Mode 3 — Electric Blue-White

```glsl
float flicker = sin(gNorm.x * 28.0 − uTime * 9.0) * 0.5 + 0.5;
float hue     = 195.0 + flicker * 45.0;   // cyan–blue range
float core    = exp(−d * d * 5.5);        // Gaussian profile
```

- A fast sine wave (`28.0` spatial frequency, `9.0` rad/s temporal) creates the
  flickering along the curve.
- The core brightness follows a **Gaussian** `e^(−d²·5.5)` rather than the linear
  `smoothstep` of mode 1 — this gives a sharper, more electric look.
- Hue varies between cyan (195°) and azure (240°) with the flicker.

---

### 3.4  Fill shader

**File:** `shaders.py` — `FILL_VERTEX_SHADER` / `FILL_FRAGMENT_SHADER`

No geometry shader. Used for histogram bars, KDE fill areas, violin shapes —
geometry that is already triangulated on the CPU. The fragment shader simply
outputs a uniform `(uColor, uAlpha)`.

---

### 3.5  Point shader

**File:** `shaders.py` — `POINT_VERTEX_SHADER` / `POINT_FRAGMENT_SHADER`

Used for scatter plots. `GL_POINTS` renders one square quad per vertex; the
fragment shader clips it to a circle and soft-edges it:

```glsl
vec2  c     = gl_PointCoord − 0.5;    // [−0.5, 0.5]² centred at origin
float d     = length(c) * 2.0;        // 0 = centre, 1 = edge
float alpha = 1.0 − smoothstep(0.65, 1.0, d);
```

`gl_PointCoord` is the built-in UV within the point quad, ranging from (0,0) to
(1,1). Subtracting 0.5 centres it; multiplying `length` by 2 maps the circle
edge to `d = 1`. The `smoothstep` anti-aliases the circle boundary.

---

### 3.6  Colored-vertex shader

**File:** `shaders.py` — `COLORED_VERTEX_SHADER` / `COLORED_FRAGMENT_SHADER`

Used for the 2-D heatmap. Each vertex carries its own RGB colour (pre-computed on
the CPU by the colourmap function). The fragment shader interpolates vertex colours
across triangles automatically (Gouraud shading).

---

### 3.7  Log-scale normalization

When `log_scale_x` or `log_scale_y` is enabled, the linear normalization in each
vertex shader is replaced with a logarithmic one:

**Linear:**
```
nx = (x − x_min) / (x_max − x_min)
```

**Log:**
```glsl
nx = (log(x) − log(x_min)) / (log(x_max) − log(x_min))
```

This maps `x_min → 0`, `x_max → 1` on a *logarithmic* scale, so equal pixel
distances represent equal *ratios* rather than equal differences.

The `max(aPos.x, 1e-9)` guard prevents `log(0) = −∞` from crashing the shader.

On the CPU side (`state.py::_build_x_array`), the sample points are also generated
in log space:

```python
np.exp(np.linspace(np.log(x_start), np.log(x_end), n))
```

This ensures sample density is uniform on the log scale rather than bunching
points near the lower end.

Tick labels in `gui.py::draw_axis_labels` switch to powers-of-10 ticks
(`1e-3, 1e-2, 0.1, 1, 10, …`) with intermediate marks at `×2` and `×5`.

---

## 4  2-D sampling strategy

**File:** `state.py` — `_compute_sample_count`, `_sample_range`, `needs_resample`

#### Adaptive sample count

```python
OVERDRAW       = 2.0    # sample 2× the visible range for smooth pan
SAMPLES_PER_PX = 3      # 3 samples per screen pixel
MIN_VISIBLE_PTS = 2000
MAX_TOTAL_PTS   = 400_000
```

```python
visible_pts = int(plot_width_pixels * SAMPLES_PER_PX)
total_pts   = int(visible_pts * OVERDRAW)
n           = clamp(total_pts, MIN_VISIBLE_PTS, MAX_TOTAL_PTS)
```

The X range is extended by `OVERDRAW / 2` on each side so that panning does not
immediately trigger a resample — the overdraw region acts as a buffer.

#### Lazy resampling trigger (`needs_resample`)

A resample is triggered only when:
- The viewport has panned so that less than 50 % overdraw remains on either side
  (the curve is about to run off-screen), **or**
- The zoom level changed by more than 20 %.

This prevents a resample on every frame during smooth pan.

---

## 5  Animation

**File:** `util.py::parse_to_numpy`, `state.py::generate_math_data`, `main.py::_frame_update`

When the user writes an expression containing `t` (e.g. `sin(x − t)`) and enables
animation, the code detects `'t' in slot.expr` and compiles the expression as a
function of *two* variables:

```python
var_str = 'x t'    # instead of just 'x'
fn = parse_to_numpy(slot.expr, variables_str=var_str)
# fn is now a callable:  fn(x_array, t_scalar)
```

Each frame, `_frame_update` advances the clock:

```python
state.anim_time += io.delta_time * state.anim_speed
state.math_data_needs_update = True
```

`generate_math_data` then calls `fn(x, self.anim_time)`, which broadcasts the
scalar `t` across the entire `x` array via NumPy.

The parse step uses `sympy.sympify` → `sympy.lambdify` in a background thread
with a 3-second timeout (implemented via `threading.Thread.join(timeout)`) to
prevent SymPy from hanging the UI on pathological input.

---

## 6  3-D rendering pipeline

**File:** `state.py::get_mvp`, `renderer.py::_render_3d`

The 3-D camera uses **spherical coordinates** and a classic MVP pipeline.

### 6.1  Camera model

The camera orbits the origin at distance `r = cam_dist`. Its position in Cartesian
coordinates is:

```
eye_x = r · sin(θ) · cos(φ)
eye_y = r · sin(θ) · sin(φ)
eye_z = r · cos(θ)
```

- **θ (theta)** — polar angle from the Z axis (elevation). Range `[0.05, π−0.05]`
  (clamped to prevent gimbal lock at the poles).
- **φ (phi)** — azimuthal angle in the XY plane (rotation). Unbounded.
- **r** — distance from origin (zoom). Range `[0.3, 30]`.

Mouse drag maps to these angles:
```python
cam_phi   += mouse_delta.x * 0.005    # horizontal drag → rotate
cam_theta += mouse_delta.y * 0.005    # vertical drag   → tilt
```

Scroll wheel scales `cam_dist`:
```python
cam_dist *= (1.0 − scroll * 0.08)
```

---

### 6.2  View matrix (look-at)

The view matrix transforms world coordinates into camera space. It is built with
the classic *look-at* algorithm:

```python
f  = target − eye;        f  /= |f|      # forward (into scene)
rv = cross(f, up);        rv /= |rv|     # right
uv = cross(rv, f)                        # corrected up
```

The world `up` vector is `(0, 0, 1)` (Z-up convention). The cross-products
produce an orthonormal right-handed camera frame `(rv, uv, −f)`.

The resulting view matrix (row-major):

```
V = | rv.x   rv.y   rv.z   −dot(rv, eye) |
    | uv.x   uv.y   uv.z   −dot(uv, eye) |
    | −f.x   −f.y   −f.z    dot(f,  eye) |
    |  0      0      0       1            |
```

The translation column `−dot(basis, eye)` moves the camera to the origin in
camera space.

---

### 6.3  Perspective projection matrix

Standard OpenGL perspective with vertical FOV `fov_y = 45°`, near plane `n = 0.01`,
far plane `fa = 100`:

```python
fv     = 1 / tan(fov_y / 2)     # focal length (cotangent of half-FOV)
aspect = width / height

P = | fv/aspect   0     0               0          |
    |    0        fv    0               0          |
    |    0         0   (fa+n)/(n−fa)   2·fa·n/(n−fa) |
    |    0         0   −1              0          |
```

This maps the view frustum to NDC:
- X: `[−aspect·tan(fov/2), +aspect·tan(fov/2)]` → `[−1, +1]`
- Y: `[−tan(fov/2), +tan(fov/2)]` → `[−1, +1]`
- Z: `[−n, −fa]` (OpenGL convention) → `[−1, +1]`

The `P[3,2] = −1` row performs the perspective divide: dividing by `−z` makes
distant objects appear smaller.

---

### 6.4  Model matrix (scene normalization)

Before rendering, the surface is normalized so that its bounding box fits in
`[−1, +1]³` regardless of the data range. The model matrix does this:

```python
cx, cy, cz = centre of bounding box
sx, sy, sz = 2.0 / (x_range, y_range, z_range)   # scale factors

M = | sx   0    0   −cx·sx |
    |  0   sy   0   −cy·sy |
    |  0    0   sz  −cz·sz |
    |  0    0    0    1    |
```

Step 1 (translation column): subtract the centre so the surface is centred at the
origin.
Step 2 (diagonal): scale so the longest axis spans exactly `[−1, +1]`.

This normalization ensures the camera's fixed `cam_dist ≈ 3.5` always frames the
surface properly regardless of the data's physical units.

---

### 6.5  Full MVP

```python
MVP = P @ V @ M
```

Applied in that order (right-to-left):
1. **M** — centre and scale the surface to `[−1,+1]³`
2. **V** — rotate and translate into camera space
3. **P** — project to NDC with perspective

The vertex shader applies it in one operation:

```glsl
gl_Position = uMVP * vec4(aPos, 1.0);
```

---

## 7  3-D surface geometry

**File:** `plot3d.py::SurfacePlot3D`

### 7.1  Mesh construction

For a surface `z = f(x, y)` sampled on an `N × M` grid:

```python
X, Y = np.meshgrid(x_data, y_data)   # (M, N) 2-D grids
R    = sqrt(X² + Y²)                  # radius (for circular mask)
A    = arctan2(Y, X)                  # angle in [−π, π]
```

Each vertex stores 8 floats (32 bytes):

| Location | Offset | Data |
|----------|--------|------|
| 0 | 0  | x, y, z  (position) |
| 1 | 12 | nx, ny, nz  (normal) |
| 2 | 24 | t  (height in [0,1]) |
| 3 | 28 | a  (angle in [−π,π]) |

`t = (z − z_min) / (z_max − z_min)` is used to look up the colourmap.
`a` is used by the HSV-angle colourmap which colours by direction in XY.

### 7.2  Central-difference normals

The surface normal at each grid point is the cross product of the two tangent
vectors. Central differences approximate the partial derivatives efficiently:

```python
zp = pad(z_fill, 1, mode='edge')          # replicate boundary values

dzdx = (zp[1:-1, 2:]  − zp[1:-1, :-2]) / (2 · dx)   # ∂z/∂x
dzdy = (zp[2:,   1:-1] − zp[:-2,  1:-1]) / (2 · dy)  # ∂z/∂y
```

The tangent in X is `(dx, 0, dzdx·dx)` and in Y is `(0, dy, dzdy·dy)`.
Their cross product gives the normal:

```
N = (−dzdx, −dzdy, 1)    (before normalization)
```

This is then unit-normalized per vertex.

> **Why `−dzdx, −dzdy, +1`?**
> The cross product `∂P/∂x × ∂P/∂y = (1,0,dzdx) × (0,1,dzdy)` expands to
> `(0·dzdy − dzdx·1, dzdx·0 − 1·dzdy, 1·1 − 0·0) = (−dzdx, −dzdy, 1)`.
> The Z component is always positive, so normals point upward by construction.

### 7.3  Index array & NaN culling

Each quad `(i,j)` is split into two triangles:

```
Triangle 1:  top-left, bottom-left, top-right
Triangle 2:  top-right, bottom-left, bottom-right
```

A triangle is included only if **all three** of its vertices have finite Z values:

```python
mask1 = m_tl & m_bl & m_tr    # triangle 1 validity
mask2 = m_tr & m_bl & m_br    # triangle 2 validity
```

NaN vertices (from masked regions, division by zero, or out-of-domain values) are
silently dropped. The GPU never sees them — no `if` branches in the fragment shader.

---

## 8  3-D lighting — Blinn-Phong

**File:** `shaders3d.py::SURFACE_FRAGMENT_SHADER`

Three-point lighting is used to illuminate the surface from different angles.

#### The three lights

| Light | Direction | Role |
|-------|-----------|------|
| Key (L1) | `(1.2, 0.7, 2.0)` normalized | Main light, warm, strong |
| Fill (L2) | `(−0.9, 0.4, 0.7)` normalized | Softer left-side light |
| Back (L3) | `(0.0, −0.7, −0.5)` normalized | Rim light for depth |

#### Diffuse term (Lambert + two-sided)

```glsl
float d1 = max(dot( N, L1), 0.0) * 0.72    // front face, key light
         + max(dot(−N, L1), 0.0) * 0.14;   // back face, key light (weaker)
```

Standard Lambertian diffuse: `max(N·L, 0)` gives the cosine of the angle between
the surface normal and the light direction. The `−N` terms illuminate the underside
of the surface at reduced strength (back-face illumination).

#### Specular term (Blinn-Phong)

```glsl
vec3  H1   = normalize(L1 + V);             // half-vector between light and camera
float spec = pow(max(dot(N, H1), 0.0), 60.0) * 0.55;
```

Blinn-Phong replaces the reflection vector with the **half-vector** `H = (L + V)/|L+V|`.
`dot(N, H)` is the cosine of the angle between the normal and H; raising to power 60
creates a tight, glossy specular highlight.

#### Hemisphere ambient

```glsl
float hemi    = dot(N, vec3(0, 0, 1)) * 0.5 + 0.5;   // 0 = down, 1 = up
float ambient = mix(0.32, 0.46, hemi);
```

Instead of a constant ambient term, a **hemisphere ambient** is used: surfaces
facing upward (toward a sky) receive more light (0.46) than those facing downward
(0.32). This is more physically plausible than flat ambient.

#### Final combination

```glsl
float lighting = clamp(ambient + diffuse * (1.0 − ambient * 0.5), 0.30, 1.25);
vec3  finalCol = surfCol * lighting + vec3(spec * 0.50);
```

- `diffuse * (1.0 − ambient * 0.5)`: reduces diffuse in well-lit (high ambient)
  areas to prevent oversaturation.
- Specular is added *after* the multiply, keeping it white regardless of surface colour.
- The floor at 0.30 prevents any surface from going fully black.

---

## 9  Colormaps

**File:** `shaders3d.py::SURFACE_FRAGMENT_SHADER`

All colormaps are implemented as pure GLSL functions that map `t ∈ [0,1]` to RGB.
No texture lookup is needed.

| Index | Name | Strategy |
|-------|------|----------|
| 0 | Viridis | 3-stop piecewise linear: dark purple → teal → yellow |
| 1 | Hot | Sequential R → G → B ramps (`t×3`, `t×3−1`, `t×3−2`) |
| 2 | Cool | Cyan to magenta: `(t, 1−t, 1)` |
| 3 | Grayscale | `(t, t, t)` |
| 4 | HSV Height | HSV hue sweeps 0°→270° as height increases |
| 5 | HSV Angle | Hue from `atan2(y,x)` — useful for azimuthal features |
| 6 | Plasma | 4-stop purple → red → orange → yellow |
| 7 | Inferno | 4-stop near-black → dark purple → red → cream |
| 8 | Turbo | 4-stop vivid full-spectrum |

The **Hot** colormap is worth explaining explicitly:

```glsl
vec3 hot(float t) {
    return vec3(clamp(t * 3.0,       0, 1),   // R: full at t > 0.33
                clamp(t * 3.0 − 1.0, 0, 1),   // G: full at t > 0.67
                clamp(t * 3.0 − 2.0, 0, 1));  // B: full at t = 1.0
}
```

At `t=0.0`: black. At `t=0.33`: red. At `t=0.67`: yellow (R+G). At `t=1.0`: white.
This mimics blackbody radiation colours.

---

## 10  3-D overlay projection

**File:** `overlay3d.py::_project`

Axis labels, tick marks, and the bounding box are drawn with the Dear ImGui
draw-list API (CPU-side 2-D drawing). To place them correctly, each 3-D world
point is projected into screen space using the same MVP matrix as the surface:

```python
def _project(pt, MVP, viewport):
    x, y, z = pt
    v = MVP @ [x, y, z, 1.0]      # homogeneous clip coordinates
    ndc = v[:3] / v[3]             # perspective divide

    # Cull if outside depth range
    if ndc[2] > 1.0 or ndc[2] < −1.0:
        return None

    # NDC → screen pixels (note Y flip: NDC Y-up → screen Y-down)
    sx = viewport_x + (ndc[0] + 1) * 0.5 * viewport_w
    sy = viewport_y + (1 − (ndc[1] + 1) * 0.5) * viewport_h
    return sx, sy
```

The depth cull (`ndc[2] > 1` or `< −1`) discards points behind the near or far
plane, preventing labels from appearing when a corner of the bounding box wraps
behind the camera.

---

## 11  Data visualisation types

**File:** `state.py::_update_data_slot`, `plot_hist.py`

| Type | `plot_type` | Geometry | Shader |
|------|-------------|----------|--------|
| Equation | 0 | `DynamicLinePlot` (VBO of (x,y) pairs) | Solid or Effect |
| Scatter | 1 | Same VBO, drawn as `GL_POINTS` | Point shader |
| Line data | 2 | Same VBO, drawn as `GL_LINE_STRIP` | Solid shader |
| Histogram | 3 | Triangulated bars uploaded as `GL_TRIANGLES` | Fill shader |
| KDE | 4 | Fill area + line curve | Fill + Solid |
| Heatmap 2D | 5 | Triangulated grid with per-vertex colours | Colored-vertex |
| Violin | 6 | Fill + line + quartile lines (imgui draw-list) | Fill + Solid |

#### Histogram
Bins are computed with `np.histogram`. Each bar is a rectangle split into two
triangles. The bar height encodes count or density depending on the `norm` setting.

#### KDE (Kernel Density Estimate)
```python
from scipy.stats import gaussian_kde
kde_fn = gaussian_kde(data, bw_method='scott')
x      = np.linspace(lo, hi, 500)
y      = kde_fn(x)
```
Scott's rule selects bandwidth automatically: `h = n^(−1/5) · σ`. The KDE curve is
rendered as a solid line over a semi-transparent filled area.

#### Violin
A mirrored KDE. The positive half is the upper wing; the fill geometry includes
both `(x, +y)` and `(x, −y)` vertices to form the symmetric shape. Quartile lines
(Q1, median, Q3) are drawn via the imgui draw-list as vertical line segments.

---

## 12  File-by-file reference

| File | Responsibility |
|------|---------------|
| `main.py` | `hello_imgui` setup, run loop, CLI argument parsing, screenshot/clipboard dispatch |
| `state.py` | `AppState` — all mutable state, math generation, camera, interactions |
| `panel.py` | Dear ImGui control panel (all UI widgets) |
| `gui.py` | Axis labels, grid, overlay widgets drawn with imgui background draw-list |
| `renderer.py` | OpenGL draw calls for 2-D and 3-D; coordinates viewport/scissor transforms |
| `shaders.py` | GLSL source for all 2-D shaders (grid, solid, effect, fill, point, colored) |
| `shaders3d.py` | GLSL source for 3-D shaders (background gradient, surface Blinn-Phong, wireframe) |
| `overlay3d.py` | 3-D → 2-D projection for bounding box, axis ticks, and labels |
| `plot.py` | `DynamicLinePlot`, `GridGeometry`, `BackgroundQuad`, `ColoredMesh` — VAO/VBO wrappers |
| `plot3d.py` | `SurfacePlot3D` — mesh builder with central-difference normals |
| `plot3d_parametric.py` | `ParametricSurfacePlot` — same mesh builder for (u,v) parametric surfaces |
| `plot3d_curve.py` | `SpaceCurve3D` — tube geometry for space curves |
| `plot_hist.py` | CPU-side geometry builders for histogram bars, KDE fill, violin shapes |
| `plot_slot.py` | `PlotSlot` dataclass — one curve or dataset's state |
| `util.py` | `parse_to_numpy` (SymPy → NumPy with timeout), `format_label` |
| `session.py` | JSON save/restore of viewport, expressions, camera, and options |
| `data_loader.py` | CSV/TSV reader (detects delimiter, header, column types) |
