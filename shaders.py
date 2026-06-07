GRID_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
out vec2 vGraphPos;
void main() {
    vGraphPos = aPos;
    vec2 normalized = (aPos - uMinBounds) / (uMaxBounds - uMinBounds);
    vec2 ndc = normalized * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}"""

GRID_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
in vec2 vGraphPos;
uniform vec2 uGridSpacing;
void main() {
    float gridThickness = 0.015;
    vec2 grid = fract(vGraphPos / uGridSpacing);
    if (grid.x < gridThickness || grid.x > (1.0 - gridThickness) ||
        grid.y < gridThickness || grid.y > (1.0 - gridThickness)) {
        FragColor = vec4(0.8, 0.8, 0.8, 0.5);
    } else {
        discard;
    }
}"""

# ── Solid line shader (anti-aliased smooth edges via gDist) ──────────────────

SOLID_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
uniform int  uLogX;
uniform int  uLogY;
void main() {
    float nx = uLogX > 0
        ? (log(max(aPos.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
          / (log(max(uMaxBounds.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
        : (aPos.x - uMinBounds.x) / (uMaxBounds.x - uMinBounds.x);
    float ny = uLogY > 0
        ? (log(max(aPos.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
          / (log(max(uMaxBounds.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
        : (aPos.y - uMinBounds.y) / (uMaxBounds.y - uMinBounds.y);
    gl_Position = vec4(nx * 2.0 - 1.0, ny * 2.0 - 1.0, 0.0, 1.0);
}"""

SOLID_GEOMETRY_SHADER = """#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform float uLineWidth;
uniform vec2  uViewport;
out float gDist;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 dir  = normalize((p1.xy - p0.xy) * uViewport);
    vec2 perp = vec2(-dir.y, dir.x) / uViewport * uLineWidth;

    gDist = -1.0; gl_Position = vec4(p0.xy - perp, p0.zw); EmitVertex();
    gDist = +1.0; gl_Position = vec4(p0.xy + perp, p0.zw); EmitVertex();
    gDist = -1.0; gl_Position = vec4(p1.xy - perp, p1.zw); EmitVertex();
    gDist = +1.0; gl_Position = vec4(p1.xy + perp, p1.zw); EmitVertex();
    EndPrimitive();
}"""

SOLID_FRAGMENT_SHADER = """#version 330 core
in  float gDist;
out vec4  FragColor;
uniform vec3 uColor;
void main() {
    float d     = abs(gDist);
    float alpha = 1.0 - smoothstep(0.60, 1.0, d);
    if (alpha < 0.01) discard;
    FragColor = vec4(uColor, alpha);
}"""

# ── Effect line shader (neon / plasma / electric) ────────────────────────────

EFFECT_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
uniform int  uLogX;
uniform int  uLogY;
out vec2 vNorm;
out vec2 vPos;
void main() {
    float nx = uLogX > 0
        ? (log(max(aPos.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
          / (log(max(uMaxBounds.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
        : (aPos.x - uMinBounds.x) / (uMaxBounds.x - uMinBounds.x);
    float ny = uLogY > 0
        ? (log(max(aPos.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
          / (log(max(uMaxBounds.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
        : (aPos.y - uMinBounds.y) / (uMaxBounds.y - uMinBounds.y);
    vNorm = vec2(nx, ny);
    vPos  = aPos;
    gl_Position = vec4(nx * 2.0 - 1.0, ny * 2.0 - 1.0, 0.0, 1.0);
}"""

EFFECT_GEOMETRY_SHADER = """#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
in  vec2 vNorm[];
in  vec2 vPos[];
uniform float uLineWidth;
uniform vec2  uViewport;
out float gDist;
out vec2  gNorm;
out vec2  gPos;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 dir  = normalize((p1.xy - p0.xy) * uViewport);
    vec2 perp = vec2(-dir.y, dir.x) / uViewport * uLineWidth;

    gDist = -1.0; gNorm = vNorm[0]; gPos = vPos[0];
    gl_Position = vec4(p0.xy - perp, p0.zw); EmitVertex();
    gDist = +1.0; gNorm = vNorm[0]; gPos = vPos[0];
    gl_Position = vec4(p0.xy + perp, p0.zw); EmitVertex();
    gDist = -1.0; gNorm = vNorm[1]; gPos = vPos[1];
    gl_Position = vec4(p1.xy - perp, p1.zw); EmitVertex();
    gDist = +1.0; gNorm = vNorm[1]; gPos = vPos[1];
    gl_Position = vec4(p1.xy + perp, p1.zw); EmitVertex();
    EndPrimitive();
}"""

EFFECT_FRAGMENT_SHADER = """#version 330 core
in  float gDist;
in  vec2  gNorm;
in  vec2  gPos;
out vec4  FragColor;
uniform vec3  uColor;
uniform int   uEffectMode;
uniform float uTime;

vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0);
    float c = v * s;
    float x = c * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
    float m = v - c;
    vec3 rgb;
    if      (h < 60.0)  rgb = vec3(c, x, 0.0);
    else if (h < 120.0) rgb = vec3(x, c, 0.0);
    else if (h < 180.0) rgb = vec3(0.0, c, x);
    else if (h < 240.0) rgb = vec3(0.0, x, c);
    else if (h < 300.0) rgb = vec3(x, 0.0, c);
    else                rgb = vec3(c, 0.0, x);
    return rgb + m;
}

void main() {
    float d = abs(gDist);

    if (uEffectMode == 1) {
        // ── Neon Glow Rainbow ─────────────────────────────────────────────
        float hue  = mod(gNorm.x * 360.0 - uTime * 45.0, 360.0);
        vec3  col  = hsv2rgb(hue, 1.0, 1.0);
        float core = 1.0 - smoothstep(0.0,  0.30, d);
        float glow = 1.0 - smoothstep(0.30, 1.0,  d);
        float alpha = clamp(core + glow * 0.45, 0.0, 1.0);
        if (alpha < 0.01) discard;
        // White-hot core blending into saturated colour
        vec3 lit = mix(col, vec3(1.0), core * 0.55) * (core * 1.4 + glow * 0.5);
        FragColor = vec4(clamp(lit, 0.0, 1.0), alpha);

    } else if (uEffectMode == 2) {
        // ── Plasma ────────────────────────────────────────────────────────
        float px  = gPos.x * 1.2 + uTime * 0.35;
        float py  = gPos.y * 1.2 + uTime * 0.20;
        float v1  = sin(px * 1.5 + uTime * 0.5);
        float v2  = sin(py * 1.5 - px * 0.5 + uTime * 0.3);
        float v3  = sin(sqrt(max(px * px + py * py, 0.0)) * 1.8 - uTime * 0.7);
        float plasma = (v1 + v2 + v3) / 3.0;          // -1..+1
        float hue    = mod(plasma * 160.0 + uTime * 25.0 + 200.0, 360.0);
        vec3  col    = hsv2rgb(hue, 0.90, 0.95);
        float edge   = 1.0 - smoothstep(0.65, 1.0, d);
        if (edge < 0.01) discard;
        FragColor = vec4(col, edge);

    } else if (uEffectMode == 3) {
        // ── Electric Blue-White ───────────────────────────────────────────
        float flicker = sin(gNorm.x * 28.0 - uTime * 9.0) * 0.5 + 0.5;
        float hue     = 195.0 + flicker * 45.0;
        float sat     = 0.65 + flicker * 0.35;
        vec3  col     = hsv2rgb(hue, sat, 1.0);
        float core    = exp(-d * d * 5.5);
        float edge    = 1.0 - smoothstep(0.50, 1.0, d);
        float alpha   = clamp(core * 1.3 + edge * 0.25, 0.0, 1.0);
        if (alpha < 0.01) discard;
        vec3 lit = mix(col, vec3(1.0), core * 0.65);  // white-hot core
        FragColor = vec4(clamp(lit, 0.0, 1.0), alpha);

    } else {
        discard;
    }
}"""

# ── Filled-polygon shader (histogram bars, no geometry shader) ────────────────

FILL_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
void main() {
    vec2 norm = (aPos - uMinBounds) / (uMaxBounds - uMinBounds);
    gl_Position = vec4(norm * 2.0 - 1.0, 0.0, 1.0);
}"""

FILL_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
uniform vec3  uColor;
uniform float uAlpha;
void main() {
    FragColor = vec4(uColor, uAlpha);
}"""

# ── Anti-aliased point shader (scatter plot) ──────────────────────────────────

POINT_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2  uMinBounds;
uniform vec2  uMaxBounds;
uniform float uPointSize;
uniform int   uLogX;
uniform int   uLogY;
void main() {
    float nx = uLogX > 0
        ? (log(max(aPos.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
          / (log(max(uMaxBounds.x, 1e-9)) - log(max(uMinBounds.x, 1e-9)))
        : (aPos.x - uMinBounds.x) / (uMaxBounds.x - uMinBounds.x);
    float ny = uLogY > 0
        ? (log(max(aPos.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
          / (log(max(uMaxBounds.y, 1e-9)) - log(max(uMinBounds.y, 1e-9)))
        : (aPos.y - uMinBounds.y) / (uMaxBounds.y - uMinBounds.y);
    gl_Position  = vec4(nx * 2.0 - 1.0, ny * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = uPointSize;
}"""

POINT_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    vec2  c     = gl_PointCoord - 0.5;
    float d     = length(c) * 2.0;
    float alpha = 1.0 - smoothstep(0.65, 1.0, d);
    if (alpha < 0.01) discard;
    FragColor = vec4(uColor, alpha);
}"""

# ── Colored-vertex shader (heatmap) ──────────────────────────────────────────

COLORED_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
out vec3 vColor;
void main() {
    vec2 norm = (aPos - uMinBounds) / (uMaxBounds - uMinBounds);
    gl_Position = vec4(norm * 2.0 - 1.0, 0.0, 1.0);
    vColor = aColor;
}"""

COLORED_FRAGMENT_SHADER = """#version 330 core
in  vec3 vColor;
uniform float uAlpha;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, uAlpha);
}"""
