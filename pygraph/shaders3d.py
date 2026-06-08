"""GLSL shaders for the 3-D surface plot (Blinn-Phong lighting, extended colormaps)."""

# ── Dark gradient background ──────────────────────────────────────────────────

BG_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 aPos;
out float vY;
void main() {
    vY = aPos.y;
    gl_Position = vec4(aPos, 0.9999, 1.0);
}"""

BG_FRAGMENT_SHADER = """#version 330 core
in  float vY;
out vec4  FragColor;
void main() {
    float t   = vY * 0.5 + 0.5;
    vec3  bot = vec3(0.970, 0.972, 0.980);   // off-white
    vec3  top = vec3(0.920, 0.932, 0.970);   // very light blue-white
    FragColor = vec4(mix(bot, top, t * t), 1.0);
}"""

# ── Surface vertex shader ─────────────────────────────────────────────────────

SURFACE_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in float aT;
layout(location = 3) in float aAngle;

uniform mat4 uMVP;
uniform mat4 uModel;

out vec3  vNormalWorld;
out vec3  vPosWorld;
out float vT;
out float vAngle;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vPosWorld     = worldPos.xyz;
    vNormalWorld  = mat3(transpose(inverse(uModel))) * aNormal;
    vT            = aT;
    vAngle        = aAngle;
    gl_Position   = uMVP * vec4(aPos, 1.0);
}"""

# ── Surface fragment shader (Blinn-Phong + rim + 9 colormaps) ─────────────────

SURFACE_FRAGMENT_SHADER = """#version 330 core
in vec3  vNormalWorld;
in vec3  vPosWorld;
in float vT;
in float vAngle;

out vec4 FragColor;

uniform vec3  uLightDir;
uniform vec3  uCamPos;
uniform float uAmbient;
uniform int   uColormap;
uniform float uAlpha;

const float PI = 3.14159265358979;

// ── colour helpers ───────────────────────────────────────────────────────────
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

// ── colormaps ─────────────────────────────────────────────────────────────────
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.127, 0.567, 0.551);
    vec3 c2 = vec3(0.993, 0.906, 0.144);
    return t < 0.5 ? mix(c0, c1, t * 2.0) : mix(c1, c2, (t - 0.5) * 2.0);
}

vec3 hot(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(clamp(t * 3.0, 0.0, 1.0),
                clamp(t * 3.0 - 1.0, 0.0, 1.0),
                clamp(t * 3.0 - 2.0, 0.0, 1.0));
}

vec3 cool(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(t, 1.0 - t, 1.0);
}

vec3 grayscale(float t) { return vec3(clamp(t, 0.0, 1.0)); }

vec3 hsv_height(float t) {
    float hue = (1.0 - clamp(t, 0.0, 1.0)) * 270.0;
    return hsv2rgb(hue, 0.90, 0.88);
}

vec3 hsv_angle(float angle) {
    float hue = (angle / PI) * 180.0 + 180.0;
    return hsv2rgb(hue, 0.80, 0.75);
}

// Plasma (perceptually uniform, vivid)
vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.050, 0.030, 0.527);
    vec3 c1 = vec3(0.557, 0.047, 0.659);
    vec3 c2 = vec3(0.867, 0.310, 0.200);
    vec3 c3 = vec3(0.940, 0.975, 0.131);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
}

// Inferno (dark-to-bright fire)
vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.014);
    vec3 c1 = vec3(0.341, 0.068, 0.430);
    vec3 c2 = vec3(0.787, 0.283, 0.172);
    vec3 c3 = vec3(0.988, 1.000, 0.644);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
}

// Turbo (vivid full-spectrum rainbow)
vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if      (t < 0.25) c = mix(vec3(0.19, 0.07, 0.23), vec3(0.13, 0.63, 0.87), t * 4.0);
    else if (t < 0.50) c = mix(vec3(0.13, 0.63, 0.87), vec3(0.27, 0.94, 0.37), (t - 0.25) * 4.0);
    else if (t < 0.75) c = mix(vec3(0.27, 0.94, 0.37), vec3(0.95, 0.60, 0.07), (t - 0.50) * 4.0);
    else               c = mix(vec3(0.95, 0.60, 0.07), vec3(0.68, 0.01, 0.05), (t - 0.75) * 4.0);
    return c;
}

// ── three-point lighting ──────────────────────────────────────────────────────
void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(uCamPos - vPosWorld);

    // Key light  — upper front-right, warm and strong
    vec3 L1 = normalize(vec3( 1.2,  0.7,  2.0));
    // Fill light — left side, cooler and softer
    vec3 L2 = normalize(vec3(-0.9,  0.4,  0.7));
    // Back light — below-behind for depth separation on undersides
    vec3 L3 = normalize(vec3( 0.0, -0.7, -0.5));

    // Two-sided diffuse: front face weight + small back-face contribution
    float d1 = max(dot( N, L1), 0.0) * 0.72 + max(dot(-N, L1), 0.0) * 0.14;
    float d2 = max(dot( N, L2), 0.0) * 0.28 + max(dot(-N, L2), 0.0) * 0.07;
    float d3 = max(dot( N, L3), 0.0) * 0.10;
    float diffuse = d1 + d2 + d3;

    // Blinn-Phong specular on key light (both sides, different shininess)
    vec3  H1   = normalize(L1 + V);
    float spec = pow(max(dot( N, H1), 0.0), 60.0) * 0.55
               + pow(max(dot(-N, H1), 0.0), 60.0) * 0.12;

    // Hemisphere ambient: sky slightly brighter than ground
    float hemi    = dot(N, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5;
    float ambient = mix(0.32, 0.46, hemi);

    // Combine — floor at 0.30 so nothing goes fully dark
    float lighting = clamp(ambient + diffuse * (1.0 - ambient * 0.5), 0.30, 1.25);

    // Colormap lookup
    vec3 surfCol;
    if      (uColormap == 1) surfCol = hot(vT);
    else if (uColormap == 2) surfCol = cool(vT);
    else if (uColormap == 3) surfCol = grayscale(vT);
    else if (uColormap == 4) surfCol = hsv_height(vT);
    else if (uColormap == 5) surfCol = hsv_angle(vAngle);
    else if (uColormap == 6) surfCol = plasma(vT);
    else if (uColormap == 7) surfCol = inferno(vT);
    else if (uColormap == 8) surfCol = turbo(vT);
    else                     surfCol = viridis(vT);

    // Surface color × lighting, then additive specular (keeps colors vibrant)
    vec3 finalCol = surfCol * lighting + vec3(spec * 0.50);

    FragColor = vec4(clamp(finalCol, 0.0, 1.0), uAlpha);
}"""

# ── Wireframe shaders ─────────────────────────────────────────────────────────

WIRE_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}"""

WIRE_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
uniform vec4 uWireColor;
void main() {
    FragColor = uWireColor;
}"""
