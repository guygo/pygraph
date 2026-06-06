"""GLSL shaders for the 3-D surface plot."""

SURFACE_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aT;
layout(location = 3) in float aAngle;   // atan2(y,x) in [-pi, pi]

uniform mat4 uMVP;
uniform mat4 uModel;

out vec3 vNormalWorld;
out vec3 vPosWorld;
out float vT;
out float vAngle;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vPosWorld    = worldPos.xyz;
    vNormalWorld = mat3(transpose(inverse(uModel))) * aNormal;
    vT           = aT;
    vAngle       = aAngle;
    gl_Position  = uMVP * vec4(aPos, 1.0);
}"""

SURFACE_FRAGMENT_SHADER = """#version 330 core
in vec3 vNormalWorld;
in vec3 vPosWorld;
in float vT;
in float vAngle;

out vec4 FragColor;

uniform vec3  uLightDir;
uniform float uAmbient;
uniform int   uColormap;   // 0=viridis, 1=hot, 2=cool, 3=grayscale, 4=hsv-height, 5=hsv-angle
uniform float uAlpha;

const float PI = 3.14159265358979;

// --- HSV to RGB ---
vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0);
    float c = v * s;
    float x = c * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
    float m = v - c;
    vec3 rgb;
    if      (h < 60.0)  rgb = vec3(c, x, 0);
    else if (h < 120.0) rgb = vec3(x, c, 0);
    else if (h < 180.0) rgb = vec3(0, c, x);
    else if (h < 240.0) rgb = vec3(0, x, c);
    else if (h < 300.0) rgb = vec3(x, 0, c);
    else                rgb = vec3(c, 0, x);
    return rgb + m;
}

// --- colour maps ---
vec3 viridis(float t) {
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.127, 0.567, 0.551);
    vec3 c2 = vec3(0.993, 0.906, 0.144);
    float s = clamp(t, 0.0, 1.0);
    if (s < 0.5) return mix(c0, c1, s * 2.0);
    return mix(c1, c2, (s - 0.5) * 2.0);
}

vec3 hot(float t) {
    float s = clamp(t, 0.0, 1.0);
    return vec3(clamp(s*3.0,0,1), clamp(s*3.0-1.0,0,1), clamp(s*3.0-2.0,0,1));
}

vec3 cool(float t) {
    float s = clamp(t, 0.0, 1.0);
    return vec3(s, 1.0 - s, 1.0);
}

vec3 grayscale(float t) { return vec3(clamp(t, 0.0, 1.0)); }

// HSV mapped to height (t)
vec3 hsv_height(float t) {
    float hue = (1.0 - clamp(t, 0.0, 1.0)) * 270.0; // blue=low, red=high
    return hsv2rgb(hue, 0.9, 0.85);
}

// HSV mapped to angle (like Desmos c1)
vec3 hsv_angle(float angle) {
    // angle in [-pi, pi] -> hue [0, 360]
    float hue = (angle / PI) * 180.0 + 180.0;  // map [-pi,pi] -> [0,360]
    return hsv2rgb(hue, 0.8, 0.7);
}

void main() {
    vec3 surfaceColor;
    if      (uColormap == 1) surfaceColor = hot(vT);
    else if (uColormap == 2) surfaceColor = cool(vT);
    else if (uColormap == 3) surfaceColor = grayscale(vT);
    else if (uColormap == 4) surfaceColor = hsv_height(vT);
    else if (uColormap == 5) surfaceColor = hsv_angle(vAngle);
    else                     surfaceColor = viridis(vT);

    vec3 N = normalize(vNormalWorld);
    float diff  = max(dot( N, normalize(uLightDir)), 0.0);
    float diff2 = max(dot(-N, normalize(uLightDir)), 0.0);
    float lighting = uAmbient + (1.0 - uAmbient) * max(diff, diff2 * 0.4);

    FragColor = vec4(surfaceColor * lighting, uAlpha);
}"""

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
