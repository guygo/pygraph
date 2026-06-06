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

SOLID_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
void main() {
    vec2 normalized = (aPos - uMinBounds) / (uMaxBounds - uMinBounds);
    vec2 ndc = normalized * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}"""

SOLID_GEOMETRY_SHADER = """#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform float uLineWidth;
uniform vec2  uViewport;   // (plot_width_px, plot_height_px)
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // Direction of the segment in screen space
    vec2 dir = normalize((p1.xy / p1.w - p0.xy / p0.w) * uViewport);
    // Perpendicular in screen space, then back to NDC
    vec2 perp = vec2(-dir.y, dir.x);
    vec2 offset = (perp / uViewport) * uLineWidth;

    gl_Position = vec4(p0.xy + vec2(-offset.x, -offset.y) * p0.w, p0.zw); EmitVertex();
    gl_Position = vec4(p0.xy + vec2( offset.x,  offset.y) * p0.w, p0.zw); EmitVertex();
    gl_Position = vec4(p1.xy + vec2(-offset.x, -offset.y) * p1.w, p1.zw); EmitVertex();
    gl_Position = vec4(p1.xy + vec2( offset.x,  offset.y) * p1.w, p1.zw); EmitVertex();
    EndPrimitive();
}"""

SOLID_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}"""

EFFECT_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uMinBounds;
uniform vec2 uMaxBounds;
out vec2 vTexCoord;
void main() {
    vec2 normalized = (aPos - uMinBounds) / (uMaxBounds - uMinBounds);
    vTexCoord = normalized;
    vec2 ndc = normalized * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}"""

EFFECT_GEOMETRY_SHADER = """#version 330 core
layout(lines) in;
layout(line_strip, max_vertices = 2) out;
in vec2 vTexCoord[];
out vec2 fTexCoord;
void main() {
    gl_Position = gl_in[0].gl_Position;
    fTexCoord = vTexCoord[0];
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    fTexCoord = vTexCoord[1];
    EmitVertex();
    EndPrimitive();
}"""

EFFECT_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor;
in vec2 fTexCoord;
uniform vec3 uColor;
uniform int uEffectMode;
uniform float uTime;
void main() {
    if (uEffectMode == 1) {
        float r = sin(fTexCoord.x * 5.0 + uTime) * 0.5 + 0.5;
        float g = sin(fTexCoord.x * 5.0 + uTime + 2.0) * 0.5 + 0.5;
        float b = sin(fTexCoord.x * 5.0 + uTime + 4.0) * 0.5 + 0.5;
        FragColor = vec4(r, g, b, 1.0);
    } else if (uEffectMode == 2) {
        float stripe = sin(fTexCoord.x * 50.0 - uTime * 5.0);
        if (stripe > 0.0) {
            FragColor = vec4(0.1, 0.1, 0.1, 1.0);
        } else {
            FragColor = vec4(0.7, 0.7, 0.7, 1.0);
        }
    } else {
        discard;
    }
}"""