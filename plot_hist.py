import numpy as np
from OpenGL.GL import *


class HistogramPlot:
    def __init__(self):
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._vertex_count = 0
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def update_data(self, bin_edges, bin_counts, gap=0.05):
        """bin_edges: (N+1,)  bin_counts: (N,)  gap: fraction of bar width left as space."""
        if len(bin_counts) == 0:
            self._vertex_count = 0
            return
        verts = []
        for i, cnt in enumerate(bin_counts):
            lo = float(bin_edges[i])
            hi = float(bin_edges[i + 1])
            hw = (hi - lo) * gap * 0.5   # half-gap each side
            l, r, h = lo + hw, hi - hw, float(cnt)
            # Two triangles = one bar
            verts += [l, 0.0,  r, 0.0,  l,  h,
                      r, 0.0,  r,  h,   l,  h]
        v = np.array(verts, dtype=np.float32)
        self._vertex_count = len(v) // 2
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_DYNAMIC_DRAW)

    def update_fill_strip(self, x_arr, y_arr):
        """Store a filled area under a curve as GL_TRIANGLE_STRIP (for KDE)."""
        n = len(x_arr)
        if n == 0:
            self._vertex_count = 0
            return
        # Interleave baseline and curve: (x,0),(x,y),(x+1,0),(x+1,y+1),...
        verts = np.empty((n * 2, 2), dtype=np.float32)
        verts[0::2, 0] = x_arr
        verts[0::2, 1] = 0.0
        verts[1::2, 0] = x_arr
        verts[1::2, 1] = y_arr
        self._vertex_count = n * 2
        self._draw_mode = GL_TRIANGLE_STRIP
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)

    def update_violin_fill(self, x_arr, y_arr):
        """Full violin shape (both mirrored halves) as GL_TRIANGLES."""
        n = len(x_arr)
        if n < 2:
            self._vertex_count = 0
            return
        verts = []
        for i in range(n - 1):
            xl, xr = float(x_arr[i]), float(x_arr[i + 1])
            yu, yv = float(y_arr[i]), float(y_arr[i + 1])
            # upper half
            verts += [xl, 0.0, xr, 0.0, xl, yu,
                      xr, 0.0, xr, yv,  xl, yu]
            # lower half (mirror)
            verts += [xl, 0.0, xr, 0.0, xl, -yu,
                      xr, 0.0, xr, -yv, xl, -yu]
        v = np.array(verts, dtype=np.float32)
        self._vertex_count = len(v) // 2
        self._draw_mode = GL_TRIANGLES
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_DYNAMIC_DRAW)

    def draw(self):
        if self._vertex_count == 0:
            return
        glBindVertexArray(self._vao)
        glDrawArrays(getattr(self, '_draw_mode', GL_TRIANGLES), 0, self._vertex_count)
        glBindVertexArray(0)
