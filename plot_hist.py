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

    def draw(self):
        if self._vertex_count == 0:
            return
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)
