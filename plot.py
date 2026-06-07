from OpenGL.GL import *
import numpy as np
import ctypes


class DynamicLinePlot:
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.point_count = 0

    def update_data(self, x_data, y_data):
        self.point_count = len(x_data)
        vertices = np.empty((self.point_count, 2), dtype=np.float32)
        vertices[:, 0] = x_data
        vertices[:, 1] = y_data
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self, draw_mode=GL_LINE_STRIP):
        if self.point_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(draw_mode, 0, self.point_count)
        glBindVertexArray(0)


class GridGeometry:
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 8 * 4, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def update_bounds(self, min_x, min_y, max_x, max_y):
        vertices = np.array([
            min_x, min_y,
            max_x, min_y,
            min_x, max_y,
            max_x, max_y,
        ], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


class BackgroundQuad:
    """Full-viewport clip-space quad for gradient backgrounds."""

    def __init__(self):
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


class ColoredMesh:
    """VAO/VBO for interleaved (x,y,r,g,b) per-vertex colored triangles."""
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.vertex_count = 0
        stride = 5 * 4  # 5 floats * 4 bytes
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def update_data(self, verts_5col):
        """verts_5col: float32 (N,5) array of [x,y,r,g,b]."""
        self.vertex_count = len(verts_5col)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts_5col.nbytes, verts_5col, GL_DYNAMIC_DRAW)

    def draw(self):
        if self.vertex_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)
