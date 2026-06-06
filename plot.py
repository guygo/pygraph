from OpenGL.GL import *
import numpy as np
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
