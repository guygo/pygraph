"""
Parametric surface GPU geometry: (u,v) -> (x(u,v), y(u,v), z(u,v))
Also used for surface-of-revolution: (u, f(u)*cos(v), f(u)*sin(v))
"""
from OpenGL.GL import *
import numpy as np
import ctypes


class ParametricSurfacePlot:
    """
    Vertex layout (8 floats, stride=32):
      0-2: position xyz       (location 0, offset 0)
      3-5: normal xyz         (location 1, offset 12)
      6:   t  height [0,1]    (location 2, offset 24)
      7:   a  atan2(y,x)      (location 3, offset 28)
    """

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.index_count = 0

    def update_data(self,
                    X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        """
        X, Y, Z: shape (M, N) — evaluated parametric surface grid.
        M = #rows (v direction), N = #cols (u direction).
        """
        M, N = X.shape

        # Replace non-finite with nan
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        Y = np.where(np.isfinite(Y), Y, np.nan).astype(np.float32)
        Z = np.where(np.isfinite(Z), Z, np.nan).astype(np.float32)

        # Height attribute t (based on Z)
        z_min  = float(np.nanmin(Z)) if np.any(np.isfinite(Z)) else 0.0
        z_max  = float(np.nanmax(Z)) if np.any(np.isfinite(Z)) else 1.0
        z_range = max(z_max - z_min, 1e-9)
        T = ((Z - z_min) / z_range).astype(np.float32)

        # Angle attribute a
        A = np.arctan2(Y, X).astype(np.float32)

        # ---- vectorised normals via finite differences ----------------------
        # Pad for edge handling
        Xp = np.pad(np.nan_to_num(X), 1, mode='edge')
        Yp = np.pad(np.nan_to_num(Y), 1, mode='edge')
        Zp = np.pad(np.nan_to_num(Z, nan=z_min), 1, mode='edge')

        # du: partial in column direction
        du_x = Xp[1:-1, 2:] - Xp[1:-1, :-2]
        du_y = Yp[1:-1, 2:] - Yp[1:-1, :-2]
        du_z = Zp[1:-1, 2:] - Zp[1:-1, :-2]

        # dv: partial in row direction
        dv_x = Xp[2:, 1:-1] - Xp[:-2, 1:-1]
        dv_y = Yp[2:, 1:-1] - Yp[:-2, 1:-1]
        dv_z = Zp[2:, 1:-1] - Zp[:-2, 1:-1]

        # Cross product du × dv
        nx = du_y * dv_z - du_z * dv_y
        ny = du_z * dv_x - du_x * dv_z
        nz = du_x * dv_y - du_y * dv_x
        length = np.sqrt(nx**2 + ny**2 + nz**2)
        length = np.maximum(length, 1e-9)
        nx /= length; ny /= length; nz /= length

        # ---- pack vertices (M*N, 8) -----------------------------------------
        Xf = np.nan_to_num(X).astype(np.float32)
        Yf = np.nan_to_num(Y).astype(np.float32)
        Zf = np.nan_to_num(Z, nan=z_min).astype(np.float32)
        Tf = np.nan_to_num(T).astype(np.float32)
        Af = np.nan_to_num(A).astype(np.float32)

        verts = np.stack([Xf, Yf, Zf,
                          nx.astype(np.float32),
                          ny.astype(np.float32),
                          nz.astype(np.float32),
                          Tf, Af], axis=-1).reshape(-1, 8)

        # ---- index array (vectorised, cull degenerate tris) ----------------
        valid = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
        j_idx, i_idx = np.mgrid[0:M-1, 0:N-1]   # (M-1, N-1)
        tl = (j_idx * N + i_idx).ravel()
        tr = tl + 1
        bl = ((j_idx + 1) * N + i_idx).ravel()
        br = bl + 1

        m_tl = valid[j_idx, i_idx].ravel()
        m_tr = valid[j_idx, i_idx+1].ravel()
        m_bl = valid[j_idx+1, i_idx].ravel()
        m_br = valid[j_idx+1, i_idx+1].ravel()

        mask1 = m_tl & m_bl & m_tr   # lower-left triangle
        mask2 = m_tr & m_bl & m_br   # upper-right triangle

        tri1 = np.stack([tl[mask1], bl[mask1], tr[mask1]], axis=1)
        tri2 = np.stack([tr[mask2], bl[mask2], br[mask2]], axis=1)
        indices = np.concatenate([tri1, tri2]).flatten().astype(np.uint32)
        self.index_count = len(indices)

        stride = 8 * 4
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(3)
        glBindVertexArray(0)

    def draw(self):
        if self.index_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
