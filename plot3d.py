from OpenGL.GL import *
import numpy as np
import ctypes


class SurfacePlot3D:
    """
    Renders a z = f(x, y) surface as a triangle mesh with per-vertex normals,
    height-based colour (t), and angle-based colour (a = atan2(y,x)).

    Vertex layout (8 floats per vertex, stride = 32 bytes):
      0: x, 1: y, 2: z          (location 0, offset  0)
      3: nx, 4: ny, 5: nz       (location 1, offset 12)
      6: t  (height [0,1])      (location 2, offset 24)
      7: a  (atan2 [-pi, pi])   (location 3, offset 28)
    """

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.index_count = 0

    def update_data(self, x_data: np.ndarray, y_data: np.ndarray,
                    z_data: np.ndarray, circular_mask: bool = False,
                    max_r: float = None):
        """
        x_data, y_data  – 1-D arrays of length N / M
        z_data          – 2-D array of shape (M, N)  (rows=y, cols=x)
        circular_mask   – if True, vertices outside max_r are set to NaN and
                          triangles that include any NaN vertex are culled.
        max_r           – clip radius; defaults to min(|x_max|, |y_max|)
        """
        N = len(x_data)
        M = len(y_data)

        # ---- build 2-D grids ------------------------------------------------
        X, Y = np.meshgrid(x_data, y_data)   # (M, N)
        R    = np.sqrt(X**2 + Y**2)           # radius
        A    = np.arctan2(Y, X)               # angle in [-pi, pi]

        z_safe = np.where(np.isfinite(z_data), z_data, np.nan).astype(np.float32)

        # Circular domain masking
        if circular_mask:
            if max_r is None:
                max_r = min(abs(x_data[-1]), abs(y_data[-1]))
            z_safe[R > max_r] = np.nan

        # Height normalisation (ignoring NaN)
        z_min  = float(np.nanmin(z_safe)) if np.any(np.isfinite(z_safe)) else 0.0
        z_max  = float(np.nanmax(z_safe)) if np.any(np.isfinite(z_safe)) else 1.0
        z_range = max(z_max - z_min, 1e-9)

        # Replace NaN with z_min for normal computation (culled in index step)
        z_fill = np.where(np.isfinite(z_safe), z_safe, z_min).astype(np.float32)

        # ---- vectorised central-difference normals ---------------------------
        dx = (x_data[-1] - x_data[0]) / max(N - 1, 1)
        dy = (y_data[-1] - y_data[0]) / max(M - 1, 1)

        # Pad edges by replication for simple boundary treatment
        zp = np.pad(z_fill, 1, mode='edge')          # (M+2, N+2)
        dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * dx)   # (M, N)
        dzdy = (zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2.0 * dy)   # (M, N)

        nx = -dzdx
        ny = -dzdy
        nz = np.ones((M, N), dtype=np.float32)
        length = np.sqrt(nx**2 + ny**2 + nz**2)
        length = np.maximum(length, 1e-9)
        nx /= length;  ny /= length;  nz /= length

        # ---- height attribute t ----------------------------------------------
        t = (z_fill - z_min) / z_range   # (M, N), always finite

        # ---- pack vertex buffer (8 floats per vertex) -----------------------
        # shape: (M, N, 8)
        verts = np.stack([
            X.astype(np.float32),
            Y.astype(np.float32),
            z_fill,
            nx.astype(np.float32),
            ny.astype(np.float32),
            nz.astype(np.float32),
            t.astype(np.float32),
            A.astype(np.float32),
        ], axis=-1)                       # (M, N, 8)

        # ---- build index array (vectorised, cull masked vertices) ----------
        valid = np.isfinite(z_safe)      # (M, N)
        j_idx, i_idx = np.mgrid[0:M-1, 0:N-1]
        tl = (j_idx * N + i_idx).ravel()
        tr = tl + 1
        bl = ((j_idx + 1) * N + i_idx).ravel()
        br = bl + 1

        m_tl = valid[j_idx, i_idx].ravel()
        m_tr = valid[j_idx, i_idx+1].ravel()
        m_bl = valid[j_idx+1, i_idx].ravel()
        m_br = valid[j_idx+1, i_idx+1].ravel()

        mask1 = m_tl & m_bl & m_tr
        mask2 = m_tr & m_bl & m_br

        tri1 = np.stack([tl[mask1], bl[mask1], tr[mask1]], axis=1)
        tri2 = np.stack([tr[mask2], bl[mask2], br[mask2]], axis=1)
        indices = np.concatenate([tri1, tri2]).flatten().astype(np.uint32)
        self.index_count = len(indices)

        # ---- upload ----------------------------------------------------------
        stride = 8 * 4   # 8 floats × 4 bytes = 32 bytes
        flat_verts = verts.reshape(-1, 8).astype(np.float32)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, flat_verts.nbytes, flat_verts, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

        # position  (location 0, offset 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
        glEnableVertexAttribArray(0)
        # normal    (location 1, offset 12)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # t height  (location 2, offset 24)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        # a angle   (location 3, offset 28)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(3)

        glBindVertexArray(0)

    def draw(self):
        if self.index_count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
