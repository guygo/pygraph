"""3D parametric space curve rendered as a smooth tube mesh.

The curve (t) → (x(t), y(t), z(t)) is extruded into a tube using a
parallel-transport frame (Rodrigues rotation), which avoids the twisting
artefacts of the classical Frenet frame near straight segments.

Vertex layout (8 floats, stride = 32 bytes) — identical to SurfacePlot3D
so the same surface shader handles lighting and colormaps:
  0-2: position xyz       (location 0, offset  0)
  3-5: normal xyz         (location 1, offset 12)
  6:   t  param [0, 1]    (location 2, offset 24)
  7:   angle atan2(y, x)  (location 3, offset 28)
"""
from OpenGL.GL import *
import numpy as np
import ctypes


class SpaceCurve3D:

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.index_count = 0

    def update_data(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    tube_radius: float = 0.05, tube_sides: int = 12):
        """
        X, Y, Z : shape (N,) — curve samples evaluated at uniform t values.
        tube_radius : radius of the tube in scene units.
        tube_sides  : number of facets around the tube circumference.
        """
        N      = len(X)
        points = np.stack([X, Y, Z], axis=1).astype(np.float64)   # (N, 3)

        # ── tangents (central differences, normalised) ────────────────────
        T = np.empty_like(points)
        T[1:-1] = points[2:] - points[:-2]
        T[0]    = points[1]  - points[0]
        T[-1]   = points[-1] - points[-2]
        T_len   = np.linalg.norm(T, axis=1, keepdims=True)
        T /= np.maximum(T_len, 1e-12)

        # ── parallel-transport frame (Rodrigues) ──────────────────────────
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(T[0], ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        N0 = np.cross(T[0], ref)
        N0 /= np.linalg.norm(N0)

        normals = np.empty_like(points)
        normals[0] = N0
        for i in range(1, N):
            t_prev = T[i - 1]
            t_curr = T[i]
            rot_ax = np.cross(t_prev, t_curr)
            ax_len = np.linalg.norm(rot_ax)
            if ax_len < 1e-12:
                normals[i] = normals[i - 1]
            else:
                rot_ax /= ax_len
                theta   = float(np.arccos(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)))
                n       = normals[i - 1]
                rotated = (n * np.cos(theta)
                           + np.cross(rot_ax, n) * np.sin(theta)
                           + rot_ax * np.dot(rot_ax, n) * (1.0 - np.cos(theta)))
                rotated -= np.dot(rotated, t_curr) * t_curr   # re-orthogonalise
                r_len    = np.linalg.norm(rotated)
                normals[i] = rotated / r_len if r_len > 1e-12 else normals[i - 1]

        binormals = np.cross(T, normals)
        bn_len    = np.linalg.norm(binormals, axis=1, keepdims=True)
        binormals /= np.maximum(bn_len, 1e-12)

        # ── tube vertices (vectorised) ─────────────────────────────────────
        S      = tube_sides
        angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
        cos_a  = np.cos(angles)   # (S,)
        sin_a  = np.sin(angles)   # (S,)

        # outward ring directions:  (N, S, 3)
        outward = (cos_a[np.newaxis, :, np.newaxis] * normals[:, np.newaxis, :]
                 + sin_a[np.newaxis, :, np.newaxis] * binormals[:, np.newaxis, :])

        positions = (points[:, np.newaxis, :] + tube_radius * outward).astype(np.float32)
        nrm_arr   = outward.astype(np.float32)

        t_param = np.linspace(0.0, 1.0, N, dtype=np.float32)
        t_arr   = np.broadcast_to(t_param[:, np.newaxis, np.newaxis],
                                   (N, S, 1)).copy()

        ang_arr = np.arctan2(positions[:, :, 1:2],
                             positions[:, :, 0:1]).astype(np.float32)

        verts = np.concatenate([positions, nrm_arr, t_arr, ang_arr],
                               axis=-1).reshape(-1, 8).astype(np.float32)

        # ── indices (two triangles per quad between adjacent rings) ────────
        i_idx  = np.arange(N - 1)[:, np.newaxis]
        s_idx  = np.arange(S)[np.newaxis, :]
        s_next = (s_idx + 1) % S

        v0 = (i_idx * S + s_idx).ravel()
        v1 = ((i_idx + 1) * S + s_idx).ravel()
        v2 = (i_idx * S + s_next).ravel()
        v3 = ((i_idx + 1) * S + s_next).ravel()

        indices = np.concatenate([
            np.stack([v0, v1, v2], axis=1),
            np.stack([v1, v3, v2], axis=1),
        ]).flatten().astype(np.uint32)
        self.index_count = len(indices)

        # ── GPU upload ────────────────────────────────────────────────────
        stride = 32   # 8 × 4 bytes
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
