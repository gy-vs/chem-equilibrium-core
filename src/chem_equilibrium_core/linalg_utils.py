"""Small linear-algebra helpers shared by model acceptance and the solver."""

from __future__ import annotations

import numpy as np


def nullspace(matrix: np.ndarray, tol: float) -> np.ndarray:
    """Return an orthonormal basis of the right nullspace of ``matrix``.

    Rows of the result satisfy ``basis @ matrix.T == 0``.
    """
    if matrix.size == 0:
        # Nullspace of an empty row map is the whole space of the columns.
        n = matrix.shape[1]
        return np.eye(n) if n else np.zeros((0, 0))
    _, s, vh = np.linalg.svd(matrix)
    r = int(np.sum(s > tol * max(1.0, float(s[0])) if s.size else 0.0))
    return vh[r:].copy()


def rank(matrix: np.ndarray, tol: float) -> int:
    """Numerical rank using the same scale-relative SVD convention."""
    if matrix.size == 0:
        return 0
    s = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, float(s[0]))))
