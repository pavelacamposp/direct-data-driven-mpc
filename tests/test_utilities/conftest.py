from typing import Any

import numpy as np
import pytest


@pytest.fixture
def test_linear_system() -> Any:
    A = np.array([[0.2, 0.5, 0.2], [0, -0.2, 0.2], [0.1, -0.5, 0.1]])
    B = np.array([0, 1, 2]).reshape(-1, 1)
    C = np.array([[1, 2, 3], [4, 5, 6]])
    D = np.array([0, 1]).reshape(-1, 1)
    t = 3

    # Precomputed observability matrix
    # Note: The test system is observable to ensure correct
    # initial state estimation testing
    Ot = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [0.5, -1.4, 0.9],
            [1.4, -2, 2.4],
            [0.19, 0.08, -0.09],
            [0.52, -0.1, 0.12],
        ]
    )

    # Precomputed Toeplitz matrix
    Tt = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [8, 0, 0],
            [17, 1, 0],
            [0.4, 8, 0],
            [2.8, 17, 1],
        ]
    )

    return A, B, C, D, t, Ot, Tt


@pytest.fixture
def dummy_plot_data() -> tuple[np.ndarray, ...]:
    T = 50
    m = 2
    p = 2
    u_k = np.zeros((T, m))
    y_k = np.ones((T, p))
    u_s = np.ones((m, 1))
    y_s = np.ones((p, 1))

    return u_k, y_k, u_s, y_s
