import numpy as np

from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem,
)


def test_nonlinear_model_simulation() -> None:
    # Define dynamics function
    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.sin(x) + u

    # Define output function
    def h(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x**2 + u

    # Define test parameters
    n = 2
    m = 2
    p = 2
    eps_max = 0.0

    # Instantiate nonlinear model
    model = NonlinearSystem(f, h, n, m, p, eps_max)

    # Test single step simulation
    u = np.ones((m,))
    w = np.zeros((p,))
    expected_x = np.array([1, 1])
    expected_y = np.array([1, 1])

    y = model.simulate_step(u, w)

    np.testing.assert_allclose(model.x, expected_x)
    np.testing.assert_allclose(y, expected_y)

    # Test multiple steps simulation
    n_steps = 5
    U = np.ones((n_steps, m))
    W = np.zeros((n_steps, p))
    expected_x = np.array([1.93321866, 1.93321866])
    expected_Y = np.array(
        [
            [2, 2],
            [4.39101539, 4.39101539],
            [4.85568853, 4.85568853],
            [4.70117209, 4.70117209],
            [4.75709853, 4.75709853],
        ]
    )

    Y = model.simulate(U, W, n_steps)

    np.testing.assert_allclose(model.x, expected_x)
    np.testing.assert_allclose(Y, expected_Y)
