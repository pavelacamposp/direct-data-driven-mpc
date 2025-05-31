from typing import Any

import numpy as np

from direct_data_driven_mpc.utilities.initial_state_estimation import (
    calculate_equilibrium_input_from_output,
    calculate_equilibrium_output_from_input,
    estimate_initial_state,
    observability_matrix,
    toeplitz_input_output_matrix,
)


def test_observability_matrix(test_linear_system: Any) -> None:
    # Define test parameters
    A, _, C, _, _, Ot, _ = test_linear_system
    expected_observability = Ot

    # Construct observability matrix
    obs_matrix = observability_matrix(A, C)

    # Verify the resulting observability matrix has correct shape and values
    assert obs_matrix.shape == expected_observability.shape
    np.testing.assert_allclose(obs_matrix, expected_observability)


def test_toeplitz_input_output_matrix(test_linear_system: Any) -> None:
    # Define test parameters
    A, B, C, D, t, _, Tt = test_linear_system
    expected_toeplitz = Tt

    # Construct Toeplitz matrix
    toeplitz_matrix = toeplitz_input_output_matrix(A, B, C, D, t)

    # Verify the resulting Toeplitz matrix has correct shape and values
    assert toeplitz_matrix.shape == expected_toeplitz.shape
    np.testing.assert_allclose(toeplitz_matrix, expected_toeplitz)


def test_estimate_initial_state(test_linear_system: Any) -> None:
    # Define test parameters
    A, B, C, D, _, Ot, Tt = test_linear_system
    x0_true = np.array([0.1, 1.5, -0.5])  # Known initial state
    np_random = np.random.default_rng(0)

    # Simulate linear system
    n_steps = A.shape[0]
    u_k = np_random.uniform(-1.0, 1.0, (n_steps, B.shape[1]))
    y_k = np.zeros((n_steps, C.shape[0]))
    x = x0_true

    for k in range(n_steps):
        y_k[k, :] = C @ x + D @ u_k[k, :]
        x = A @ x + B @ u_k[k, :]

    # Calculate estimated initial state
    x0_estimate = estimate_initial_state(
        Ot=Ot, Tt=Tt, U=u_k.flatten(), Y=y_k.flatten()
    )

    # Verify that the estimated value is close to the simulated one
    np.testing.assert_allclose(x0_estimate, x0_true)


def test_calculate_equilibrium_output_from_input(
    test_linear_system: Any,
) -> None:
    # Define test parameters
    A, B, C, D, _, _, _ = test_linear_system
    u_eq = np.array([0.5])  # Equilibrium input

    # Simulate linear system for a large number of steps to reach equilibrium
    n_steps = 15
    u_k = np.tile(u_eq, n_steps).reshape(-1, 1)
    y_k = np.zeros((n_steps, C.shape[0]))
    x = np.zeros(A.shape[0])

    for k in range(n_steps):
        y_k[k, :] = C @ x + D @ u_k[k, :]
        x = A @ x + B @ u_k[k, :]

    y_eq_sim = y_k[-1, :]

    # Calculate equilibrium output
    y_eq = calculate_equilibrium_output_from_input(A, B, C, D, u_eq)

    # Verify that the estimated value is close to the simulated one
    np.testing.assert_allclose(y_eq, y_eq_sim)


def test_calculate_equilibrium_input_from_output(
    test_linear_system: Any,
) -> None:
    # Define test parameters
    A, B, C, D, _, _, _ = test_linear_system
    expected_u_eq = np.array([0.5])  # Expected equilibrium input

    # Simulate linear system for a large number of steps to reach equilibrium
    n_steps = 15
    u_k = np.tile(expected_u_eq, n_steps).reshape(-1, 1)
    y_k = np.zeros((n_steps, C.shape[0]))
    x = np.zeros(A.shape[0])

    for k in range(n_steps):
        y_k[k, :] = C @ x + D @ u_k[k, :]
        x = A @ x + B @ u_k[k, :]

    y_eq_sim = y_k[-1, :]

    # Calculate equilibrium input
    u_eq = calculate_equilibrium_input_from_output(A, B, C, D, y_eq_sim)

    # Verify that the estimated value is close to the simulated one
    np.testing.assert_allclose(u_eq, expected_u_eq)
