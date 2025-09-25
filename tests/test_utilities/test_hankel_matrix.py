import numpy as np
import pytest

from direct_data_driven_mpc.utilities import (
    evaluate_persistent_excitation,
    hankel_matrix,
)


def test_hankel_matrix() -> None:
    # Define test parameters
    X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    L = 2
    expected_hankel = np.array(
        [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [3, 5, 7, 9],
        ]
    )

    # Construct hankel matrix
    hankel = hankel_matrix(X, L)

    # Verify the resulting hankel matrix has correct shape and values
    assert hankel.shape == expected_hankel.shape
    np.testing.assert_array_equal(hankel, expected_hankel)


@pytest.mark.parametrize("pers_exc", [True, False])
def test_evaluate_persistent_excitation(pers_exc: bool) -> None:
    # Define test parameters
    N = 5
    n = 2
    expected_order = 2

    # Generate input based on the desired persistent excitation condition
    if pers_exc:
        X = np.random.uniform(-1, 1, (N, n))
    else:
        X = np.ones((N, n))

    # Evaluate persistent excitation
    hankel_rank, is_pers_exc = evaluate_persistent_excitation(
        X, expected_order
    )

    # Verify Hankel matrix rank and persistent excitation status
    if pers_exc:
        assert hankel_rank >= expected_order
        assert is_pers_exc
    else:
        assert hankel_rank < expected_order
        assert not is_pers_exc
