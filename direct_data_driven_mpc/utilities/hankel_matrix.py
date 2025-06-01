import numpy as np
from numpy.lib.stride_tricks import as_strided


def hankel_matrix(X: np.ndarray, L: int) -> np.ndarray:
    """
    Construct a Hankel matrix from the input data matrix `X` with a window
    length `L`. The matrix `X` consists of a sequence of `N` elements, each of
    length `n`.

    Args:
        X (np.ndarray): Input data matrix of shape (N, n), where N is the
            number of elements, and n is the length of each element.
        L (int): Window length for the Hankel matrix.

    Returns:
        np.ndarray: A Hankel matrix of shape (L * n, N - L + 1), where each
            column represents a flattened window of length L sliding over the
            N data elements.

    Raises:
        ValueError: If the number of elements N is less than the window length
            L, indicating that the window length exceeds the available data
            length.

    Examples:
        >>> import numpy as np
        >>> N = 4  # Data length
        >>> L = 2  # Hankel matrix window length
        >>> n = 2  # Data vector length
        >>> rng = np.random.default_rng(0)  # RNG for reproducibility
        >>> u_d = rng.uniform(-1, 1, (N, n))  # Generate data matrix
        >>> print(hankel_matrix(u_d, L))
        [[ 0.27392337 -0.91805295  0.62654048]
         [-0.46042657 -0.96694473  0.82551115]
         [-0.91805295  0.62654048  0.21327155]
         [-0.96694473  0.82551115  0.45899312]]
    """
    # Get data matrix shape
    N, n = X.shape

    # Validate input dimensions
    if N < L:
        raise ValueError("N must be greater than or equal to L.")

    X = X.ravel()  # Transform X into a 1-D array

    # Construct Hankel matrix striding on the data array
    out_shp = L * n, N - L + 1
    n_row = X.strides[0]  # Move 1-by-1 element in rows
    n_col = X.strides[0] * n  # Move n-by-n elements in columns

    return as_strided(X, shape=out_shp, strides=(n_row, n_col)).copy()


def evaluate_persistent_excitation(
    X: np.ndarray, order: int
) -> tuple[int, bool]:
    """
    Evaluate whether a data sequence `X` is persistently exciting of a given
    order based on the rank of its Hankel matrix. The matrix `X` consists of a
    sequence of `N` elements, each of length `n`.

    This is determined by checking if the rank of the Hankel matrix
    constructed from `X` is greater than or equal to the expected rank
    `n * order`.

    Args:
        X (np.ndarray): Input data matrix of shape (N, n), where N is the
            number of elements, and n is the length of each element.
        order (int): The order of persistent excitation to evaluate.

    Returns:
        tuple[int, bool]: A tuple containing the rank of the Hankel matrix and
            a boolean indicating whether `X` is persistently exciting of the
            given order.
    """
    # Get data sequence element length
    n = X.shape[1]
    # Construct Hankel matrix from X
    H_order = hankel_matrix(X, order)
    # Calculate the Hankel matrix order
    rank_H_order = np.linalg.matrix_rank(H_order)

    # Check if X is persistently exciting of the given order
    pers_exciting = rank_H_order >= n * order

    return rank_H_order, pers_exciting
