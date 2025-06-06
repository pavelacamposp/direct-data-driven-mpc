from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle

from direct_data_driven_mpc.utilities.visualization.plot_utilities import (
    HandlerInitMeasurementRect,
    create_input_output_figure,
    validate_data_dimensions,
)

matplotlib.use("Agg")  # Prevent GUI backend


def test_validate_data_dimensions_input_output(
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    u_k, y_k, u_s, y_s = dummy_plot_data

    # Verify `ValueError` is raised on dimension
    # mismatch between input-output arrays
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_data_dimensions(u_k=u_k[:-1], y_k=y_k, y_s=y_s)

    # Verify `ValueError` is raised on shape mismatch between setpoint arrays
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_data_dimensions(u_k=u_k, y_k=y_k, y_s=y_s[:1])

    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_data_dimensions(u_k=u_k, y_k=y_k, u_s=u_s[:1], y_s=y_s)


@pytest.mark.parametrize(
    "invalid_parameter",
    [
        "u_bounds_list",
        "y_bounds_list",
        "u_ylimits_list",
        "y_ylimits_list",
    ],
)
def test_validate_data_dimensions_bounds_lists(
    invalid_parameter: str,
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    u_k, y_k, u_s, y_s = dummy_plot_data
    m = u_k.shape[1]
    p = y_k.shape[1]

    # Determine correct length based on param_name
    expected_len = m if "u_" in invalid_parameter else p
    invalid_list = [(0.0, 1.0)] * (expected_len + 1)

    kwargs: dict[str, Any] = {invalid_parameter: invalid_list}

    # Verify `ValueError` is raised when bounds or
    # y-limits list lengths mismatch
    with pytest.raises(
        ValueError,
        match=rf"{invalid_parameter}.*does not match",
    ):
        validate_data_dimensions(
            u_k=u_k,
            y_k=y_k,
            u_s=u_s,
            y_s=y_s,
            **kwargs,
        )


def test_create_input_output_figure() -> None:
    # Define test parameters
    m = 2
    p = 3

    fig, axs_u, axs_y = create_input_output_figure(
        m, p, figsize=(10, 6), dpi=100, fontsize=12, title="Test Fig"
    )

    # Ensure figure is created with correct number of subplots
    assert isinstance(fig, Figure)
    assert len(axs_u) == m
    assert len(axs_y) == p
    for ax in list(axs_u) + list(axs_y):
        assert hasattr(ax, "plot")

    plt.close(fig)


def test_custom_legend_handler() -> None:
    # Define test parameters
    handler = HandlerInitMeasurementRect()
    fig, ax = plt.subplots()
    dummy_legend = Legend(ax, handles=[], labels=[])
    dummy_rect = Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.2)

    legend_artists = handler.create_artists(
        legend=dummy_legend,
        orig_handle=dummy_rect,
        xdescent=0,
        ydescent=0,
        width=1,
        height=1,
        fontsize=10,
        trans=plt.gca().transData,
    )

    assert any(isinstance(artist, Rectangle) for artist in legend_artists)

    plt.close(fig)
