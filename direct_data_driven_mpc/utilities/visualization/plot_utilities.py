"""
Utility functions and classes for static and animated control input-output
plots.

This module provides helper functions used in control input-output data plot
generation, and a custom Matplotlib legend handler class for highlighting
initial measurement periods in animated plots.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Transform


class HandlerInitMeasurementRect(HandlerPatch):
    """
    A custom legend handler for the rectangle representing the initial
    input-output measurement period in control input-output plot animations.
    """

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Rectangle | Line2D]:
        # Make sure orig_handle is a Rectangle
        assert isinstance(orig_handle, Rectangle)

        # Define the main rectangle
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            transform=trans,
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )

        # Create dashed vertical lines at the sides of the rectangle
        line1 = Line2D(
            [xdescent, xdescent],
            [ydescent, ydescent + height],
            color="black",
            linestyle=(0, (2, 2)),
            linewidth=1,
        )
        line2 = Line2D(
            [xdescent + width, xdescent + width],
            [ydescent, ydescent + height],
            color="black",
            linestyle=(0, (2, 2)),
            linewidth=1,
        )

        # Add transform to the vertical lines
        line1.set_transform(trans)
        line2.set_transform(trans)

        return [rect, line1, line2]


def validate_data_dimensions(
    u_k: np.ndarray,
    y_k: np.ndarray,
    y_s: np.ndarray,
    u_s: np.ndarray | None = None,
    u_bounds_list: list[tuple[float, float]] | None = None,
    y_bounds_list: list[tuple[float, float]] | None = None,
    u_ylimits_list: list[tuple[float, float]] | None = None,
    y_ylimits_list: list[tuple[float, float]] | None = None,
    u_setpoint_labels: list[str] | None = None,
    y_setpoint_labels: list[str] | None = None,
    x_axis_labels: list[str] | None = None,
    input_y_axis_labels: list[str] | None = None,
    output_y_axis_labels: list[str] | None = None,
) -> None:
    """
    Validate that input-output data arrays, and bound and ylimit lists have the
    expected shapes and lengths.

    Args:
        u_k (np.ndarray): An array containing control input data of shape (T,
            m), where `m` is the number of inputs and `T` is the number of
            time steps.
        y_k (np.ndarray): An array containing system output data of shape (T,
            p), where `p` is the number of outputs and `T` is the number of
            time steps.
        y_s (np.ndarray): An array containing output setpoint values of shape
            (T, p), where `p` is the number of outputs and `T` is the number of
            time steps.
        u_s (np.ndarray | None): An array containing input setpoint values of
            shape (T, m), where `m` is the number of inputs and `T` is the
            number of time steps.
        u_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence.
        y_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each output data
            sequence.
        u_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            input subplot.
        y_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            output subplot.
        u_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for input setpoint series.
        y_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for output setpoint series.
        x_axis_labels (list[str] | None): A list of strings specifying custom
            X-axis labels for each subplot.
        input_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each input subplot.
        output_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each output subplot.

    Raises:
        ValueError: If any array dimensions mismatch expected shapes, or if
            the lengths of the list arguments do not match the number of
            subplots.
    """
    # Check input-output data dimensions
    if u_k.shape[0] != y_k.shape[0]:
        raise ValueError(
            "Dimension mismatch. The number of time steps for `u_k` "
            f"({u_k.shape[0]}) and `y_k` ({y_k.shape[0]}) must match."
        )
    if y_k.shape != y_s.shape:
        raise ValueError(
            f"Shape mismatch. The shapes of `y_k` ({y_k.shape}) and "
            f"`y_s` ({y_s.shape}) must match."
        )

    # If input setpoint is passed, verify input data dimension match
    if u_s is not None:
        if u_k.shape != u_s.shape:
            raise ValueError(
                f"Shape mismatch. The shape of `u_k` ({u_k.shape}) and "
                f"`u_s` ({u_s.shape}) must match."
            )

    # Error handling for list lengths
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs

    check_list_length("u_bounds_list", u_bounds_list, m)
    check_list_length("y_bounds_list", y_bounds_list, p)
    check_list_length("u_ylimits_list", u_ylimits_list, m)
    check_list_length("y_ylimits_list", y_ylimits_list, p)

    check_list_length("u_setpoint_labels", u_setpoint_labels, m)
    check_list_length("y_setpoint_labels", y_setpoint_labels, p)

    # Lists for Y-axis labels
    check_list_length("x_axis_labels", x_axis_labels, max(m, p))
    check_list_length("input_y_axis_labels", input_y_axis_labels, m)
    check_list_length("output_y_axis_labels", output_y_axis_labels, p)


def get_padded_limits(
    X: np.ndarray,
    X_s: np.ndarray | None = None,
    pad_percentage: float = 0.05,
) -> tuple[float, float]:
    """
    Get the minimum and maximum limits from two data sequences extended by
    a specified percentage of the combined data range.

    Args:
        X (np.ndarray): First data array.
        X_s (np.ndarray | None): Second data array. If `None`, only `X` is
            considered. Defaults to `None`.
        pad_percentage (float): The percentage of the data range to be used
            as padding. Defaults to 0.05.

    Returns:
        tuple[float, float]: A tuple containing padded minimum and maximum
        limits for the combined data from `X` and `X_s`.
    """
    # Get minimum and maximum limits from data sequences
    X_min, X_max = np.min(X), np.max(X)
    if X_s is not None:
        X_s_min, X_s_max = np.min(X_s), np.max(X_s)
        X_lim_min = min(X_min, X_s_min)
        X_lim_max = max(X_max, X_s_max)
    else:
        X_lim_min, X_lim_max = X_min, X_max

    # Extend limits by a percentage of the overall data range
    X_range = X_lim_max - X_lim_min
    X_lim_min -= X_range * pad_percentage
    X_lim_max += X_range * pad_percentage

    return (X_lim_min, X_lim_max)


def get_text_width_in_data(
    text_object: Text, axis: Axes, fig: Figure | SubFigure
) -> float:
    """
    Calculate the bounding box width of a text object in data coordinates.

    Args:
        text_object (Text): A Matplotlib text object.
        axis (Axes): The axis on which the text object is displayed.
        fig (Figure | SubFigure): The Matplotlib figure or subfigure that
            contains the axis.

    Returns:
        float: The width of the text object's bounding box in data coordinates.
    """
    # Get the bounding box of the text object in pixel coordinates
    render = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    text_box = text_object.get_window_extent(renderer=render)

    # Convert the bounding box from pixel coordinates to data coordinates
    text_box_data = axis.transData.inverted().transform(text_box)

    # Calculate the width of the bounding box in data coordinates
    text_box_width = text_box_data[1][0] - text_box_data[0][0]

    return text_box_width


def filter_and_reorder_legend(
    axis: Axes,
    legend_params: dict[str, Any],
    end_labels_list: list[str] | None = None,
) -> None:
    """
    Remove duplicate entries from the legend of a Matplotlib axis. Optionally,
    move specified labels to the end of the legend.

    Note:
        The appearance of the plot legend can be customized by passing a
        dictionary of Matplotlib legend properties.

    Args:
        axis (Axes): The Matplotlib axis containing the legend to modify.
        legend_params (dict[str, Any]): A dictionary of Matplotlib properties
            for customizing the plot legend (e.g., fontsize, loc,
            handlelength).
        end_labels_list (list[str] | None): A list of labels to move to the end
            of the legend. Labels are moved in the order provided, with the
            last label in the list becoming the final legend entry. If not
            provided, the legend labels will not be reordered. Defaults to
            `None`.
    """
    # Initialize `last_labels_list` if not provided
    if end_labels_list is None:
        end_labels_list = []

    # Get labels and handles from axis without duplicates
    handles, labels = axis.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles, strict=False))

    # Reorder labels if `last_label_list` is provided
    for last_label in end_labels_list:
        if last_label in unique_labels:
            last_handle = unique_labels.pop(last_label)
            unique_labels[last_label] = last_handle

    # Update the legend with the unique handles and labels
    axis.legend(unique_labels.values(), unique_labels.keys(), **legend_params)


def create_input_output_figure(
    m: int,
    p: int,
    figsize: tuple[float, float],
    dpi: int,
    fontsize: int,
    title: str | None = None,
) -> tuple[Figure, list[Axes], list[Axes]]:
    """
    Create a Matplotlib figure with two rows of subplots: one for control
    inputs and one for system outputs, and return the created figure and
    axes.

    If a title is provided, it will be set as the overall figure title.
    Each row of subplots will have its own title for 'Control Inputs' and
    'System Outputs'.

    Args:
        m (int): The number of control inputs (subplots in the first row).
        p (int): The number of system outputs (subplots in the second row).
        figsize (tuple[float, float]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        fontsize (int): The fontsize for suptitles.
        title (str | None): The title for the overall figure. If `None`, no
            title will be added. Defaults to `None`.

    Returns:
        tuple: A tuple containing:

        - Figure: The created Matplotlib figure.
        - list[Axes]: A list of axes for control inputs subplots.
        - list[Axes]: A list of axes for system outputs subplots.
    """
    # Create figure
    fig = plt.figure(num=title, layout="constrained", figsize=figsize, dpi=dpi)

    # Modify constrained layout padding, preventing mypy [call-arg] error
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, ConstrainedLayoutEngine):
        layout_engine.set(w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0)

    # assert layout_engine is ConstrainedLayoutEngine
    # layout_engine.set(w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0)

    # Set overall figure title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize + 3, fontweight="bold")

    # Create subfigures for input and output data plots
    subfigs = fig.subfigures(2, 1)

    # Add titles for input and output subfigures
    subfigs[0].suptitle(
        "Control Inputs", fontsize=fontsize + 2, fontweight="bold"
    )
    subfigs[1].suptitle(
        "System Outputs", fontsize=fontsize + 2, fontweight="bold"
    )

    # Create subplots
    axs_u = subfigs[0].subplots(1, m)
    axs_y = subfigs[1].subplots(1, p)

    # Ensure axs_u and axs_y are always lists
    if m == 1:
        axs_u = [axs_u]
    if p == 1:
        axs_y = [axs_y]

    return fig, axs_u, axs_y


def init_dict_if_none(input_dict: dict | None) -> dict:
    """
    Return an empty dictionary if the input is `None`, otherwise return the
    input.

    Args:
        input_dict (dict | None): A dictionary or `None`.

    Returns:
        dict: The original dictionary or an empty one.
    """
    return {} if input_dict is None else input_dict


def get_label_from_list(
    label_list: list[str], index: int, fallback: str
) -> str:
    """
    Get a label from a list by index, or return a fallback value if the list is
    empty.

    Args:
        label_list (list[str]): A list of label strings.
        index (int): The index of the desired label.
        fallback (str): A fallback string to return if the list is empty.

    Returns:
        str: The label at the specified index or the fallback string.
    """
    return label_list[index] if label_list else fallback


def check_list_length(
    name: str, data_list: list | None, expected: int
) -> None:
    """
    Verify whether a list contains the expected number of elements.

    Args:
        name (str): The name of the list (for error message context).
        data_list (list | None): The list to check.
        expected (int): The expected number of elements in the list.

    Raises:
        ValueError: If the list length does not match the expected value.
    """
    if data_list and len(data_list) != expected:
        raise ValueError(
            f"The length of `{name}` ({len(data_list)}) does not match "
            f"the expected value ({expected})."
        )
