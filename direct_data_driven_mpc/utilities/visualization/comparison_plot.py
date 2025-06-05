from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .control_plot import (
    plot_input_output,
)
from .plot_utilities import (
    create_input_output_figure,
    init_dict_if_none,
)


def plot_input_output_comparison(
    u_data: list[np.ndarray],
    y_data: list[np.ndarray],
    y_s: np.ndarray,
    u_s: np.ndarray | None = None,
    inputs_line_param_list: list[dict[str, Any]] | None = None,
    outputs_line_param_list: list[dict[str, Any]] | None = None,
    setpoints_line_params: dict[str, Any] | None = None,
    var_suffix_list: list[str] | None = None,
    legend_params: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
    u_ylimits_list: list[tuple[float, float]] | None = None,
    y_ylimits_list: list[tuple[float, float]] | None = None,
    fontsize: int = 12,
    title: str | None = None,
) -> None:
    """
    Plot multiple input-output trajectories with setpoints in a Matplotlib
    figure for control system comparison.

    This function creates a figure with two rows of subplots: the first row
    for control inputs, and the second for system outputs. Each subplot shows
    the trajectories of each data series alongside its setpoint line. Useful
    for comparing the performance of different control systems.

    Args:
        u_data (list[np.ndarray]): A list of `M` arrays of shape (T, m)
            containing control input data from `M` simulations. `T` is the
            number of time steps, and `m` is the number of control inputs.
        y_data (list[np.ndarray]): A list of `M` arrays of shape (T, p)
            containing system output data from `M` simulations. `T` is the
            number of time steps, and `p` is the number of system outputs.
        y_s (np.ndarray): An array of shape (T, p) containing `p` output
            setpoint values. These setpoints correspond to the system outputs
            from `y_data`.
        u_s (np.ndarray | None): An array of shape (T, m) containing `m` input
            setpoint values. These setpoints correspond to the control inputs
            from `u_data`. If `None`, input setpoint lines will not be plotted.
            Defaults to `None`.
        inputs_line_param_list (list[dict[str, Any]] | None): A list of
            `M` dictionaries, where each dictionary specifies Matplotlib
            properties for customizing the plot lines corresponding to one of
            the `M` input data arrays in `u_data`. If not provided,
            Matplotlib's default line properties will be used.
        outputs_line_param_list (list[dict[str, Any]] | None): A list of
            `M` dictionaries, where each dictionary specifies Matplotlib
            properties for customizing the plot lines corresponding to one of
            the `M` output data arrays in `y_data`. If not provided,
            Matplotlib's default line properties will be used.
        setpoints_line_params (dict[str, Any] | None): A dictionary of
            Matplotlib properties for customizing the lines used to plot the
            setpoint values (e.g., color, linestyle, linewidth). If not
            provided, Matplotlib's default line properties will be used.
        var_suffix_list (list[str] | None): A list of strings appended to each
            data series label in the plot legend. If not provided, no strings
            are appended.
        legend_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the plot legends (e.g., fontsize,
            loc, handlelength). If not provided, Matplotlib's default legend
            properties will be used.
        figsize (tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        u_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            input subplot. If `None`, the Y-axis limits will be determined
            automatically.
        y_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            output subplot. If `None`, the Y-axis limits will be determined
            automatically.
        fontsize (int): The fontsize for labels, legends and axes ticks.
        title (str | None): The title for the created plot figure.

    Raises:
        ValueError: If input/output array shapes, or line parameter list
            lengths, are not as expected.
    """
    u_shape = u_data[0].shape
    y_shape = y_data[0].shape

    # Validate input-output data dimensions
    if not all(u.shape == u_shape for u in u_data):
        raise ValueError(
            f"All `u_data` arrays must have the same shape ({u_shape})."
        )

    if not all(y.shape == y_shape for y in y_data):
        raise ValueError(
            f"All `y_data` arrays must have the same shape ({y_shape})."
        )

    # Validate plot line parameter list lengths
    if inputs_line_param_list and outputs_line_param_list:
        input_line_params_len = len(inputs_line_param_list)
        output_line_params_len = len(outputs_line_param_list)
        if input_line_params_len != output_line_params_len:
            raise ValueError(
                "The lengths of `inputs_line_param_list` ("
                f"{input_line_params_len}) and `outputs_line_param_list` ("
                f"{output_line_params_len}) do not match."
            )

    # Initialize Matplotlib params if not provided
    setpoints_line_params = init_dict_if_none(setpoints_line_params)
    legend_params = init_dict_if_none(legend_params)

    # Create figure with subplots
    m = u_shape[1]  # Number of inputs
    p = y_shape[1]  # Number of outputs

    _, axs_u, axs_y = create_input_output_figure(
        m=m, p=p, figsize=figsize, dpi=dpi, fontsize=fontsize, title=title
    )

    # Plot data iterating through each data array
    for i in range(len(u_data)):
        # Initialize Matplotlib params if not provided
        inputs_line_params = (
            init_dict_if_none(inputs_line_param_list[i])
            if inputs_line_param_list
            else None
        )
        outputs_line_params = (
            init_dict_if_none(outputs_line_param_list[i])
            if outputs_line_param_list
            else None
        )

        var_suffix = var_suffix_list[i] if var_suffix_list else ""

        # Plot input-output data
        plot_input_output(
            u_k=u_data[i],
            y_k=y_data[i],
            u_s=u_s,
            y_s=y_s,
            inputs_line_params=inputs_line_params,
            outputs_line_params=outputs_line_params,
            setpoints_line_params=setpoints_line_params,
            data_label=var_suffix,
            dpi=dpi,
            u_ylimits_list=u_ylimits_list,
            y_ylimits_list=y_ylimits_list,
            fontsize=fontsize,
            legend_params=legend_params,
            axs_u=axs_u,
            axs_y=axs_y,
        )

    # Show plot
    plt.show()
