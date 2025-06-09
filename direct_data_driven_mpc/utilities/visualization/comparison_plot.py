from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .control_plot import (
    plot_input_output,
)
from .plot_utilities import (
    check_list_length,
    create_input_output_figure,
    init_dict_if_none,
)


def plot_input_output_comparison(
    u_data: list[np.ndarray],
    y_data: list[np.ndarray],
    y_s: np.ndarray,
    u_s: np.ndarray | None = None,
    u_bounds_list: list[tuple[float, float]] | None = None,
    y_bounds_list: list[tuple[float, float]] | None = None,
    inputs_line_param_list: list[dict[str, Any]] | None = None,
    outputs_line_param_list: list[dict[str, Any]] | None = None,
    setpoints_line_params: dict[str, Any] | None = None,
    bounds_line_params: dict[str, Any] | None = None,
    var_suffix_list: list[str] | None = None,
    legend_params: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
    u_ylimits_list: list[tuple[float, float]] | None = None,
    y_ylimits_list: list[tuple[float, float]] | None = None,
    fontsize: int = 12,
    title: str | None = None,
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
    u_setpoint_labels: list[str] | None = None,
    y_setpoint_labels: list[str] | None = None,
    x_axis_labels: list[str] | None = None,
    input_y_axis_labels: list[str] | None = None,
    output_y_axis_labels: list[str] | None = None,
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
        u_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            input data sequences. Defaults to `None`.
        y_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each output data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            output data sequences. Defaults to `None`.
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
        bounds_line_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the lines used to plot the bounds of
            input-output data series (e.g., color, linestyle, linewidth). If
            not provided, Matplotlib's default line properties will be used.
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
        input_labels (list[str] | None): A list of strings specifying custom
            legend labels for input data series. If provided, the label at each
            index will override the default label constructed using
            `var_suffix_list`.
        output_labels (list[str] | None): A list of strings specifying custom
            legend labels for output data series. If provided, the label at
            each index will override the default label constructed using
            `var_suffix_list`.
        u_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for input setpoint series. If provided, the
            label at each index will override the corresponding default label.
        y_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for output setpoint series. If provided, the
            label at each index will override the corresponding default label.
        x_axis_labels (list[str] | None): A list of strings specifying custom
            X-axis labels for each subplot. If provided, the label at each
            index will override the corresponding default label.
        input_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each input subplot. If provided, the label
            at each index will override the corresponding default label.
        output_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each output subplot. If provided, the
            label at each index will override the corresponding default label.

    Raises:
        ValueError: If input/output array shapes, or line parameter list
            lengths, are not as expected.
    """
    validate_comparison_plot_parameters(
        u_data=u_data,
        y_data=y_data,
        inputs_line_param_list=inputs_line_param_list,
        outputs_line_param_list=outputs_line_param_list,
        var_suffix_list=var_suffix_list,
        input_labels=input_labels,
        output_labels=output_labels,
    )

    # Initialize Matplotlib params if not provided
    setpoints_line_params = init_dict_if_none(setpoints_line_params)
    legend_params = init_dict_if_none(legend_params)

    # Create figure with subplots
    m = u_data[0].shape[1]  # Number of inputs
    p = y_data[0].shape[1]  # Number of outputs

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

        # Retrieve plot labels for each index
        var_suffix = var_suffix_list[i] if var_suffix_list else ""
        input_label = input_labels[i] if input_labels else None
        output_label = output_labels[i] if output_labels else None

        # Plot setpoint line only for the first data set to prevent cluttering
        plot_setpoint_lines = i == 0

        # Plot input-output data
        plot_input_output(
            u_k=u_data[i],
            y_k=y_data[i],
            u_s=u_s,
            y_s=y_s,
            u_bounds_list=u_bounds_list,
            y_bounds_list=y_bounds_list,
            inputs_line_params=inputs_line_params,
            outputs_line_params=outputs_line_params,
            setpoints_line_params=setpoints_line_params,
            bounds_line_params=bounds_line_params,
            var_suffix=var_suffix,
            dpi=dpi,
            u_ylimits_list=u_ylimits_list,
            y_ylimits_list=y_ylimits_list,
            fontsize=fontsize,
            legend_params=legend_params,
            axs_u=axs_u,
            axs_y=axs_y,
            input_label=input_label,
            output_label=output_label,
            u_setpoint_labels=u_setpoint_labels,
            y_setpoint_labels=y_setpoint_labels,
            x_axis_labels=x_axis_labels,
            input_y_axis_labels=input_y_axis_labels,
            output_y_axis_labels=output_y_axis_labels,
            plot_setpoint_lines=plot_setpoint_lines,
        )

    # Show plot
    plt.show()


def validate_comparison_plot_parameters(
    u_data: list[np.ndarray],
    y_data: list[np.ndarray],
    inputs_line_param_list: list[dict[str, Any]] | None = None,
    outputs_line_param_list: list[dict[str, Any]] | None = None,
    var_suffix_list: list[str] | None = None,
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
) -> None:
    if not u_data or not y_data:
        raise ValueError(
            "`u_data` and `y_data` must contain at least one simulation."
        )

    if len(u_data) != len(y_data):
        raise ValueError(
            "`u_data` and `y_data` must have the same number of trajectories."
        )

    # Validate input-output data dimensions
    u_shape = u_data[0].shape
    y_shape = y_data[0].shape

    if not all(u.shape == u_shape for u in u_data):
        raise ValueError(
            f"All `u_data` arrays must have the same shape ({u_shape})."
        )

    if not all(y.shape == y_shape for y in y_data):
        raise ValueError(
            f"All `y_data` arrays must have the same shape ({y_shape})."
        )

    # Validate list lengths if provided
    n_sim = len(u_data)

    # Lists for input plots
    check_list_length("inputs_line_param_list", inputs_line_param_list, n_sim)
    check_list_length(
        "outputs_line_param_list", outputs_line_param_list, n_sim
    )
    check_list_length("var_suffix_list", var_suffix_list, n_sim)
    check_list_length("input_labels", input_labels, n_sim)
    check_list_length("output_labels", output_labels, n_sim)
