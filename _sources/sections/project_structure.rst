Project Structure
=================

This section outlines the internal code structure of the project.

Direct Data-Driven MPC Controller
---------------------------------

The project is structured as a Python package, encapsulating the core logic of the Data-Driven MPC controllers within the following modules:

- :mod:`direct_data_driven_mpc.lti_data_driven_mpc_controller`: Implements a Data-Driven MPC controller for Linear Time-Invariant (LTI) systems in the :class:`~direct_data_driven_mpc.lti_data_driven_mpc_controller.LTIDataDrivenMPCController` class. This implementation is based on the **Nominal and Robust Data-Driven MPC schemes** described in :ref:`[1] <lti-citation>`.

- :mod:`direct_data_driven_mpc.nonlinear_data_driven_mpc_controller`: Implements a Data-Driven MPC controller for nonlinear systems in the :class:`~direct_data_driven_mpc.nonlinear_data_driven_mpc_controller.NonlinearDataDrivenMPCController` class. This implementation is based on the **Nonlinear Data-Driven MPC scheme** described in :ref:`[2] <nonlinear-citation>`.

The utility module :mod:`direct_data_driven_mpc.utilities.hankel_matrix` is used for constructing Hankel matrices and evaluating whether data sequences are persistently exciting of a given order.

Model Simulation
----------------

.. currentmodule:: direct_data_driven_mpc.utilities.models

The following utility modules have been implemented to simulate LTI and nonlinear systems:

- :mod:`direct_data_driven_mpc.utilities.models.lti_model`: Implements the :class:`~lti_model.LTIModel` and :class:`~lti_model.LTISystemModel` classes for simulating LTI systems.
- :mod:`direct_data_driven_mpc.utilities.models.nonlinear_model`: Implements the :class:`~nonlinear_model.NonlinearSystem` class for simulating nonlinear systems.

Controller Creation
-------------------

To modularize the creation of Data-Driven MPC controllers, the following utility modules are provided:

- :mod:`direct_data_driven_mpc.utilities.controller.controller_creation`: Provides functions for creating Data-Driven MPC controller instances from specified configuration parameters for both LTI and nonlinear controllers.
- :mod:`direct_data_driven_mpc.utilities.controller.controller_params`: Provides functions for loading Data-Driven MPC controller parameters from YAML configuration files for both LTI and nonlinear controllers.
- :mod:`direct_data_driven_mpc.utilities.controller.initial_data_generation`: Provides functions for generating input-output data from LTI and nonlinear systems, which can be used as initial trajectories in Data-Driven MPC controller creation.

Data-Driven Controller Simulation
---------------------------------

The :mod:`direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim` module implements the main control loops for both Data-Driven MPC controllers, following **Algorithms 1 and 2 of** :ref:`[1] <lti-citation>` for LTI systems and **Algorithm 1 of** :ref:`[2] <nonlinear-citation>` for nonlinear systems.

Visualization (Static and Animated Plots)
-----------------------------------------

Custom functions are provided in :ref:`direct_data_driven_mpc/utilities/visualization/ <visualization-utils>` to display input-output data in static and animated plots. These functions use Matplotlib for visualization and FFmpeg for saving animations in various formats (e.g., GIF, MP4).

Examples
--------

The ``examples`` directory contains scripts that demonstrate the operation of the Data-Driven MPC controller and reproduce the results presented in the referenced papers.

- `examples/lti_control/lti_dd_mpc_example.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/lti_control/lti_dd_mpc_example.py>`_: Demonstrates the setup, simulation, and data visualization of a Data-Driven MPC controller applied to an LTI system.
- `examples/lti_control/robust_lti_dd_mpc_reproduction.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/lti_control/robust_lti_dd_mpc_reproduction.py>`_: Implements a reproduction of the example presented in :ref:`[1] <lti-citation>`, showing various Robust Data-Driven MPC schemes applied to an LTI system.
- `examples/nonlinear_control/nonlinear_dd_mpc_example.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/nonlinear_control/nonlinear_dd_mpc_example.py>`_: Demonstrates the setup, simulation, and data visualization of a Data-Driven MPC controller applied to a nonlinear system while closely following the example presented in :ref:`[2] <nonlinear-citation>`.

Configuration Files
-------------------

The system and controller parameters used in the example scripts are defined in YAML configuration files in `examples/config/ <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/>`_. These parameters are based on the examples from **Section V of** :ref:`[1] <lti-citation>` for LTI systems, and from **Section V of** :ref:`[2] <nonlinear-citation>` for nonlinear systems.

- **Data-Driven MPC controllers:**

  - `examples/config/controllers/lti_dd_mpc_example_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/controllers/lti_dd_mpc_example_params.yaml>`_: Defines parameters for a Data-Driven MPC controller designed for LTI systems.
  - `examples/config/controllers/nonlinear_dd_mpc_example_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/controllers/nonlinear_dd_mpc_example_params.yaml>`_: Defines parameters for a Data-Driven MPC controller designed for nonlinear systems.

- **Models:**

  - `examples/config/models/four_tank_system_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/models/four_tank_system_params.yaml>`_: Defines system model parameters for a linearized version of a four-tank system.
  - `examples/config/models/nonlinear_cstr_system_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/models/nonlinear_cstr_system_params.yaml>`_: Defines system model parameters for a nonlinear continuous stirred tank reactor (CSTR) system.

- **Plots:**

  - `examples/config/plots/plot_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/plots/plot_params.yaml>`_: Defines Matplotlib properties of lines, legends, and figures for input-output plots.

A YAML loading function is provided in :mod:`direct_data_driven_mpc.utilities.yaml_config_loading`.


References
----------

See the full citations of the reference papers in the :ref:`Citation <citation-section>` section.
