Usage
=====

Example Scripts
---------------

The example scripts `lti_dd_mpc_example.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/lti_control/lti_dd_mpc_example.py>`_ and `nonlinear_dd_mpc_example.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/nonlinear_control/nonlinear_dd_mpc_example.py>`_ demonstrate the setup, simulation, and data visualization of the Data-Driven MPC controllers applied to Linear Time-Invariant (LTI) and nonlinear systems, respectively.

To run the example scripts, use the following commands:

Data-Driven MPC for LTI systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the example script with a seed of `18`, a simulation length of `400` steps, a verbosity level of `1`, and save the generated animation to a file:

.. code-block:: bash

    python examples/lti_control/lti_dd_mpc_example.py --seed 18 --t_sim 400 --verbose 1 --save_anim

Data-Driven MPC for Nonlinear systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the example script with a seed of `0`, a simulation length of `3000` steps, a verbosity level of `1`, and save the generated animation to a file:

.. code-block:: bash

    python examples/nonlinear_control/nonlinear_dd_mpc_example.py --seed 0 --t_sim 3000 --verbose 1 --save_anim

.. note::
   The ``--save_anim`` flag requires FFmpeg to be installed. See the :ref:`requirements` section for more details.

Customizing Controller Parameters
---------------------------------

To use different controller parameters, modify the configuration files in `examples/config/controllers/ <https://github.com/pavelacamposp/direct-data-driven-mpc/tree/main/examples/config/controllers>`_ for each controller, or specify a custom configuration file using the ``--controller_config_path`` argument.

Customizing System Models
-------------------------

Example system parameters are defined in `examples/config/models/ <https://github.com/pavelacamposp/direct-data-driven-mpc/tree/main/examples/config/models>`_.

- **LTI system:** Parameters can be modified directly in `four_tank_system_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/models/four_tank_system_params.yaml>`_.
- **Nonlinear system:** The system dynamics are defined in `nonlinear_cstr_model.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/nonlinear_control/utilities/nonlinear_cstr_model.py>`_ and its parameters in `nonlinear_cstr_system_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/models/nonlinear_cstr_system_params.yaml>`_.

Customizing Plots
-----------------

Matplotlib properties for input-output plots can be customized by modifying `plot_params.yaml <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/config/plots/plot_params.yaml>`_.

Additional Information
----------------------

Some key arguments used in the scripts are listed below:

+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+
| Argument                     | Type   | Description                                                                                                  |
+==============================+========+==============================================================================================================+
| ``--controller_config_path`` | str    | Path to the YAML file containing Data-Driven MPC controller parameters.                                      |
+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+
| ``--t_sim``                  | int    | Simulation length in time steps.                                                                             |
+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+
| ``--save_anim``              | flag   | If passed, saves the generated animation to a file using FFmpeg.                                             |
+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+
| ``--anim_path``              | str    | Path where the generated animation file will be saved. Includes the file name and its extension              |
|                              |        | (e.g., ``data-driven_mpc_sim.gif``).                                                                         |
+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+
| ``--verbose``                | int    | Verbosity level: ``0`` = no output, ``1`` = minimal output, ``2`` = detailed output.                         |
+------------------------------+--------+--------------------------------------------------------------------------------------------------------------+

To get the full list of arguments, run each script with the ``--help`` flag.

----

For a deeper understanding of the project and how the controllers operate, we recommend reading through the scripts and the docstrings of the implemented utility functions and classes. The documentation includes detailed descriptions of how the implementations follow the Data-Driven MPC controller schemes and algorithms described in the referenced papers.
