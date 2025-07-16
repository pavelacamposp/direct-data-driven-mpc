.. _nonlinear_controller:

.. currentmodule:: direct_data_driven_mpc.nonlinear_data_driven_mpc_controller

.. automodule:: direct_data_driven_mpc.nonlinear_data_driven_mpc_controller
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

Nonlinear Data-Driven MPC
=========================

This section documents the implementation of the Data-Driven MPC controller for nonlinear systems.

Nonlinear Data-Driven MPC Controller
------------------------------------

The main controller is implemented in the following class, following the **Nonlinear Data-Driven MPC scheme** described in :ref:`[2] <nonlinear-citation>`.

.. autosummary::
   :toctree: nonlinear_controller
   :template: autosummary/class_no_init

   NonlinearDataDrivenMPCController

Configuration Enums
-------------------

The following enumeration is used to define configurations for the architecture of Nonlinear Data-Driven MPC controllers.

.. autosummary::
   :toctree: nonlinear_controller
   :template: autosummary/enum

   AlphaRegType
