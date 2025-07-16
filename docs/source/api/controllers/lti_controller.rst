.. _lti_controller:

.. currentmodule:: direct_data_driven_mpc.lti_data_driven_mpc_controller

.. automodule:: direct_data_driven_mpc.lti_data_driven_mpc_controller
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

LTI Data-Driven MPC
===================

This section documents the implementation of the Data-Driven MPC controller for Linear Time-Invariant (LTI) systems.

LTI Data-Driven MPC Controller
------------------------------

The main controller is implemented in the following class, following the **Nominal and Robust Data-Driven MPC schemes** described in :ref:`[1] <lti-citation>`.

.. autosummary::
   :toctree: lti_controller
   :template: autosummary/class_no_init

   LTIDataDrivenMPCController

Configuration Enums
-------------------

The following enumerations are used to define configurations for the architecture of LTI Data-Driven MPC controllers.

.. autosummary::
   :toctree: lti_controller
   :template: autosummary/enum

   LTIDataDrivenMPCType
   SlackVarConstraintType
