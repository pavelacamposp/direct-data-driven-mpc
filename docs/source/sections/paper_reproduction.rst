Paper Reproduction
==================

Reproduction scripts are provided to validate our implementations by comparing them with the results presented in the referenced papers.

Data-Driven MPC for LTI systems
-------------------------------

The reproduction is implemented in `robust_lti_dd_mpc_reproduction.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/lti_control/robust_lti_dd_mpc_reproduction.py>`_. This script closely follows the example presented in **Section V of** :ref:`[1] <lti-citation>`, which demonstrates various Robust Data-Driven MPC controller schemes applied to a four-tank system model.

To run the script, execute the following command:

.. code-block:: bash

    python examples/lti_control/robust_lti_dd_mpc_reproduction.py

Data-Driven MPC for Nonlinear systems
-------------------------------------

The reproduction is included in the example script `nonlinear_dd_mpc_example.py <https://github.com/pavelacamposp/direct-data-driven-mpc/blob/main/examples/nonlinear_control/nonlinear_dd_mpc_example.py>`_, which closely follows the example presented in **Section V of** :ref:`[2] <nonlinear-citation>` for the control of a nonlinear continuous stirred tank reactor (CSTR) system.

To run the script, execute the following command:

.. code-block:: bash

    python examples/nonlinear_control/nonlinear_dd_mpc_example.py --seed 0

The figures below show the expected output from executing these scripts. The graphs from our results closely resemble those shown in **Fig. 2 of** :ref:`[1] <lti-citation>` and **Fig. 2 of** :ref:`[2] <nonlinear-citation>`, with minor differences due to randomization.

.. list-table::
   :header-rows: 0
   :class: vcenter

   * - .. image:: https://raw.githubusercontent.com/pavelacamposp/direct-data-driven-mpc/main/docs/resources/robust_dd_mpc_reproduction.png
         :alt: Robust Data-Driven MPC Animation
     - .. image:: https://raw.githubusercontent.com/pavelacamposp/direct-data-driven-mpc/main/docs/resources/nonlinear_dd_mpc_reproduction.png
         :alt: Nonlinear Data-Driven MPC Animation
   * - Reproduction of results from :ref:`[1] <lti-citation>`
     - Reproduction of results from :ref:`[2] <nonlinear-citation>`


References
----------

See the full citations of the reference papers in the :ref:`Citation <citation-section>` section.
