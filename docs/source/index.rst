Welcome to Direct Data-Driven Model Predictive Control (MPC)!
=============================================================

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Robust Data-Driven MPC
     - Nonlinear Data-Driven MPC
   * - .. image:: https://raw.githubusercontent.com/pavelacamposp/direct-data-driven-mpc/main/docs/resources/robust_dd_mpc_anim.gif
          :alt: Robust Data-Driven MPC Animation
     - .. image:: https://raw.githubusercontent.com/pavelacamposp/direct-data-driven-mpc/main/docs/resources/nonlinear_dd_mpc_anim.gif
          :alt: Nonlinear Data-Driven MPC Animation
   * - Robust controller applied to an LTI system.
     - Nonlinear controller applied to a nonlinear system.

This repository provides a Python implementation of Direct Data-Driven Model Predictive Control (MPC) controllers for Linear Time-Invariant (LTI) and nonlinear systems using CVXPY. It includes **robust** and **nonlinear** controllers implemented based on the Data-Driven MPC schemes presented in the papers `Data-Driven Model Predictive Control With Stability and Robustness Guarantees <https://ieeexplore.ieee.org/document/9109670>`_ and `Linear Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case <https://ieeexplore.ieee.org/document/9756053>`_ by J. Berberich et al.

A **direct data-driven controller** maps measured input-output data from an unknown system *directly* onto the controller without requiring an explicit system identification step. This approach is particularly useful in applications where the system dynamics are too complex to be modeled accurately or where traditional system identification methods are impractical or difficult to apply.

----

**Disclaimer:** This is an independent project based on the referenced papers and does not contain the official implementations from the authors.

----

License
=======

This project is licensed under the MIT License. Please refer to :ref:`license` for more details.

.. _citation-section:

Citation
========

If you use these controller implementations in your research, please cite the original papers:

.. _lti-citation:

Data-Driven MPC Control for Linear Time-Invariant (LTI) systems
---------------------------------------------------------------

[1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Data-Driven Model Predictive Control With Stability and Robustness Guarantees," in IEEE Transactions on Automatic Control, vol. 66, no. 4, pp. 1702-1717, April 2021, doi: `10.1109/TAC.2020.3000182 <https://doi.org/10.1109/TAC.2020.3000182>`_

**BibTeX entry:**

.. code-block:: bibtex

   @ARTICLE{9109670,
      author={Berberich, Julian and Köhler, Johannes and Müller, Matthias A. and Allgöwer, Frank},
      journal={IEEE Transactions on Automatic Control},
      title={Data-Driven Model Predictive Control With Stability and Robustness Guarantees},
      year={2021},
      volume={66},
      number={4},
      pages={1702-1717},
      doi={10.1109/TAC.2020.3000182}}


.. _nonlinear-citation:

Data-Driven MPC Control for Nonlinear systems
---------------------------------------------

[2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case," in IEEE Transactions on Automatic Control, vol. 67, no. 9, pp. 4406-4421, Sept. 2022, doi: `10.1109/TAC.2022.3166851 <https://doi.org/10.1109/TAC.2022.3166851>`_

**BibTeX entry:**

.. code-block:: bibtex

   @ARTICLE{9756053,
      author={Berberich, Julian and Köhler, Johannes and Müller, Matthias A. and Allgöwer, Frank},
      journal={IEEE Transactions on Automatic Control},
      title={Linear Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case},
      year={2022},
      volume={67},
      number={9},
      pages={4406-4421},
      doi={10.1109/TAC.2022.3166851}}


Table of Contents
=================

.. toctree::
   :caption: Overview

   sections/installation
   sections/usage
   sections/paper_reproduction
   sections/project_structure

.. toctree::
   :maxdepth: 2
   :caption: Source API

   api/index

.. toctree::
   :caption: References

   sections/license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
