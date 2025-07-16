Package Installation
====================

.. _requirements:

Requirements
------------

This package requires the following:

- **Python** (>=3.10, <3.13). Python 3.13 is not fully supported, as some dependencies have not been compiled for this version yet. We recommend using Python 3.10 to 3.12.
- **FFmpeg:** Required for saving animations (e.g., GIF or MP4).

  - **On Windows:** You can download FFmpeg from `the official FFmpeg website <https://ffmpeg.org/download.html>`_. Ensure it's correctly added to your system's `PATH`.

  - **On Unix:** You can install it using your package manager. For Debian/Ubuntu:

    .. code-block:: bash

        sudo apt install ffmpeg

  Verify the installation by running this command:

  .. code-block:: bash

    ffmpeg -version


Installation
------------

Install ``direct-data-driven-mpc`` via PyPI:

.. code-block:: bash

    pip install direct-data-driven-mpc

**Note:** We recommend using a virtual environment to install this package, although it is not required.

For Contributors
----------------

If you plan to contribute to or develop the project, follow these steps to set up a local development environment:

1. Clone the repository and navigate to the project directory:

   .. code-block:: bash

    git clone https://github.com/pavelacamposp/direct-data-driven-mpc.git && cd direct-data-driven-mpc

2. Create and activate a virtual environment:

   .. tab-set::

        .. tab-item:: Unix/macOS

            .. code-block:: bash

                python3 -m venv .venv && source .venv/bin/activate

        .. tab-item:: Windows

            .. code-block:: batch

                python -m venv venv && venv\Scripts\activate

3. Install the package with development dependencies:

   .. code-block:: bash

    pip install -e ".[dev]"


This will install tools like ``pre-commit`` and ``mypy``. To enable automatic checks before each commit using ``pre-commit`` hooks, run:

.. code-block:: bash

    pre-commit install
