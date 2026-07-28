Tutorial Examples
=================

The executable tutorials for this project are maintained in the
``examples`` directory.

Running the examples
--------------------

Run the commands on this page from the repository root after activating the
Python environment in which ``idaes-sdoe`` is installed.

Install the Jupyter or marimo interface through the corresponding optional
dependency:

.. code-block:: bash

   pip install -e ".[notebook]"
   pip install -e ".[marimo]"

Both interfaces can be installed together with:

.. code-block:: bash

   pip install -e ".[notebook,marimo]"

Jupyter Notebook
~~~~~~~~~~~~~~~~

A Jupyter notebook presents a linear sequence of Markdown and Python cells.
Launch it with the exact command given in its catalog entry, then run its
cells in order.

marimo
~~~~~~

Use ``marimo edit`` to open the notebook with editable code cells and the
interactive interface. Use ``marimo run`` to open the streamlined application
interface without editable code cells. Both modes provide the same interactive
controls and computations.

Example catalog
---------------

Uniform space-filling in 2D — Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`examples/example-uniform.ipynb
<https://github.com/IDAES/idaes-sdoe/blob/main/examples/example-uniform.ipynb>`__
demonstrates a two-stage sequential uniform space-filling design on a regular
candidate grid in ``X1`` and ``X2``. It compares minimax and maximin designs
in the first stage, then augments a selected completed design.

**Supporting data:** ``examples/supporting_data/SDOE_Ex1_Candidates.csv``;
the notebook loads it automatically.

**Launch:**

.. code-block:: bash

   jupyter notebook examples/example-uniform.ipynb

Artifacts are written to a timestamped directory under
``examples/temp/nb_output``.

Uniform space-filling in 5D — Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`examples/example-uniform-5d.ipynb
<https://github.com/IDAES/idaes-sdoe/blob/main/examples/example-uniform-5d.ipynb>`__
demonstrates a two-stage minimax design for an irregular, five-dimensional
carbon-capture candidate set. It creates several Stage 1 design sizes, carries
the selected design forward as previous data, and adds Stage 2 runs.

**Supporting data:**
``examples/supporting_data/Candidate Points 8perc.csv``; the notebook loads it
automatically.

**Launch:**

.. code-block:: bash

   jupyter notebook examples/example-uniform-5d.ipynb

Artifacts are written to a timestamped directory under
``examples/temp/nb_output``.

Uniform space-filling in 5D — marimo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`examples/example-uniform-5d-marimo.py
<https://github.com/IDAES/idaes-sdoe/blob/main/examples/example-uniform-5d-marimo.py>`__
provides an interactive interface for the same two-stage 5D workflow. It
allows users to enter or upload candidates, configure both stages, estimate
runtime, run the design searches, select visualizations, and export results.

**Supporting data:**
``examples/supporting_data/Candidate Points 8perc.csv``. Under **Prepare the
Candidate Set**, choose **Upload a candidate CSV**, then select this file. The
manual-entry option can be used for a different candidate table.

**Launch with editable code cells:**

.. code-block:: bash

   marimo edit examples/example-uniform-5d-marimo.py

**Launch as an application:**

.. code-block:: bash

   marimo run examples/example-uniform-5d-marimo.py

The configurable session artifact directory appears near the top of the
interface. It defaults to a timestamped directory under
``examples/temp/marimo_output``.
