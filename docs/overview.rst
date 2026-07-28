Overview
========

``idaes-sdoe`` is a Python package for Design of Experiments in Process Systems Engineering. Currently it mainly provides tools for sequential design
of experiments (SDOE).

The package is currently organized around the following core modules:

* reading and aggregating candidate or previous-data tables
* validation and setup of candidate and previous-data tables
* scaling and distance calculations used by design criteria
* design construction for uniform, non-uniform, and input-response workflows
* run ordering after a design has been selected
* optional helpers for candidate generation and missing-data imputation

Typical flow
------------

Most workflows follow the same path:

1. Load candidate data into a table.
2. Assign column roles.
3. Build a validated ``DesignSetup``.
4. Run one of the design methods under ``idaes_sdoe.design``.
5. Inspect the returned result and optionally apply ordering or plotting.

Most user errors occur at the table-contract level rather than inside the
algorithms. Candidate rows should represent allowable points. Inputs,
responses, and weights should be numeric. If an index column is not supplied,
``prepare_design_setup()`` can create one automatically.

Design families
---------------

Uniform space filling
   Selects points to cover the input space under a maximin or minimax
   criterion.

Non-uniform space filling
   Extends the uniform case with a weight column so that some candidate regions
   are emphasized more heavily than others.

Input-response space filling
   Builds a Pareto front of designs that trade off coverage in the input space
   against coverage in the response space.

What lives outside the core
---------------------------

The package also includes utilities that support the surrounding workflow but do
not define the main design criteria:

* ``idaes_sdoe.extras.candidate_generation`` for first-stage or sequential
  candidate generation
* ``idaes_sdoe.extras.imputation`` for response-surface fitting and imputation of
  missing values
* ``idaes_sdoe.plotting`` for compact Plotly visualizations
