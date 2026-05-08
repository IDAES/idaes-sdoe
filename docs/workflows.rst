Common Workflows
================

Uniform design workflow
-----------------------

The simplest path is:

1. Load candidate data with ``load_csv()`` or ``aggregate_tables()``.
2. Mark inputs with ``ColumnRoles``.
3. Build a ``DesignSetup`` with ``prepare_design_setup()``.
4. Call ``design_uniform()`` or ``design_uniform_batch()``.
5. Optionally call ``order_runs()`` on the selected design.

Required table content:

* one row per candidate point
* at least one numeric input column
* optional previous-data table with the same relevant columns

Typical role configuration: ``ColumnRoles(inputs=[...], index="__id")``

Runtime estimates can be obtained with ``estimate_uniform_runtime()`` before
running a full search.

Non-uniform design workflow
---------------------------

The non-uniform path adds a weight column:

1. Load the candidate table.
2. Mark inputs and the weight column in ``ColumnRoles``.
3. Build a ``DesignSetup``.
4. Call ``design_nonuniform()`` with an ``mwr`` and weight-scaling method.
5. Use ``plot_nonuniform_weights()`` if diagnostic plots are helpful.

Required table content:

* one row per candidate point
* numeric input columns
* one numeric weight column

Typical role configuration:
``ColumnRoles(inputs=[...], weight="Weight", index="__id")``

Runtime estimates can be obtained with ``estimate_nonuniform_runtime()``.

Input-response workflow
-----------------------

The input-response path adds one or more response columns:

1. Load the candidate table.
2. Mark inputs and responses in ``ColumnRoles``.
3. Build a ``DesignSetup``.
4. Call ``design_input_response()``.
5. Inspect the returned Pareto front and per-design tables.

Required table content:

* one row per candidate point
* numeric input columns
* one or more numeric response columns

Typical role configuration:
``ColumnRoles(inputs=[...], responses=[...], index="__id")``

Runtime estimates can be obtained with ``estimate_input_response_runtime()``.

Candidate generation workflow
-----------------------------

Candidate generation can be used before any design search:

* ``load_template_specs()`` reads first-stage template bounds.
* ``specs_from_previous()`` infers bounds from existing data.
* ``generate_candidates()`` produces a new candidate table from those specs.

Imputation workflow
-------------------

When weight or response columns are incomplete:

1. Choose the input columns that explain the missing target.
2. Call ``fit_response_surface()`` to inspect model quality.
3. Call ``impute_missing_values()`` to fill the missing entries.

The imputation code is intentionally compact and model-light.

Run ordering workflow
---------------------

Run ordering is a post-processing step applied after a design has been chosen.

1. Take a selected design table.
2. Choose the input columns that define movement cost.
3. Call ``order_runs()``.

The result contains both the original design and the reordered table.
