Package Map
===========

This page explains how the package is organized. For symbol-level API details,
see :doc:`api`.

Top-level modules
-----------------

``idaes_sdoe.io``
   Lightweight CSV I/O and table aggregation helpers.

``idaes_sdoe.models``
   Dataclasses that carry validated setup state and design results.

``idaes_sdoe.exceptions``
   Small exception hierarchy for configuration, validation, and optional
   dependency failures.

Model dataclasses
-----------------

The ``idaes_sdoe.models`` module defines the main API objects that correspond to
the setup and results a user would inspect in an SDoE workflow.

``ColumnRoles``
   Describes which table columns act as inputs, responses, weights, or an
   index.

``DesignSetup``
   Bundles the validated candidate table, optional previous data, column-role
   assignments, and scaling bounds before a design algorithm runs.

``UniformDesignResult``
   Stores the output of a uniform space-filling search.

``NonUniformDesignResult``
   Stores the output of a non-uniform space-filling search, including scaled
   design values and the MWR setting.

``InputResponseDesignResult``
   Stores the IRSF Pareto front and the corresponding design tables.

``RunOrderResult``
   Stores the reordered execution sequence for a chosen design.

``InputSpec``
   Describes one variable or fixed input for candidate generation.

``CandidateGenerationResult``
   Stores the generated candidate table and the sampling metadata.

``ResponseSurfaceValidation``
   Stores validation metrics and fitted-model state for response-surface
   fitting and imputation.

``idaes_sdoe.validation``
   Checks table structure, aligns previous data, inserts an index column when
   needed, and builds the shared ``DesignSetup`` object.

``idaes_sdoe.scaling``
   Builds bounds, scales columns into a unit range, rescales arrays back into
   engineering units, and scales non-uniform weights.

``idaes_sdoe.distance``
   Distance-matrix helpers shared by the design and run-ordering code.

``idaes_sdoe.ordering``
   Post-processes a chosen design into a run order using either an exact TSP
   solver or a greedy fallback.

``idaes_sdoe.plotting``
   Compact Plotly figures for pairwise design views, weight diagnostics, and
   Pareto plots.

Public import surface
---------------------

The package exposes a small top-level import layer:

* from ``idaes_sdoe``: table I/O, ``prepare_design_setup()``, and the main
  dataclasses
* from ``idaes_sdoe.design``: design algorithms and runtime estimators
* from ``idaes_sdoe.extras``: candidate-generation and imputation helpers

This keeps common entry points short without flattening the whole package.

Design modules
--------------

``idaes_sdoe.design.uniform``
   Uniform space-filling search with ``maximin`` and ``minimax`` modes.

``idaes_sdoe.design.nonuniform``
   Weighted maximin search for non-uniform space filling.

``idaes_sdoe.design.input_response``
   Input-response space-filling search that returns a Pareto front of
   trade-off designs.

Support modules
---------------

``idaes_sdoe.extras.candidate_generation``
   Builds candidate tables from template ranges or previous data.

``idaes_sdoe.extras.imputation``
   Fits simple response-surface models and fills missing target values.

Shared data flow
----------------

The source tree follows a consistent pattern:

1. ``io`` loads tables.
2. ``validation`` checks roles and creates ``DesignSetup``.
3. ``scaling`` and ``distance`` provide the numeric helpers used by the
   algorithms.
4. One of the ``design`` modules constructs a design result object.
5. ``ordering`` or ``plotting`` can be applied afterwards if needed.

Result objects
--------------

Each design family returns a dedicated result dataclass from
``idaes_sdoe.models``:

* ``UniformDesignResult``
* ``NonUniformDesignResult``
* ``InputResponseDesignResult``
* ``RunOrderResult``

These results carry the selected design rows plus the criterion values and
metadata needed by calling code.
