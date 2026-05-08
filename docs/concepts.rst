Concepts
========

``idaes-sdoe`` is currently built around a small set of recurring SDoE concepts. These are
the main terms to understand before using the API.

Candidate set
-------------

A candidate set is the full table of allowable points. Each row is one
candidate point. Design algorithms select rows from this table.

Design
------

A design is the selected subset of candidate rows. The design size is the
number of rows selected into that subset.

One candidate set can support many alternative designs. For example, a uniform
workflow may build designs of size 8, 9, and 10 from the same candidate pool.

Previous data
-------------

Previous data represents runs that already exist. It is optional, but important
for sequential workflows.

When previous data is supplied:

* the new design is evaluated relative to both the candidate rows and the
  existing runs
* the design size usually means the number of new points to add

Column roles
------------

The package needs to know how each relevant column participates in the
workflow. This is described with ``ColumnRoles``.

The main roles are:

* ``inputs``: columns that define the design space
* ``responses``: columns used by input-response workflows
* ``weight``: column used by non-uniform workflows
* ``index``: optional identifier column used to track rows

Design setup
------------

``prepare_design_setup()`` validates the tables, aligns previous data to the
candidate schema, adds an index column when requested, and builds a reusable
``DesignSetup`` object.

This setup object is the shared input to the design algorithms.

Bounds and scaling
------------------

Distance-based design criteria depend on consistent scaling. The setup stage
therefore builds per-column bounds that are used by the uniform, non-uniform,
and input-response methods.

If explicit bounds are not provided, the package infers them from the supplied
tables.

Random starts
-------------

The uniform, non-uniform, and input-response methods are randomized search
procedures. ``num_restarts`` controls how many candidate designs are tried.

* fewer restarts: faster, but less search effort
* more restarts: slower, but better chance of finding a stronger design

Criteria
--------

Uniform space filling
   Uses ``maximin`` or ``minimax`` to spread design points across the input
   space.

Non-uniform space filling
   Uses a weighted distance criterion so some candidate regions are emphasized
   more strongly than others.

Input-response space filling
   Balances coverage in the input space and response space, and returns a
   Pareto front of trade-off designs.

Result objects
--------------

Most public design functions return a dataclass from ``idaes_sdoe.models`` rather
than a bare table. Batch and Pareto workflows may return either a list of
result objects or one result object containing several designs.
