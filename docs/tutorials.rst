Usage Patterns
==============

Interactive use
---------------

The APIs are designed to work well in:

* standard Python modules
* notebooks
* interactive consoles

The package does not require a GUI layer.

Saved outputs
-------------

Most workflows produce one or more of the following artifacts:

* candidate tables
* selected design tables
* Pareto-front summaries
* reordered design tables
* plots written through ``idaes_sdoe.plotting``

These outputs are ordinary pandas tables or Plotly figures, so they can be
saved using the host application or workflow that calls the package.
