PSE-DoE Documentation
=====================

``idaes-sdoe`` is a Python package for Design of Experiments in Process Systems Engineering. Currently it mainly provides tools for sequential design
of experiments (SDOE).

This site combines workflow-oriented guides with an API reference generated
from the package source.

Use the guide pages to understand the workflow and the reference pages to look
up the public objects exported by the package.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   overview
   concepts
   workflows
   tutorials
   modules

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Local build
-----------

.. code-block:: bash

   pip install -e ".[docs]"
   python -m sphinx -b html docs docs/_build/html
   open docs/_build/html/index.html

On systems without ``open``, open ``docs/_build/html/index.html`` in a browser
directly.
