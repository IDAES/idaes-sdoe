# idaes-sdoe

[![Documentation Status](https://readthedocs.org/projects/idaes-sdoe/badge/?version=latest)](https://idaes-sdoe.readthedocs.io/en/latest/?badge=latest)

`idaes-sdoe` is a Python package for design of experiments in process systems
engineering.

Documentation: https://idaes-sdoe.readthedocs.io

`idaes-sdoe` is part of the IDAES integrated software platform, specifically
the Institute for the Design of Advanced Energy Systems Process Systems
Engineering Framework (IDAES PSE Framework).

## Install

Install from source with an editable install. First clone the repository:

```bash
git clone https://github.com/IDAES/idaes-sdoe.git
cd idaes-sdoe
```

Then install into a Conda environment:

```bash
conda create -n idaes-sdoe python=3.11
conda activate idaes-sdoe
pip install -e .
```

To use an existing environment, run only the `pip install -e .` step.

### Optional extras

Add extras in brackets to install optional tooling (combine as needed):

- `notebook` — Jupyter Notebook
- `marimo` — the marimo notebook interface
- `docs` — Sphinx and the documentation theme (see [Development](#development))
- `dev` — the test stack
- `all-dev` — all of the above

```bash
pip install -e ".[notebook,marimo]"   # e.g. both notebook interfaces
pip install -e ".[all-dev]"           # everything for development
```

### Notebook examples

After installing the notebook extras, launch the Jupyter example:

```bash
jupyter notebook examples/example-uniform-5d.ipynb
```

Open the interactive marimo notebook (`edit` for editable cells, `run` for the
app view):

```bash
marimo edit examples/example-uniform-5d-marimo.py
marimo run examples/example-uniform-5d-marimo.py
```

## Quick start

The example below uses a bundled candidate set from the repository.

```python
from pathlib import Path

from idaes_sdoe import ColumnRoles, load_csv, prepare_design_setup
from idaes_sdoe.design import design_uniform_batch

candidate = load_csv(Path("examples/supporting_data/SDOE_Ex1_Candidates.csv"))

setup = prepare_design_setup(
    candidate=candidate,
    roles=ColumnRoles(inputs=["X1", "X2"]),
)

results = design_uniform_batch(
    setup=setup,
    design_sizes=[8, 9, 10],
    num_restarts=1000,
    mode="minimax",
    random_state=7,
)

design8 = results[0]
print(design8.criterion_value)
print(design8.design.head())
```

## Usage

The package is designed for direct use from Python modules, notebooks, and
interactive sessions. Typical workflow:

- load candidate data into pandas tables
- define `ColumnRoles`
- call `prepare_design_setup()`
- run a design method from `idaes_sdoe.design`
- inspect the returned result
- optionally apply plotting, candidate generation, imputation, or run ordering

The main public surface is split across `idaes_sdoe`, `idaes_sdoe.design`,
`idaes_sdoe.ordering`, and `idaes_sdoe.extras`.

## Layout

- `src/idaes_sdoe/design`: core design algorithms
- `src/idaes_sdoe/extras`: candidate-generation and imputation helpers
- `src/idaes_sdoe/plotting.py`: Plotly plotting helpers
- `examples`: runnable Jupyter and marimo workflows
- `tests`: standalone test suite

## Development

Set up a development environment with all developer extras:

```bash
conda create -n idaes-sdoe python=3.11
conda activate idaes-sdoe
pip install -e ".[all-dev]"
```

Run the test suite:

```bash
pytest
```

Build the documentation locally:

```bash
python -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser (on macOS,
`open docs/_build/html/index.html`).

## License

See [LICENSE.md](LICENSE.md) and [COPYRIGHT.md](COPYRIGHT.md).

## Contributing

By contributing to this repository, you are agreeing to all the terms set out
in the [LICENSE.md](LICENSE.md) and [COPYRIGHT.md](COPYRIGHT.md) files in this
directory.

## Contact

For questions about `idaes-sdoe`, contact Xiangyu Bi at xbi@lbl.gov.
