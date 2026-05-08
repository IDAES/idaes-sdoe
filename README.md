# idaes-sdoe

`idaes-sdoe` is a Python package for design of experiments in process systems
engineering.

`idaes-sdoe` is part of the IDAES integrated software platform, specifically
the Institute for the Design of Advanced Energy Systems Process Systems
Engineering Framework (IDAES PSE Framework).

## Install

Recommended setup: Conda environment

```bash
conda create -n idaes-sdoe python=3.11
conda activate idaes-sdoe
pip install -e .
```

Direct install into an existing environment:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[docs]"
pip install -e ".[mars]"
pip install -e ".[notebook]"
pip install -e ".[dev]"
```

Use `docs` for local documentation builds, `mars` if you need the
`method="mars"` response-surface option in imputation, `notebook` if you want
the notebook stack managed with the project, and `dev` for the test stack.
The `docs` extra installs Sphinx plus the HTML theme and code-block helpers
used by the site.

To use notebooks in the recommended Conda environment:

```bash
conda activate idaes-sdoe
pip install -e ".[notebook]"
jupyter notebook
```

To build the docs locally:

```bash
conda activate idaes-sdoe
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
```

On systems without `open`, open `docs/_build/html/index.html` in a browser
directly.

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
- `tests`: standalone test suite

## License

See [LICENSE.md](LICENSE.md) and [COPYRIGHT.md](COPYRIGHT.md).

## Contributing

By contributing to this repository, you are agreeing to all the terms set out
in the [LICENSE.md](LICENSE.md) and [COPYRIGHT.md](COPYRIGHT.md) files in this
directory.

## Contact

For questions about `idaes-sdoe`, contact Xiangyu Bi at xbi@lbl.gov.
