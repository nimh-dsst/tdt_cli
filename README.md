# tdt-cli

`tdt-cli` is a command-line package for exporting stream CSVs and optional OLS dF/F outputs directly from TDT tank folders.

## Package Installation

Python requirement: `>=3.11`

### Recommended: UV workflow

```bash
# from repo root
uv sync
uv run tank-cli --help
```

Run as a module if preferred:

```bash
uv run python -m tank_cli --help
```

### pip / venv workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
tank-cli --help
```

### Quick usage example

```bash
tank-cli \
  --tank-dir /path/to/TDT_TANK \
  --num-subjects 1 \
  --first-iso _4054 \
  --first-exp _4654 \
  --run-ols
```

By default, outputs are written under:

```text
<tank-dir>/archiveflow_cli_output/<subject_id>_<timestamp>/
```

Common output files:

- `iso_stream.csv`
- `exp_stream.csv`
- `ttl_stream.csv` (if TTL stream provided)
- `epoc.csv` (if `--export-epoc-csv`)
- `ols_processed.csv` (if `--run-ols`)

## Dependencies (TDT-First)

This package depends on scientific Python libraries, but the **critical dependency is `tdt`**.

### Core dependency: `tdt`

- The CLI reads tank folders via `tdt.read_block(...)`.
- If `tdt` is missing or incompatible, the package cannot parse tank streams.
- Test setup uses `tdt.download_demo_data()` to fetch sample data.

Verify availability:

```bash
python -c "import tdt; print(tdt.__version__)"
```

### Other runtime dependencies

- `numpy`: numeric array handling and output conversion
- `pandas`: CSV creation/export
- `scipy`: interpolation utilities in preprocessing
- `statsmodels`: LOWESS smoothing for signal processing

Runtime validation also enforces key assumptions, including matching sampling rates between iso/exp/ttl streams when TTL is included in OLS export.

## Developer Notes (UV, pyproject.toml, and requirements.txt)

This repository is managed with **UV + `pyproject.toml`**.

### Source of truth

- Project metadata and dependencies live in `pyproject.toml`.
- Locked resolution lives in `uv.lock`.
- Dependency groups (for example, tests) are defined in `pyproject.toml`.

### Why UV here

- Fast and reproducible resolution
- Lockfile-based environment consistency
- First-class dependency group support (`--group tests`)

### Standard dev commands

```bash
uv sync
uv run tank-cli --help
uv run --group tests pytest -q
```

### About `requirements.txt`

- `requirements.txt` is **not** the canonical dependency definition in this repo.
- If another tool needs it, export from the lockfile instead of maintaining it manually:

```bash
uv export --format requirements.txt -o requirements.txt
```

### Using non-UV installers

- `pip` can install the project (`pip install .` or `pip install -e .`), but it does not replace UV's lockfile workflow.
- If contributing to this repo, prefer UV commands so dependency resolution matches project expectations.
