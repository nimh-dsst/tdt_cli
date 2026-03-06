# tdt-cli

`tdt-cli` is a command-line package for exporting stream CSVs and optional OLS dF/F outputs directly from TDT tank folders.

## Package Installation

Python requirement: `>=3.11`

Use the command block for your OS.

### Recommended: UV workflow

#### Unix/macOS

```bash
# from repo root
uv sync
uv run tank-cli --help
```

```bash
uv run python -m tank_cli --help
```

#### Windows (PowerShell)

```powershell
# from repo root
uv sync
uv run tank-cli --help
```

```powershell
uv run python -m tank_cli --help
```

#### Windows (cmd.exe)

```cmd
REM from repo root
uv sync
uv run tank-cli --help
uv run python -m tank_cli --help
```

### pip / venv workflow

#### Unix/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
tank-cli --help
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
tank-cli --help
```

#### Windows (cmd.exe)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install -e .
tank-cli --help
```

### Quick usage example

#### Unix/macOS

```bash
tank-cli \
  --tank-dir /path/to/TDT_TANK \
  --num-subjects 1 \
  --first-iso _4054 \
  --first-exp _4654 \
  --run-ols
```

#### Windows (PowerShell)

```powershell
tank-cli `
  --tank-dir C:\path\to\TDT_TANK `
  --num-subjects 1 `
  --first-iso _4054 `
  --first-exp _4654 `
  --run-ols
```

#### Windows (cmd.exe)

```cmd
tank-cli ^
  --tank-dir C:\path\to\TDT_TANK ^
  --num-subjects 1 ^
  --first-iso _4054 ^
  --first-exp _4654 ^
  --run-ols
```

By default, outputs are written under:

```text
<tank-dir-parent>/<tank-dir-name>_extract/
```

If `--output-dir` is provided, that path is used as the root folder instead.

Inside that folder:

- `run_metadata.json` (resolved parameters + `tank_cli_version`)
- `first/`
- `second/` (only when `--num-subjects 2`)

Common files in each subject folder:

- `iso_stream.csv`
- `exp_stream.csv`
- `ttl_stream.csv` (if TTL stream provided)
- `epoc.csv` (if `--export-epoc-csv`)
- `ols_processed.csv` (if `--run-ols`)

### JSON parameter file usage

You can provide most run parameters from a JSON file and still set tank path
explicitly on the CLI:

```bash
tank-cli --json my_parameters.json --tank-dir /path/to/TDT_TANK
```

Example `my_parameters.json`:

```json
{
  "num_subjects": 1,
  "first_iso": "_4054",
  "first_exp": "_4654",
  "first_ttl": "None",
  "epoc": "None",
  "export_epoc_csv": false,
  "run_ols": true,
  "smoothing_method": "Downsample then smooth",
  "smoothing_fraction": 0.002,
  "new_sampling_rate": 10.0,
  "ttl_filtering": false,
  "ttl_start_offset": 0
}
```

Rules:

- JSON keys must be `snake_case` argparse destination names.
- Unknown keys fail fast with an error.
- If a value is provided in both JSON and CLI:
  - same value: allowed
  - different values: error

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

```powershell
python -c "import tdt; print(tdt.__version__)"
```

```cmd
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

#### Unix/macOS

```bash
uv sync
uv run tank-cli --help
uv run --group tests pytest -q
```

#### Windows (PowerShell)

```powershell
uv sync
uv run tank-cli --help
uv run --group tests pytest -q
```

#### Windows (cmd.exe)

```cmd
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

```powershell
uv export --format requirements.txt -o requirements.txt
```

```cmd
uv export --format requirements.txt -o requirements.txt
```

### Using non-UV installers

- `pip` can install the project (`pip install .` or `pip install -e .`), but it does not replace UV's lockfile workflow.
- If contributing to this repo, prefer UV commands so dependency resolution matches project expectations.
