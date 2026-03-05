import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import tdt

from tank_cli import (
    DEFAULT_NEW_SAMPLING_RATE,
    DEFAULT_SMOOTHING_FRACTION,
    DEFAULT_SMOOTHING_METHOD,
    DEFAULT_TTL_START_OFFSET,
    build_parser,
    main,
)


@pytest.fixture(scope="session")
def ensure_demo_data() -> Path:
    """Download TDT demo data into the repo root if missing."""
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        tdt.download_demo_data()
    finally:
        os.chdir(cwd)
    return repo_root / "data" / "FiPho-180416"


@pytest.fixture
def tank_dir(tmp_path: Path, ensure_demo_data: Path) -> Path:
    """Copy demo TDT tank into a parseable tank-name directory."""
    src = ensure_demo_data
    dest = tmp_path / "123456_M1-250101-101500"
    shutil.copytree(src, dest)
    return dest


def _base_args(tank_dir: Path, out_dir: Path) -> list[str]:
    return [
        "--tank-dir",
        str(tank_dir),
        "--num-subjects",
        "1",
        "--first-iso",
        "_4054",
        "--first-exp",
        "_4654",
        "--output-dir",
        str(out_dir),
    ]


def test_single_subject_exports_csvs(tank_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    code = main(_base_args(tank_dir, out_dir))
    assert code == 0

    subject_dir = out_dir / "123456_M1_20250101-101500"
    assert (subject_dir / "iso_stream.csv").exists()
    assert (subject_dir / "exp_stream.csv").exists()
    assert not (subject_dir / "ttl_stream.csv").exists()


def test_num_subjects_two_requires_second_streams(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir)
    args[3] = "2"
    code = main(args)
    assert code == 1


def test_num_subjects_two_exports_both_subject_dirs(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir)
    args[3] = "2"
    args += [
        "--second-iso",
        "_4054",
        "--second-exp",
        "_4654",
    ]
    code = main(args)
    assert code == 0

    subject1 = out_dir / "subject1_20250101-101500"
    subject2 = out_dir / "subject2_20250101-101500"
    assert (subject1 / "iso_stream.csv").exists()
    assert (subject1 / "exp_stream.csv").exists()
    assert (subject2 / "iso_stream.csv").exists()
    assert (subject2 / "exp_stream.csv").exists()


def test_rejects_second_subject_flags_when_one_subject(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + [
        "--second-iso",
        "_4054",
        "--second-exp",
        "_4654",
    ]
    code = main(args)
    assert code == 1


def test_export_epoc_requires_epoc_name(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + ["--export-epoc-csv"]
    code = main(args)
    assert code == 1


def test_export_epoc_csv_writes_expected_columns(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + [
        "--epoc",
        "PtAB",
        "--export-epoc-csv",
    ]
    code = main(args)
    assert code == 0

    epoc_path = out_dir / "123456_M1_20250101-101500" / "epoc.csv"
    epoc_df = pd.read_csv(epoc_path)
    assert list(epoc_df.columns) == ["index", "onset", "offset"]
    assert len(epoc_df) > 0


def test_run_ols_writes_merged_processed_csv(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + ["--run-ols"]
    code = main(args)
    assert code == 0

    ols_path = out_dir / "123456_M1_20250101-101500" / "ols_processed.csv"
    assert ols_path.exists()
    ols_df = pd.read_csv(ols_path)
    assert list(ols_df.columns) == [
        "time",
        "smoothed_exp",
        "smoothed_iso",
        "delta_f",
        "delta_f_norm",
        "delta_f_zscore",
    ]
    assert len(ols_df) > 0


def test_ols_defaults_match_pipeline() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tank-dir",
            "/tmp/fake",
            "--num-subjects",
            "1",
            "--first-iso",
            "iso",
            "--first-exp",
            "exp",
        ]
    )
    assert args.smoothing_method == DEFAULT_SMOOTHING_METHOD
    assert args.smoothing_fraction == DEFAULT_SMOOTHING_FRACTION
    assert args.new_sampling_rate == DEFAULT_NEW_SAMPLING_RATE
    assert args.ttl_filtering is False
    assert args.ttl_start_offset == DEFAULT_TTL_START_OFFSET


def test_ttl_filtering_requires_ttl_stream(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + ["--ttl-filtering", "--run-ols"]
    code = main(args)
    assert code == 1


def test_unknown_stream_name_fails(tank_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir)
    args[7] = "not_a_stream"
    code = main(args)
    assert code == 1
