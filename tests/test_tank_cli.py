import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tdt
import tank_cli.cli as cli_module

from tank_cli import (
    DEFAULT_NEW_SAMPLING_RATE,
    DEFAULT_SMOOTHING_FRACTION,
    DEFAULT_SMOOTHING_METHOD,
    DEFAULT_TTL_START_OFFSET,
    build_parser,
    main,
)
from tank_cli.cli import _export_ols_csv


@pytest.fixture(scope="session")
def ensure_demo_data() -> Path:
    """Download TDT demo data into the repo root if missing."""
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "data" / "FiPho-180416"
    if demo_path.exists():
        return demo_path
    cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        tdt.download_demo_data()
    finally:
        os.chdir(cwd)
    return demo_path


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


def _write_json(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_single_subject_exports_csvs(tank_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    code = main(_base_args(tank_dir, out_dir))
    assert code == 0

    subject_dir = out_dir / "first"
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

    subject1 = out_dir / "first"
    subject2 = out_dir / "second"
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


def test_rejects_second_ttl_when_one_subject(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    args = [
        "--tank-dir",
        str(tank_dir),
        "--num-subjects",
        "1",
        "--first-iso",
        "iso",
        "--first-exp",
        "exp",
        "--second-ttl",
        "ttl",
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

    epoc_path = out_dir / "first" / "epoc.csv"
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

    ols_path = out_dir / "first" / "ols_processed.csv"
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


def test_default_output_root_uses_tank_name_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    monkeypatch.setattr(
        tdt,
        "read_block",
        lambda _: {"streams": {"_4054": {}, "_4654": {}}, "epocs": {}},
    )

    def fake_export_stream_csvs(
        *, subject_dir: Path, **_: object
    ) -> None:
        (subject_dir / "iso_stream.csv").write_text("", encoding="utf-8")
        (subject_dir / "exp_stream.csv").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cli_module, "_export_stream_csvs", fake_export_stream_csvs
    )

    args = [
        "--tank-dir",
        str(tank_dir),
        "--num-subjects",
        "1",
        "--first-iso",
        "_4054",
        "--first-exp",
        "_4654",
    ]
    code = main(args)
    assert code == 0

    out_root = tank_dir.parent / f"{tank_dir.name}_extract"
    assert (out_root / "first" / "iso_stream.csv").exists()
    assert (out_root / "first" / "exp_stream.csv").exists()


def test_run_metadata_json_contains_version_and_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        tdt,
        "read_block",
        lambda _: {"streams": {"_4054": {}, "_4654": {}}, "epocs": {}},
    )
    monkeypatch.setattr(
        cli_module, "_export_stream_csvs", lambda **_: None
    )
    monkeypatch.setattr(cli_module.metadata, "version", lambda _: "9.9.9")
    code = main(_base_args(tank_dir, out_dir))
    assert code == 0

    metadata_path = out_dir / "run_metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["tank_cli_version"] == "9.9.9"
    assert "parameters" in payload
    assert payload["parameters"]["num_subjects"] == 1
    assert payload["parameters"]["first_iso"] == "_4054"
    assert payload["parameters"]["first_exp"] == "_4654"
    assert payload["parameters"]["tank_dir"] == str(tank_dir)
    assert payload["parameters"]["output_dir"] == str(out_dir)


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


def test_view_streams_prints_available_stream_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()

    def fake_read_block(_: str) -> dict[str, dict[str, object]]:
        return {
            "streams": {
                "_4654": {"data": np.array([1.0]), "fs": 10.0},
                "_4054": {"data": np.array([1.0]), "fs": 10.0},
                "Wav1": {"data": np.array([1.0]), "fs": 10.0},
            }
        }

    monkeypatch.setattr(tdt, "read_block", fake_read_block)

    code = main(["--tank-dir", str(tank_dir), "--view-streams"])
    assert code == 0
    assert capsys.readouterr().out.strip().splitlines() == sorted(
        ["_4054", "_4654", "Wav1"]
    )


def test_view_epocs_prints_available_epoc_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()

    def fake_read_block(_: str) -> dict[str, dict[str, object]]:
        return {
            "streams": {"_4054": {"data": np.array([1.0]), "fs": 10.0}},
            "epocs": {
                "PtAB": {
                    "onset": np.array([1.0]),
                    "offset": np.array([2.0]),
                    "data": np.array([1.0]),
                },
                "PuAB": {
                    "onset": np.array([3.0]),
                    "offset": np.array([4.0]),
                    "data": np.array([2.0]),
                },
            },
        }

    monkeypatch.setattr(tdt, "read_block", fake_read_block)

    code = main(["--tank-dir", str(tank_dir), "--view-epocs"])
    assert code == 0
    assert capsys.readouterr().out.strip().splitlines() == sorted(
        ["PtAB", "PuAB"]
    )


def test_missing_processing_flags_without_view_streams_fails(
    tmp_path: Path,
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()

    code = main(["--tank-dir", str(tank_dir)])
    assert code == 1


def test_json_parameters_can_supply_required_processing_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    out_dir = tmp_path / "out"
    json_path = _write_json(
        tmp_path,
        {
            "num_subjects": 1,
            "first_iso": "iso",
            "first_exp": "exp",
            "output_dir": str(out_dir),
        },
    )

    monkeypatch.setattr(
        tdt,
        "read_block",
        lambda _: {"streams": {"iso": {}, "exp": {}}, "epocs": {}},
    )
    monkeypatch.setattr(
        cli_module, "_export_stream_csvs", lambda **_: None
    )

    code = main(
        ["--tank-dir", str(tank_dir), "--json", str(json_path)]
    )
    assert code == 0


def test_json_conflict_with_explicit_cli_value_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(
        tmp_path,
        {
            "num_subjects": 1,
            "first_iso": "json_iso",
            "first_exp": "exp",
        },
    )

    code = main(
        [
            "--tank-dir",
            str(tank_dir),
            "--json",
            str(json_path),
            "--first-iso",
            "cli_iso",
        ]
    )
    assert code == 1


def test_json_duplicate_equal_to_cli_value_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(
        tmp_path,
        {
            "num_subjects": 1,
            "first_iso": "iso",
            "first_exp": "exp",
        },
    )

    monkeypatch.setattr(
        tdt,
        "read_block",
        lambda _: {"streams": {"iso": {}, "exp": {}}, "epocs": {}},
    )
    monkeypatch.setattr(
        cli_module, "_export_stream_csvs", lambda **_: None
    )

    code = main(
        [
            "--tank-dir",
            str(tank_dir),
            "--json",
            str(json_path),
            "--first-iso",
            "iso",
            "--num-subjects",
            "1",
        ]
    )
    assert code == 0


def test_json_unknown_key_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(
        tmp_path,
        {
            "num_subjects": 1,
            "first_iso": "iso",
            "first_exp": "exp",
            "unexpected_key": "value",
        },
    )

    code = main(["--tank-dir", str(tank_dir), "--json", str(json_path)])
    assert code == 1


def test_json_invalid_typed_value_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(
        tmp_path,
        {"num_subjects": "not_an_int"},
    )

    code = main(["--tank-dir", str(tank_dir), "--json", str(json_path)])
    assert code == 1


def test_json_store_true_booleans_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(
        tmp_path,
        {
            "num_subjects": 1,
            "first_iso": "iso",
            "first_exp": "exp",
            "run_ols": True,
            "export_epoc_csv": True,
            "epoc": "PtAB",
        },
    )

    calls = {"epoc": 0, "ols": 0}

    monkeypatch.setattr(
        tdt,
        "read_block",
        lambda _: {
            "streams": {"iso": {}, "exp": {}},
            "epocs": {"PtAB": {}},
        },
    )
    monkeypatch.setattr(
        cli_module, "_export_stream_csvs", lambda **_: None
    )

    def fake_export_epoc_csv(**_: object) -> None:
        calls["epoc"] += 1

    def fake_export_ols_csv(**_: object) -> None:
        calls["ols"] += 1

    monkeypatch.setattr(cli_module, "_export_epoc_csv", fake_export_epoc_csv)
    monkeypatch.setattr(cli_module, "_export_ols_csv", fake_export_ols_csv)

    code = main(["--tank-dir", str(tank_dir), "--json", str(json_path)])
    assert code == 0
    assert calls["epoc"] == 1
    assert calls["ols"] == 1


def test_json_missing_file_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    missing_json = tmp_path / "missing.json"

    code = main(["--tank-dir", str(tank_dir), "--json", str(missing_json)])
    assert code == 1


def test_json_invalid_content_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = tmp_path / "params.json"
    json_path.write_text("{invalid_json", encoding="utf-8")

    code = main(["--tank-dir", str(tank_dir), "--json", str(json_path)])
    assert code == 1


def test_json_top_level_non_object_fails(tmp_path: Path) -> None:
    tank_dir = tmp_path / "dummy_tank"
    tank_dir.mkdir()
    json_path = _write_json(tmp_path, [1, 2, 3])

    code = main(["--tank-dir", str(tank_dir), "--json", str(json_path)])
    assert code == 1


def test_epoc_ttl_source_writes_marker_not_ttl_stream(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + ["--first-ttl", "PtAB"]
    code = main(args)
    assert code == 0

    subject_dir = out_dir / "first"
    assert not (subject_dir / "ttl_stream.csv").exists()
    marker_path = subject_dir / "epoc_marker.csv"
    assert marker_path.exists()
    marker_df = pd.read_csv(marker_path)
    assert list(marker_df.columns) == [
        "onset",
        "offset",
        "data",
        "first_onset_for_ttl",
    ]
    assert marker_df["first_onset_for_ttl"].astype(bool).sum() == 1


def test_stream_ttl_source_writes_stream_not_marker(
    tank_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    args = _base_args(tank_dir, out_dir) + ["--first-ttl", "_4054"]
    code = main(args)
    assert code == 0

    subject_dir = out_dir / "first"
    assert (subject_dir / "ttl_stream.csv").exists()
    assert not (subject_dir / "epoc_marker.csv").exists()


def test_unknown_ttl_source_lists_streams_and_epocs(
    tank_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out_dir = tmp_path / "out"
    caplog.set_level(logging.ERROR, logger="tank_cli.cli")

    args = _base_args(tank_dir, out_dir) + ["--first-ttl", "missing_ttl"]
    code = main(args)
    assert code == 1
    assert "Unknown ttl source 'missing_ttl'" in caplog.text
    assert "Available streams:" in caplog.text
    assert "Available epocs:" in caplog.text


def test_ttl_name_collision_prefers_stream() -> None:
    source_type, source_name = cli_module._resolve_ttl_source(
        ttl_name="ttl",
        available_stream_set={"ttl", "iso"},
        available_epoc_set={"ttl", "PtAB"},
    )
    assert source_type == "stream"
    assert source_name == "ttl"


def test_export_ols_rejects_mismatched_stream_fs(tmp_path: Path) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "ttl": {"data": np.array([0.0, 1.0], dtype=np.float32), "fs": 5.0},
        },
        "epocs": {},
    }

    with pytest.raises(ValueError, match="sampling rates must match"):
        _export_ols_csv(
            row_data=row_data,
            subject_dir=tmp_path,
            iso_stream="iso",
            exp_stream="exp",
            ttl_source_type="stream",
            ttl_source_name="ttl",
            smoothing_method=DEFAULT_SMOOTHING_METHOD,
            smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
            new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
            ttl_filtering=False,
            ttl_start_offset=DEFAULT_TTL_START_OFFSET,
        )


def test_export_ols_allows_matching_stream_fs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "ttl": {"data": np.array([0.0, 1.0], dtype=np.float32), "fs": 10.0},
        },
        "epocs": {},
    }

    def fake_run_ols_processing(**_: object) -> dict[str, np.ndarray]:
        return {
            "time_array": np.array([0.0, 0.1], dtype=np.float64),
            "smoothed_exp": np.array([1.0, 2.0], dtype=np.float64),
            "smoothed_iso": np.array([1.0, 2.0], dtype=np.float64),
            "delta_f": np.array([0.1, 0.2], dtype=np.float64),
            "delta_f_norm": np.array([0.01, 0.02], dtype=np.float64),
            "delta_f_zscore": np.array([-1.0, 1.0], dtype=np.float64),
        }

    monkeypatch.setattr(
        cli_module,
        "run_ols_processing",
        fake_run_ols_processing,
    )

    _export_ols_csv(
        row_data=row_data,
        subject_dir=tmp_path,
        iso_stream="iso",
        exp_stream="exp",
        ttl_source_type="stream",
        ttl_source_name="ttl",
        smoothing_method=DEFAULT_SMOOTHING_METHOD,
        smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
        new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
        ttl_filtering=False,
        ttl_start_offset=DEFAULT_TTL_START_OFFSET,
    )
    assert (tmp_path / "ols_processed.csv").exists()


def test_export_ols_epoc_ttl_filtering_trims_from_first_onset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
        },
        "epocs": {
            "PtAB": {
                "onset": np.array([1.2, 3.0], dtype=np.float64),
                "offset": np.array([1.4, 3.2], dtype=np.float64),
                "data": np.array([1.0, 2.0], dtype=np.float64),
            }
        },
    }
    called: dict[str, object] = {}

    def fake_run_ols_processing(**kwargs: object) -> dict[str, np.ndarray]:
        called.update(kwargs)
        signal = np.arange(0.0, 6.0, 0.5, dtype=np.float64)
        return {
            "time_array": signal.copy(),
            "smoothed_exp": signal.copy(),
            "smoothed_iso": signal.copy(),
            "delta_f": signal.copy(),
            "delta_f_norm": signal.copy(),
            "delta_f_zscore": signal.copy(),
        }

    monkeypatch.setattr(
        cli_module,
        "run_ols_processing",
        fake_run_ols_processing,
    )

    _export_ols_csv(
        row_data=row_data,
        subject_dir=tmp_path,
        iso_stream="iso",
        exp_stream="exp",
        ttl_source_type="epoc",
        ttl_source_name="PtAB",
        smoothing_method=DEFAULT_SMOOTHING_METHOD,
        smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
        new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
        ttl_filtering=True,
        ttl_start_offset=1,
    )

    assert called["ttl_filtering"] is False
    assert called["ttl_start_offset"] == 0
    output_df = pd.read_csv(tmp_path / "ols_processed.csv")
    assert output_df["time"].iloc[0] == pytest.approx(1.0)


def test_export_ols_epoc_source_without_filtering_does_not_epoc_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
        },
        "epocs": {
            "PtAB": {
                "onset": np.array([1.2], dtype=np.float64),
                "offset": np.array([1.4], dtype=np.float64),
                "data": np.array([1.0], dtype=np.float64),
            }
        },
    }
    called: dict[str, object] = {}

    def fake_run_ols_processing(**kwargs: object) -> dict[str, np.ndarray]:
        called.update(kwargs)
        signal = np.arange(0.0, 2.0, 0.5, dtype=np.float64)
        return {
            "time_array": signal.copy(),
            "smoothed_exp": signal.copy(),
            "smoothed_iso": signal.copy(),
            "delta_f": signal.copy(),
            "delta_f_norm": signal.copy(),
            "delta_f_zscore": signal.copy(),
        }

    monkeypatch.setattr(
        cli_module,
        "run_ols_processing",
        fake_run_ols_processing,
    )

    _export_ols_csv(
        row_data=row_data,
        subject_dir=tmp_path,
        iso_stream="iso",
        exp_stream="exp",
        ttl_source_type="epoc",
        ttl_source_name="PtAB",
        smoothing_method=DEFAULT_SMOOTHING_METHOD,
        smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
        new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
        ttl_filtering=False,
        ttl_start_offset=2,
    )

    assert called["ttl_filtering"] is False
    assert called["ttl_start_offset"] == 2
    output_df = pd.read_csv(tmp_path / "ols_processed.csv")
    assert output_df["time"].iloc[0] == pytest.approx(0.0)


def test_export_ols_epoc_ttl_filtering_rejects_empty_onset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
        },
        "epocs": {
            "PtAB": {
                "onset": np.array([], dtype=np.float64),
                "offset": np.array([], dtype=np.float64),
                "data": np.array([], dtype=np.float64),
            }
        },
    }

    def fake_run_ols_processing(**_: object) -> dict[str, np.ndarray]:
        signal = np.arange(0.0, 2.0, 0.5, dtype=np.float64)
        return {
            "time_array": signal.copy(),
            "smoothed_exp": signal.copy(),
            "smoothed_iso": signal.copy(),
            "delta_f": signal.copy(),
            "delta_f_norm": signal.copy(),
            "delta_f_zscore": signal.copy(),
        }

    monkeypatch.setattr(
        cli_module,
        "run_ols_processing",
        fake_run_ols_processing,
    )

    with pytest.raises(ValueError, match="empty onset array"):
        _export_ols_csv(
            row_data=row_data,
            subject_dir=tmp_path,
            iso_stream="iso",
            exp_stream="exp",
            ttl_source_type="epoc",
            ttl_source_name="PtAB",
            smoothing_method=DEFAULT_SMOOTHING_METHOD,
            smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
            new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
            ttl_filtering=True,
            ttl_start_offset=0,
        )


def test_export_ols_epoc_ttl_filtering_rejects_onset_outside_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_data = {
        "streams": {
            "iso": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
            "exp": {"data": np.array([1.0, 2.0], dtype=np.float32), "fs": 10.0},
        },
        "epocs": {
            "PtAB": {
                "onset": np.array([99.0], dtype=np.float64),
                "offset": np.array([100.0], dtype=np.float64),
                "data": np.array([1.0], dtype=np.float64),
            }
        },
    }

    def fake_run_ols_processing(**_: object) -> dict[str, np.ndarray]:
        signal = np.arange(0.0, 2.0, 0.5, dtype=np.float64)
        return {
            "time_array": signal.copy(),
            "smoothed_exp": signal.copy(),
            "smoothed_iso": signal.copy(),
            "delta_f": signal.copy(),
            "delta_f_norm": signal.copy(),
            "delta_f_zscore": signal.copy(),
        }

    monkeypatch.setattr(
        cli_module,
        "run_ols_processing",
        fake_run_ols_processing,
    )

    with pytest.raises(ValueError, match="outside processed OLS time range"):
        _export_ols_csv(
            row_data=row_data,
            subject_dir=tmp_path,
            iso_stream="iso",
            exp_stream="exp",
            ttl_source_type="epoc",
            ttl_source_name="PtAB",
            smoothing_method=DEFAULT_SMOOTHING_METHOD,
            smoothing_fraction=DEFAULT_SMOOTHING_FRACTION,
            new_sampling_rate=DEFAULT_NEW_SAMPLING_RATE,
            ttl_filtering=True,
            ttl_start_offset=0,
        )
