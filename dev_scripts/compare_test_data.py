#!/usr/bin/env python3
"""Compare generated tank_cli output against extracted subject test data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OLS_METHOD_NAME = "OLS dF/F"
STREAM_FILES = ("iso_stream.csv", "exp_stream.csv")
ORDER_TO_DIR = {"first": "first", "second": "second"}
OLS_COLUMN_TO_PARAM_KEY = {
    "smoothed_exp": "smoothed exp data",
    "smoothed_iso": "smoothed iso data",
    "delta_f": "delta f",
    "delta_f_norm": "delta f norm",
    "delta_f_zscore": "delta f zscore",
}
SCALAR_FIELDS = ("exp mean", "iso mean", "signal ratio")


class ComparisonError(RuntimeError):
    """Raised when setup or comparison validation fails."""


@dataclass
class SubjectContext:
    subject_dir: Path
    generated_dir_name: str
    subject_id: str
    streams_payload: dict[str, Any]
    smoothing_settings: dict[str, Any]
    processed_data: dict[str, Any]


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tank_cli on test_data tank input and compare generated output "
            "with extracted subject folders."
        )
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path("test_data"),
        help="Path to test_data directory (default: test_data).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for numeric comparisons (default: 1e-6).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for numeric comparisons (default: 1e-8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where tank_cli output will be generated. If omitted, a "
            "temporary directory is created."
        ),
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help=(
            "Keep temporary output directory when --output-dir is omitted. "
            "Ignored when --output-dir is provided."
        ),
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ComparisonError(f"Expected object in JSON file: {path}")
    return payload


def find_tank_dir(test_data_dir: Path) -> Path:
    tsq_paths = sorted(test_data_dir.rglob("*.tsq"))
    if not tsq_paths:
        raise ComparisonError(
            f"No .tsq files found under '{test_data_dir}'."
        )

    parent_dirs = {path.parent.resolve() for path in tsq_paths}
    if len(parent_dirs) != 1:
        resolved = "\n".join(f"  - {path}" for path in sorted(parent_dirs))
        raise ComparisonError(
            "Expected exactly one tank directory parent for .tsq files, "
            f"found {len(parent_dirs)}:\n{resolved}"
        )
    return next(iter(parent_dirs))


def find_subject_dirs(test_data_dir: Path) -> list[Path]:
    stream_jsons = sorted(test_data_dir.rglob("streams.json"))
    subject_dirs = [
        path.parent for path in stream_jsons if (path.parent / "parameters.json").exists()
    ]
    unique_dirs = sorted({path.resolve() for path in subject_dirs})
    if len(unique_dirs) != 2:
        formatted = "\n".join(f"  - {path}" for path in unique_dirs)
        raise ComparisonError(
            "Expected exactly two extracted subject directories (with both "
            f"streams.json and parameters.json), found {len(unique_dirs)}.\n"
            f"{formatted}"
        )
    return unique_dirs


def parse_order(order_value: Any, path: Path) -> str:
    if not isinstance(order_value, str):
        raise ComparisonError(
            f"'order' in {path} must be a string ('First' or 'Second')."
        )
    normalized = order_value.strip().lower()
    if normalized not in ORDER_TO_DIR:
        raise ComparisonError(
            f"'order' in {path} must be 'First' or 'Second'; got '{order_value}'."
        )
    return ORDER_TO_DIR[normalized]


def get_ols_method(parameters_payload: dict[str, Any], path: Path) -> dict[str, Any]:
    methods = parameters_payload.get("analysis methods")
    if not isinstance(methods, list):
        raise ComparisonError(
            f"Missing or invalid 'analysis methods' array in {path}."
        )
    for method in methods:
        if isinstance(method, dict) and method.get("name") == OLS_METHOD_NAME:
            return method
    raise ComparisonError(
        f"Analysis method '{OLS_METHOD_NAME}' not found in {path}."
    )


def load_subject_context(subject_dir: Path) -> SubjectContext:
    streams_path = subject_dir / "streams.json"
    parameters_path = subject_dir / "parameters.json"

    streams_payload = load_json(streams_path)
    parameters_payload = load_json(parameters_path)
    order_dir = parse_order(streams_payload.get("order"), streams_path)
    subject_id = str(streams_payload.get("subject_id", subject_dir.name))

    ols_method = get_ols_method(parameters_payload, parameters_path)
    parameters = ols_method.get("parameters")
    processed_data = ols_method.get("processed data")
    if not isinstance(parameters, dict):
        raise ComparisonError(
            f"'{OLS_METHOD_NAME}' parameters is missing/invalid in {parameters_path}."
        )
    if not isinstance(processed_data, dict):
        raise ComparisonError(
            f"'{OLS_METHOD_NAME}' processed data is missing/invalid in {parameters_path}."
        )
    smoothing_settings = parameters.get("smoothing")
    if not isinstance(smoothing_settings, dict):
        raise ComparisonError(
            f"'{OLS_METHOD_NAME}' smoothing settings are missing in {parameters_path}."
        )

    return SubjectContext(
        subject_dir=subject_dir,
        generated_dir_name=order_dir,
        subject_id=subject_id,
        streams_payload=streams_payload,
        smoothing_settings=smoothing_settings,
        processed_data=processed_data,
    )


def validate_and_index_subjects(contexts: list[SubjectContext]) -> dict[str, SubjectContext]:
    indexed: dict[str, SubjectContext] = {}
    for ctx in contexts:
        if ctx.generated_dir_name in indexed:
            raise ComparisonError(
                f"Duplicate subject order '{ctx.generated_dir_name}' found in "
                f"{ctx.subject_dir} and {indexed[ctx.generated_dir_name].subject_dir}."
            )
        indexed[ctx.generated_dir_name] = ctx
    for required in ("first", "second"):
        if required not in indexed:
            raise ComparisonError(
                "Missing subject order in extracted data. Required orders: "
                "'First' and 'Second'."
            )
    return indexed


def resolve_ttl_source(streams_payload: dict[str, Any]) -> str:
    ttl_stream = str(streams_payload.get("ttl_stream", "None"))
    if ttl_stream != "None":
        return ttl_stream
    epoc = str(streams_payload.get("epoc", "None"))
    return epoc if epoc else "None"


def extract_cli_smoothing_args(
    first: SubjectContext, second: SubjectContext
) -> tuple[str, float, float, bool, int]:
    if first.smoothing_settings != second.smoothing_settings:
        raise ComparisonError(
            "Smoothing settings differ between subjects. A single 2-subject "
            f"tank_cli run requires one shared setting set.\n"
            f"first: {first.smoothing_settings}\n"
            f"second: {second.smoothing_settings}"
        )

    smoothing = first.smoothing_settings
    try:
        method = str(smoothing["method"])
        fraction = float(smoothing["lowess fraction"])
        new_rate = float(smoothing["new sampling rate"])
        ttl_filtered = bool(smoothing["ttl filtered"])
        ttl_start_offset = int(smoothing["ttl start offset"])
    except KeyError as exc:
        raise ComparisonError(
            "Missing smoothing key in parameters.json: "
            f"{exc.args[0]!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ComparisonError(
            "Invalid smoothing value type in parameters.json."
        ) from exc

    return method, fraction, new_rate, ttl_filtered, ttl_start_offset


def build_tank_cli_command(
    *,
    tank_dir: Path,
    output_dir: Path,
    first: SubjectContext,
    second: SubjectContext,
    smoothing_method: str,
    smoothing_fraction: float,
    new_sampling_rate: float,
    ttl_filtering: bool,
    ttl_start_offset: int,
) -> list[str]:
    first_iso = str(first.streams_payload.get("iso_stream", ""))
    first_exp = str(first.streams_payload.get("exp_stream", ""))
    second_iso = str(second.streams_payload.get("iso_stream", ""))
    second_exp = str(second.streams_payload.get("exp_stream", ""))
    first_ttl = resolve_ttl_source(first.streams_payload)
    second_ttl = resolve_ttl_source(second.streams_payload)

    if not first_iso or not first_exp or not second_iso or not second_exp:
        raise ComparisonError("Missing iso/exp stream names in streams.json.")
    if ttl_filtering and (first_ttl == "None" or second_ttl == "None"):
        raise ComparisonError(
            "TTL filtering is enabled in parameters.json, but resolved TTL "
            "source is 'None' for at least one subject."
        )

    command = [
        sys.executable,
        "-m",
        "tank_cli",
        "--tank-dir",
        str(tank_dir),
        "--num-subjects",
        "2",
        "--first-iso",
        first_iso,
        "--first-exp",
        first_exp,
        "--first-ttl",
        first_ttl,
        "--second-iso",
        second_iso,
        "--second-exp",
        second_exp,
        "--second-ttl",
        second_ttl,
        "--run-ols",
        "--smoothing-method",
        smoothing_method,
        "--smoothing-fraction",
        str(smoothing_fraction),
        "--new-sampling-rate",
        str(new_sampling_rate),
        "--ttl-start-offset",
        str(ttl_start_offset),
        "--output-dir",
        str(output_dir),
    ]
    if ttl_filtering:
        command.append("--ttl-filtering")
    return command


def run_tank_cli(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ComparisonError(
            "tank_cli run failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def _max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        diffs = np.abs(actual - expected)
    finite = diffs[np.isfinite(diffs)]
    if finite.size:
        return float(np.max(finite))
    if np.any(np.isinf(diffs)):
        return math.inf
    return float("nan")


def _compare_numeric_arrays(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    rtol: float,
    atol: float,
) -> list[str]:
    close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if np.all(close):
        return []
    bad_indices = np.flatnonzero(~close)
    first_idx = int(bad_indices[0])
    max_abs = _max_abs_diff(actual, expected)
    return [
        (
            f"{label}: max_abs_diff={max_abs:.6g}, "
            f"first_mismatch_index={first_idx}, "
            f"actual={actual[first_idx]!r}, expected={expected[first_idx]!r}"
        )
    ]


def _compare_scalar(
    *,
    actual: float,
    expected: float,
    label: str,
    rtol: float,
    atol: float,
) -> list[str]:
    if np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        return []
    abs_diff = abs(actual - expected)
    return [
        (
            f"{label}: abs_diff={abs_diff:.6g}, "
            f"actual={actual!r}, expected={expected!r}"
        )
    ]


def compare_stream_csv(
    *,
    generated_path: Path,
    expected_path: Path,
    rtol: float,
    atol: float,
) -> CheckResult:
    name = expected_path.name
    details: list[str] = []
    if not generated_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            details=[f"Missing generated file: {generated_path}"],
        )
    if not expected_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            details=[f"Missing expected file: {expected_path}"],
        )

    generated_df = pd.read_csv(generated_path)
    expected_df = pd.read_csv(expected_path)

    generated_cols = list(generated_df.columns)
    expected_cols = list(expected_df.columns)
    if generated_cols != expected_cols:
        details.append(
            f"Column mismatch: generated={generated_cols}, expected={expected_cols}"
        )
    if len(generated_df) != len(expected_df):
        details.append(
            f"Row count mismatch: generated={len(generated_df)}, expected={len(expected_df)}"
        )

    if details:
        return CheckResult(name=name, passed=False, details=details)

    for column in expected_cols:
        generated_col = generated_df[column]
        expected_col = expected_df[column]
        if pd.api.types.is_numeric_dtype(generated_col) and pd.api.types.is_numeric_dtype(
            expected_col
        ):
            generated_values = generated_col.to_numpy(dtype=np.float64)
            expected_values = expected_col.to_numpy(dtype=np.float64)
            details.extend(
                _compare_numeric_arrays(
                    actual=generated_values,
                    expected=expected_values,
                    label=f"Column '{column}'",
                    rtol=rtol,
                    atol=atol,
                )
            )
            continue

        equal = (generated_col == expected_col) | (
            generated_col.isna() & expected_col.isna()
        )
        if not bool(equal.all()):
            bad_indices = np.flatnonzero(~equal.to_numpy(dtype=bool))
            first_idx = int(bad_indices[0])
            details.append(
                "Column "
                f"'{column}': first_mismatch_index={first_idx}, "
                f"actual={generated_col.iloc[first_idx]!r}, "
                f"expected={expected_col.iloc[first_idx]!r}"
            )

    return CheckResult(name=name, passed=not details, details=details)


def compare_ols(
    *,
    generated_ols_path: Path,
    processed_data: dict[str, Any],
    rtol: float,
    atol: float,
) -> CheckResult:
    details: list[str] = []
    check_name = "ols_processed.csv vs parameters.json"

    if not generated_ols_path.exists():
        return CheckResult(
            name=check_name,
            passed=False,
            details=[f"Missing generated file: {generated_ols_path}"],
        )

    ols_df = pd.read_csv(generated_ols_path)

    for csv_col, param_key in OLS_COLUMN_TO_PARAM_KEY.items():
        if csv_col not in ols_df.columns:
            details.append(f"Missing generated OLS column: {csv_col}")
            continue
        if param_key not in processed_data:
            details.append(f"Missing parameter processed-data key: {param_key}")
            continue

        actual_values = ols_df[csv_col].to_numpy(dtype=np.float64)
        expected_raw = np.asarray(processed_data[param_key], dtype=np.float64)
        if actual_values.size != expected_raw.size:
            details.append(
                f"{csv_col}: length mismatch generated={actual_values.size}, "
                f"expected={expected_raw.size}"
            )
            continue
        details.extend(
            _compare_numeric_arrays(
                actual=actual_values,
                expected=expected_raw,
                label=f"{csv_col} <-> {param_key}",
                rtol=rtol,
                atol=atol,
            )
        )

    if {"smoothed_exp", "smoothed_iso"} <= set(ols_df.columns):
        actual_exp_mean = float(np.mean(ols_df["smoothed_exp"].to_numpy(dtype=np.float64)))
        actual_iso_mean = float(np.mean(ols_df["smoothed_iso"].to_numpy(dtype=np.float64)))
        actual_signal_ratio = (
            float("inf") if actual_iso_mean == 0 else actual_exp_mean / actual_iso_mean
        )
        scalar_actual = {
            "exp mean": actual_exp_mean,
            "iso mean": actual_iso_mean,
            "signal ratio": actual_signal_ratio,
        }

        for scalar_name in SCALAR_FIELDS:
            if scalar_name not in processed_data:
                details.append(
                    f"Missing parameter processed-data key: {scalar_name}"
                )
                continue
            expected_value = float(processed_data[scalar_name])
            details.extend(
                _compare_scalar(
                    actual=scalar_actual[scalar_name],
                    expected=expected_value,
                    label=f"{scalar_name}",
                    rtol=rtol,
                    atol=atol,
                )
            )
    else:
        details.append(
            "Missing required OLS columns for scalar checks: "
            "smoothed_exp and/or smoothed_iso."
        )

    return CheckResult(name=check_name, passed=not details, details=details)


def print_subject_report(context: SubjectContext, checks: list[CheckResult]) -> None:
    print(f"\nSubject {context.subject_id} ({context.generated_dir_name})")
    print(f"  expected_dir: {context.subject_dir}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}")
        if not check.passed:
            for detail in check.details:
                print(f"    - {detail}")


def run_comparison(args: argparse.Namespace) -> int:
    test_data_dir = args.test_data_dir.resolve()
    if not test_data_dir.exists():
        raise ComparisonError(f"test_data directory not found: {test_data_dir}")

    tank_dir = find_tank_dir(test_data_dir)
    subject_dirs = find_subject_dirs(test_data_dir)
    subject_contexts = [load_subject_context(path) for path in subject_dirs]
    indexed = validate_and_index_subjects(subject_contexts)
    first_ctx = indexed["first"]
    second_ctx = indexed["second"]

    smoothing_method, smoothing_fraction, new_sampling_rate, ttl_filtering, ttl_start_offset = (
        extract_cli_smoothing_args(first_ctx, second_ctx)
    )

    temp_output_created = False
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="tank_cli_compare_")).resolve()
        temp_output_created = True
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_tank_cli_command(
        tank_dir=tank_dir,
        output_dir=output_dir,
        first=first_ctx,
        second=second_ctx,
        smoothing_method=smoothing_method,
        smoothing_fraction=smoothing_fraction,
        new_sampling_rate=new_sampling_rate,
        ttl_filtering=ttl_filtering,
        ttl_start_offset=ttl_start_offset,
    )

    print("Running tank_cli command:")
    print("  " + " ".join(command))
    run_tank_cli(command)

    subject_reports: list[tuple[SubjectContext, list[CheckResult]]] = []
    total_checks = 0
    failed_checks = 0

    for subject in (first_ctx, second_ctx):
        generated_subject_dir = output_dir / subject.generated_dir_name
        checks: list[CheckResult] = []
        for stream_file in STREAM_FILES:
            checks.append(
                compare_stream_csv(
                    generated_path=generated_subject_dir / stream_file,
                    expected_path=subject.subject_dir / stream_file,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
        checks.append(
            compare_ols(
                generated_ols_path=generated_subject_dir / "ols_processed.csv",
                processed_data=subject.processed_data,
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        subject_reports.append((subject, checks))
        total_checks += len(checks)
        failed_checks += sum(1 for check in checks if not check.passed)

    for subject, checks in subject_reports:
        print_subject_report(subject, checks)

    print("\nSummary")
    print(f"  total_checks: {total_checks}")
    print(f"  passed_checks: {total_checks - failed_checks}")
    print(f"  failed_checks: {failed_checks}")
    print(f"  output_dir: {output_dir}")

    if temp_output_created and not args.keep_output:
        shutil.rmtree(output_dir, ignore_errors=True)
        print("  temp_output_cleaned: yes")
    elif temp_output_created:
        print("  temp_output_cleaned: no (--keep-output set)")
    else:
        print("  temp_output_cleaned: not-applicable (--output-dir provided)")

    return 0 if failed_checks == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_comparison(args)
    except ComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
