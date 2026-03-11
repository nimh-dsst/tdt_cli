"""Simple argparse CLI to parse one TDT tank and export subject CSVs."""

import argparse
import json
import logging
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ols_processing import run_ols_processing
from .utils import stream_formatter

logger = logging.getLogger("tank_cli.cli")

DEFAULT_SMOOTHING_METHOD = "Downsample then smooth"
DEFAULT_SMOOTHING_FRACTION = 0.002
DEFAULT_NEW_SAMPLING_RATE = 10.0
DEFAULT_TTL_START_OFFSET = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one TDT tank folder and export per-subject stream CSVs, "
            "with optional epoc CSV and OLS dF/F outputs."
        )
    )
    parser.add_argument("--tank-dir", type=Path)
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Path to a JSON file containing CLI parameter values.",
    )
    parser.add_argument(
        "--view-streams",
        action="store_true",
        help="List available stream names in the tank and exit.",
    )
    parser.add_argument(
        "--view-epocs",
        action="store_true",
        help="List available epoc names in the tank and exit.",
    )
    parser.add_argument("--num-subjects", type=int, choices=[1, 2])

    parser.add_argument("--first-iso")
    parser.add_argument("--first-exp")
    parser.add_argument(
        "--first-ttl",
        default="None",
        help=(
            "TTL source for first subject: stream name, epoc name, "
            "or 'None'. Stream name is preferred if both exist."
        ),
    )

    parser.add_argument("--second-iso")
    parser.add_argument("--second-exp")
    parser.add_argument(
        "--second-ttl",
        default="None",
        help=(
            "TTL source for second subject: stream name, epoc name, "
            "or 'None'. Stream name is preferred if both exist."
        ),
    )

    parser.add_argument("--epoc", default="None")
    parser.add_argument("--export-epoc-csv", action="store_true")

    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--run-ols", action="store_true")
    parser.add_argument(
        "--smoothing-method",
        default=DEFAULT_SMOOTHING_METHOD,
        choices=[
            "Downsample then smooth",
            "Smooth at downsampled points",
            "Downsampled only",
        ],
    )
    parser.add_argument(
        "--smoothing-fraction",
        type=float,
        default=DEFAULT_SMOOTHING_FRACTION,
    )
    parser.add_argument(
        "--new-sampling-rate",
        type=float,
        default=DEFAULT_NEW_SAMPLING_RATE,
    )
    parser.add_argument("--ttl-filtering", action="store_true")
    parser.add_argument(
        "--ttl-start-offset", type=int, default=DEFAULT_TTL_START_OFFSET
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(argv_tokens)
    except SystemExit as exc:
        return int(exc.code)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        args = _merge_json_parameters(
            parser=parser, args=args, argv_tokens=argv_tokens
        )
        run_cli(args)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _load_tdt_read_block() -> Any:
    # tdt currently emits invalid escape-sequence SyntaxWarnings on import
    # under newer Python versions; suppress those third-party warnings only.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*invalid escape sequence.*",
            category=SyntaxWarning,
        )
        from tdt import read_block

    return read_block


def _merge_json_parameters(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv_tokens: list[str],
) -> argparse.Namespace:
    json_path = args.json_path
    if json_path is None:
        return args

    json_params = _load_json_parameters(json_path)
    json_params.pop("json_path", None)
    action_by_dest = _build_action_index(parser)
    allowed_keys = set(action_by_dest.keys()) - {"json_path"}
    unknown_keys = sorted(set(json_params.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "Unknown key(s) in --json file: "
            + ", ".join(repr(key) for key in unknown_keys)
        )

    explicit_cli_dests = _explicit_cli_destinations(
        parser=parser, argv_tokens=argv_tokens
    )
    explicit_cli_dests.discard("json_path")

    for key, raw_value in json_params.items():
        action = action_by_dest[key]
        coerced_value = _coerce_json_value(
            key=key, raw_value=raw_value, action=action
        )
        if key in explicit_cli_dests:
            if key == "tank_dir":
                continue
            cli_value = getattr(args, key)
            if cli_value != coerced_value:
                raise ValueError(
                    f"Conflict for '{key}': CLI value {cli_value!r} differs "
                    f"from JSON value {coerced_value!r}"
                )
            continue
        setattr(args, key, coerced_value)

    return args


def _load_json_parameters(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON parameter file not found: {json_path}"
        )

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in parameter file '{json_path}': {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"JSON parameter file '{json_path}' must contain "
            "a top-level JSON object"
        )

    return _extract_json_parameters(json_path=json_path, payload=payload)


def _extract_json_parameters(
    *, json_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    if "parameters" in payload or "tank_cli_version" in payload:
        if "parameters" not in payload:
            raise ValueError(
                f"Run metadata file '{json_path}' must include "
                "'parameters'"
            )
        parameters = payload["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Run metadata file '{json_path}' must contain "
                "an object at key 'parameters'"
            )
        return parameters
    return payload


def _build_action_index(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help"}
    }


def _explicit_cli_destinations(
    *, parser: argparse.ArgumentParser, argv_tokens: list[str]
) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_dests: set[str] = set()
    for token in argv_tokens:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_dests.add(dest)
    return explicit_dests


def _coerce_json_value(
    *, key: str, raw_value: Any, action: argparse.Action
) -> Any:
    if isinstance(action, argparse._StoreTrueAction):
        if type(raw_value) is not bool:
            raise ValueError(
                f"JSON key '{key}' must be a boolean "
                f"(got {type(raw_value).__name__})"
            )
        return raw_value

    if isinstance(action, argparse._StoreFalseAction):
        if type(raw_value) is not bool:
            raise ValueError(
                f"JSON key '{key}' must be a boolean "
                f"(got {type(raw_value).__name__})"
            )
        return raw_value

    if raw_value is None:
        if action.default is None:
            return None
        raise ValueError(
            f"JSON key '{key}' cannot be null for this option"
        )

    converter = action.type if action.type is not None else str
    try:
        coerced_value = converter(raw_value)
    except (TypeError, ValueError) as exc:
        converter_name = getattr(converter, "__name__", str(converter))
        raise ValueError(
            f"JSON key '{key}' must be coercible to {converter_name}; "
            f"got {raw_value!r}"
        ) from exc

    if (
        action.choices is not None
        and coerced_value not in set(action.choices)
    ):
        raise ValueError(
            f"JSON key '{key}' must be one of {list(action.choices)}; "
            f"got {coerced_value!r}"
        )

    return coerced_value


def run_cli(args: argparse.Namespace) -> None:
    read_block = _load_tdt_read_block()

    tank_dir: Path | None = args.tank_dir
    if tank_dir is None:
        raise ValueError(
            "--tank-dir is required unless provided in --json"
        )
    if not tank_dir.exists():
        raise FileNotFoundError(f"Tank directory not found: {tank_dir}")

    if args.view_streams or args.view_epocs:
        row_data = read_block(str(tank_dir))
        if args.view_streams:
            available_streams = sorted(row_data["streams"].keys())
            for stream_name in available_streams:
                print(stream_name)
        if args.view_epocs:
            available_epocs = sorted(row_data["epocs"].keys())
            for epoc_name in available_epocs:
                print(epoc_name)
        return

    _validate_flag_combinations(args)

    row_data = read_block(str(tank_dir))
    available_streams = sorted(row_data["streams"].keys())
    available_epocs = sorted(row_data["epocs"].keys())

    output_root = (
        args.output_dir
        if args.output_dir is not None
        else tank_dir.parent / f"{tank_dir.name}_extract"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    subject_stream_cfgs = _build_subject_configs(args)
    resolved_subject_cfgs = _resolve_subject_configs(
        subject_stream_cfgs=subject_stream_cfgs,
        available_streams=available_streams,
        available_epocs=available_epocs,
        ttl_filtering=args.ttl_filtering,
    )
    _validate_epoc_name(
        export_epoc_csv=args.export_epoc_csv,
        epoc_name=args.epoc,
        available_epocs=available_epocs,
    )
    _write_run_metadata(output_root=output_root, args=args)

    for cfg in resolved_subject_cfgs:
        subject_dir = output_root / cfg["subject_dir_name"]
        subject_dir.mkdir(parents=True, exist_ok=True)
        _export_stream_csvs(
            row_data=row_data,
            subject_dir=subject_dir,
            iso_stream=cfg["iso_stream"],
            exp_stream=cfg["exp_stream"],
            ttl_source_type=cfg["ttl_source_type"],
            ttl_source_name=cfg["ttl_source_name"],
        )
        if cfg["ttl_source_type"] == "epoc":
            _export_epoc_marker_csv(
                row_data=row_data,
                epoc_name=cfg["ttl_source_name"],
                subject_dir=subject_dir,
            )
        if args.export_epoc_csv and args.epoc != "None":
            _export_epoc_csv(
                row_data=row_data,
                epoc_name=args.epoc,
                subject_dir=subject_dir,
            )
        if args.run_ols:
            _export_ols_csv(
                row_data=row_data,
                subject_dir=subject_dir,
                iso_stream=cfg["iso_stream"],
                exp_stream=cfg["exp_stream"],
                ttl_source_type=cfg["ttl_source_type"],
                ttl_source_name=cfg["ttl_source_name"],
                smoothing_method=args.smoothing_method,
                smoothing_fraction=args.smoothing_fraction,
                new_sampling_rate=args.new_sampling_rate,
                ttl_filtering=args.ttl_filtering,
                ttl_start_offset=args.ttl_start_offset,
            )

    logger.info(
        "Exported %d subject(s) from tank '%s' to %s",
        len(subject_stream_cfgs),
        tank_dir.name,
        output_root,
    )


def _validate_flag_combinations(args: argparse.Namespace) -> None:
    if args.num_subjects is None:
        raise ValueError(
            "--num-subjects is required unless --view-streams or "
            "--view-epocs is set"
        )
    if args.first_iso is None:
        raise ValueError(
            "--first-iso is required unless --view-streams or "
            "--view-epocs is set"
        )
    if args.first_exp is None:
        raise ValueError(
            "--first-exp is required unless --view-streams or "
            "--view-epocs is set"
        )

    if args.num_subjects == 1:
        if (
            args.second_iso is not None
            or args.second_exp is not None
            or args.second_ttl != "None"
        ):
            raise ValueError(
                "second-subject flags are not allowed when --num-subjects=1"
            )
    else:
        if args.second_iso is None or args.second_exp is None:
            raise ValueError(
                "--second-iso and --second-exp are required when "
                "--num-subjects=2"
            )

    if args.export_epoc_csv and args.epoc == "None":
        raise ValueError(
            "--export-epoc-csv requires --epoc to be provided"
        )

    if args.ttl_start_offset < 0:
        raise ValueError("--ttl-start-offset must be >= 0")
    if args.new_sampling_rate <= 0:
        raise ValueError("--new-sampling-rate must be > 0")
    if args.smoothing_method != "Downsampled only" and not (
        0 < args.smoothing_fraction <= 1
    ):
        raise ValueError(
            "--smoothing-fraction must be between 0 and 1 "
            "for smoothing methods"
        )


def _build_subject_configs(args: argparse.Namespace) -> list[dict[str, str]]:
    cfgs = [
        {
            "subject_dir_name": "first",
            "iso_stream": args.first_iso,
            "exp_stream": args.first_exp,
            "ttl_name": args.first_ttl,
        }
    ]
    if args.num_subjects == 2:
        cfgs.append(
            {
                "subject_dir_name": "second",
                "iso_stream": args.second_iso,
                "exp_stream": args.second_exp,
                "ttl_name": args.second_ttl,
            }
        )
    return cfgs


def _resolve_subject_configs(
    *,
    subject_stream_cfgs: list[dict[str, str]],
    available_streams: list[str],
    available_epocs: list[str],
    ttl_filtering: bool,
) -> list[dict[str, str]]:
    available_stream_set = set(available_streams)
    available_epoc_set = set(available_epocs)
    resolved_cfgs: list[dict[str, str]] = []
    for cfg in subject_stream_cfgs:
        subject_name = cfg["subject_dir_name"]
        for key in ("iso_stream", "exp_stream"):
            value = cfg[key]
            if value not in available_stream_set:
                raise ValueError(
                    f"Unknown {key} '{value}' for {subject_name}. "
                    f"Available streams: {available_streams}"
                )

        ttl_source_type, ttl_source_name = _resolve_ttl_source(
            ttl_name=cfg["ttl_name"],
            available_stream_set=available_stream_set,
            available_epoc_set=available_epoc_set,
        )
        if ttl_source_type == "unknown":
            raise ValueError(
                f"Unknown ttl source '{cfg['ttl_name']}' for {subject_name}. "
                f"Available streams: {available_streams}. "
                f"Available epocs: {available_epocs}"
            )
        if ttl_filtering and ttl_source_type == "none":
            raise ValueError(
                "TTL filtering requested but ttl source is 'None' "
                f"for {subject_name}"
            )
        resolved_cfg = dict(cfg)
        resolved_cfg["ttl_source_type"] = ttl_source_type
        resolved_cfg["ttl_source_name"] = ttl_source_name
        resolved_cfgs.append(resolved_cfg)
    return resolved_cfgs


def _resolve_ttl_source(
    *,
    ttl_name: str,
    available_stream_set: set[str],
    available_epoc_set: set[str],
) -> tuple[str, str]:
    if ttl_name == "None":
        return "none", "None"
    if ttl_name in available_stream_set:
        return "stream", ttl_name
    if ttl_name in available_epoc_set:
        return "epoc", ttl_name
    return "unknown", ttl_name


def _write_run_metadata(
    *, output_root: Path, args: argparse.Namespace
) -> None:
    metadata_path = output_root / "run_metadata.json"
    payload = {
        "tank_cli_version": _get_tank_cli_version(),
        "parameters": {
            key: _to_jsonable(value)
            for key, value in vars(args).items()
            if key != "json_path"
        },
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _get_tank_cli_version() -> str:
    try:
        return metadata.version("tdt-cli")
    except metadata.PackageNotFoundError:
        return "unknown"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _validate_epoc_name(
    *, export_epoc_csv: bool, epoc_name: str, available_epocs: list[str]
) -> None:
    if not export_epoc_csv:
        return
    if epoc_name == "None":
        raise ValueError("--export-epoc-csv requires a non-None --epoc")
    if epoc_name not in set(available_epocs):
        raise ValueError(
            f"Unknown epoc '{epoc_name}'. Available epocs: {available_epocs}"
        )


def _export_stream_csvs(
    *,
    row_data: Any,
    subject_dir: Path,
    iso_stream: str,
    exp_stream: str,
    ttl_source_type: str,
    ttl_source_name: str,
) -> None:
    info = row_data.info
    iso_df = stream_formatter(info, row_data["streams"][iso_stream])
    exp_df = stream_formatter(info, row_data["streams"][exp_stream])
    iso_df.to_csv(subject_dir / "iso_stream.csv", index=False)
    exp_df.to_csv(subject_dir / "exp_stream.csv", index=False)
    if ttl_source_type == "stream":
        ttl_df = stream_formatter(info, row_data["streams"][ttl_source_name])
        ttl_df.to_csv(subject_dir / "ttl_stream.csv", index=False)


def _export_epoc_csv(
    *, row_data: Any, epoc_name: str, subject_dir: Path
) -> None:
    onset = row_data["epocs"][epoc_name]["onset"]
    offset = row_data["epocs"][epoc_name]["offset"]
    data = row_data["epocs"][epoc_name]["data"]
    epoc_df = pd.DataFrame({"index": data, "onset": onset, "offset": offset})
    epoc_df.to_csv(subject_dir / "epoc.csv", index=False)


def _load_epoc_events(
    *, row_data: Any, epoc_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epocs = row_data["epocs"]
    if epoc_name not in set(epocs.keys()):
        raise ValueError(f"Unknown epoc '{epoc_name}' for TTL source")
    epoc = epocs[epoc_name]
    try:
        onset = np.asarray(epoc["onset"], dtype=np.float64)
        offset = np.asarray(epoc["offset"], dtype=np.float64)
        data = np.asarray(epoc["data"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Epoc '{epoc_name}' must contain onset, offset, and data arrays"
        ) from exc
    if onset.size == 0:
        raise ValueError(
            f"Epoc '{epoc_name}' has an empty onset array for TTL source"
        )
    if not (onset.size == offset.size == data.size):
        raise ValueError(
            f"Epoc '{epoc_name}' onset/offset/data lengths must match "
            f"(got onset={onset.size}, offset={offset.size}, data={data.size})"
        )
    return onset, offset, data


def _first_epoc_onset(
    *, row_data: Any, epoc_name: str
) -> tuple[float, int]:
    onset, _, _ = _load_epoc_events(row_data=row_data, epoc_name=epoc_name)
    first_idx = int(np.argmin(onset))
    return float(onset[first_idx]), first_idx


def _export_epoc_marker_csv(
    *, row_data: Any, epoc_name: str, subject_dir: Path
) -> None:
    onset, offset, data = _load_epoc_events(
        row_data=row_data, epoc_name=epoc_name
    )
    _, first_idx = _first_epoc_onset(row_data=row_data, epoc_name=epoc_name)
    first_onset_for_ttl = np.zeros(onset.size, dtype=bool)
    first_onset_for_ttl[first_idx] = True
    marker_df = pd.DataFrame(
        {
            "onset": onset,
            "offset": offset,
            "data": data,
            "first_onset_for_ttl": first_onset_for_ttl,
        }
    )
    marker_df.to_csv(subject_dir / "epoc_marker.csv", index=False)


def _build_epoc_ttl_stream(
    *,
    row_data: Any,
    epoc_name: str,
    fs: float,
    signal_length: int,
) -> np.ndarray:
    onset, offset, _ = _load_epoc_events(row_data=row_data, epoc_name=epoc_name)
    ttl_stream = np.zeros(signal_length, dtype=np.float32)
    if signal_length <= 0:
        raise ValueError("Signal length must be > 0 for epoc TTL conversion")

    onset_idx = np.floor(onset * fs).astype(np.int64)
    offset_idx = np.ceil(offset * fs).astype(np.int64)

    for start_raw, end_raw in zip(onset_idx, offset_idx):
        start = int(np.clip(start_raw, 0, signal_length))
        end = int(np.clip(end_raw, 0, signal_length))
        if end < start:
            raise ValueError(
                "Epoc onset/offset produced an invalid interval for TTL source "
                f"(start={start}, end={end})"
            )
        if end > start:
            ttl_stream[start:end] = 1.0

    # Match pipeline behavior: when multiple epoc pulses exist, retain data
    # from the first onset through the end of the recording.
    rising_edges = np.where(np.diff(ttl_stream.astype(np.int32)) == 1)[0]
    if rising_edges.size > 1:
        ttl_stream = np.maximum.accumulate(ttl_stream)

    if not np.any(ttl_stream == 1.0):
        raise ValueError(
            "No epoc intervals overlap the raw signal timeline for TTL source"
        )

    return ttl_stream


def _export_ols_csv(
    *,
    row_data: Any,
    subject_dir: Path,
    iso_stream: str,
    exp_stream: str,
    ttl_source_type: str,
    ttl_source_name: str,
    smoothing_method: str,
    smoothing_fraction: float,
    new_sampling_rate: float,
    ttl_filtering: bool,
    ttl_start_offset: int,
) -> None:
    iso = row_data["streams"][iso_stream]
    exp = row_data["streams"][exp_stream]
    iso_fs = float(iso["fs"])
    exp_fs = float(exp["fs"])
    ttl_data = None
    ttl_fs = None
    if ttl_filtering and ttl_source_type == "none":
        raise ValueError("TTL filtering requires a non-None TTL source")
    if ttl_source_type == "stream":
        ttl = row_data["streams"][ttl_source_name]
        ttl_data = ttl["data"]
        ttl_fs = float(ttl["fs"])
        if not (iso_fs == exp_fs == ttl_fs):
            raise ValueError(
                "iso, exp, and ttl stream sampling rates must match "
                f"(got iso={iso_fs}, exp={exp_fs}, ttl={ttl_fs})"
            )
    if ttl_filtering and ttl_source_type == "epoc":
        ttl_data = _build_epoc_ttl_stream(
            row_data=row_data,
            epoc_name=ttl_source_name,
            fs=exp_fs,
            signal_length=int(np.asarray(exp["data"]).size),
        )
        ttl_fs = exp_fs

    results = run_ols_processing(
        iso_data=iso["data"],
        exp_data=exp["data"],
        iso_fs=iso_fs,
        exp_fs=exp_fs,
        smoothing_method=smoothing_method,
        smoothing_fraction=smoothing_fraction,
        new_sampling_rate=new_sampling_rate,
        ttl_stream_data=ttl_data,
        ttl_fs=ttl_fs,
        ttl_filtering=ttl_filtering,
        ttl_start_offset=ttl_start_offset,
    )

    output_df = pd.DataFrame(
        {
            "time": np.asarray(results["time_array"], dtype=np.float64),
            "smoothed_exp": np.asarray(
                results["smoothed_exp"], dtype=np.float64
            ),
            "smoothed_iso": np.asarray(
                results["smoothed_iso"], dtype=np.float64
            ),
            "delta_f": np.asarray(results["delta_f"], dtype=np.float64),
            "delta_f_norm": np.asarray(
                results["delta_f_norm"], dtype=np.float64
            ),
            "delta_f_zscore": np.asarray(
                results["delta_f_zscore"], dtype=np.float64
            ),
        }
    )
    output_df.to_csv(subject_dir / "ols_processed.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
