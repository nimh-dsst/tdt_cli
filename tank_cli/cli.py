"""Simple argparse CLI to parse one TDT tank and export subject CSVs."""

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ols_processing import run_ols_processing
from .utils import stream_formatter, tank_dir_parser

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
    parser.add_argument("--tank-dir", required=True, type=Path)
    parser.add_argument(
        "--view-streams",
        action="store_true",
        help="List available stream names in the tank and exit.",
    )
    parser.add_argument("--num-subjects", type=int, choices=[1, 2])

    parser.add_argument("--first-iso")
    parser.add_argument("--first-exp")
    parser.add_argument("--first-ttl", default="None")

    parser.add_argument("--second-iso")
    parser.add_argument("--second-exp")
    parser.add_argument("--second-ttl", default="None")

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
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
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


def run_cli(args: argparse.Namespace) -> None:
    read_block = _load_tdt_read_block()

    tank_dir: Path = args.tank_dir
    if not tank_dir.exists():
        raise FileNotFoundError(f"Tank directory not found: {tank_dir}")

    if args.view_streams:
        row_data = read_block(str(tank_dir))
        available_streams = sorted(row_data["streams"].keys())
        for stream_name in available_streams:
            print(stream_name)
        return

    _validate_flag_combinations(args)

    row_data = read_block(str(tank_dir))
    available_streams = sorted(row_data["streams"].keys())
    available_epocs = sorted(row_data["epocs"].keys())

    output_root = (
        args.output_dir
        if args.output_dir is not None
        else tank_dir / "archiveflow_cli_output"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    subject_ids, dt_str = _resolve_subject_ids(
        tank_dir=tank_dir, num_subjects=args.num_subjects
    )

    subject_stream_cfgs = _build_subject_configs(args, subject_ids)
    _validate_stream_names(
        subject_stream_cfgs=subject_stream_cfgs,
        available_streams=available_streams,
        ttl_filtering=args.ttl_filtering,
    )
    _validate_epoc_name(
        export_epoc_csv=args.export_epoc_csv,
        epoc_name=args.epoc,
        available_epocs=available_epocs,
    )

    for cfg in subject_stream_cfgs:
        subject_dir = output_root / f"{cfg['subject_id']}_{dt_str}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        _export_stream_csvs(
            row_data=row_data,
            subject_dir=subject_dir,
            iso_stream=cfg["iso_stream"],
            exp_stream=cfg["exp_stream"],
            ttl_stream=cfg["ttl_stream"],
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
                ttl_stream=cfg["ttl_stream"],
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
            "--num-subjects is required unless --view-streams is set"
        )
    if args.first_iso is None:
        raise ValueError(
            "--first-iso is required unless --view-streams is set"
        )
    if args.first_exp is None:
        raise ValueError(
            "--first-exp is required unless --view-streams is set"
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


def _resolve_subject_ids(
    *, tank_dir: Path, num_subjects: int
) -> tuple[list[str], str]:
    dt_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    try:
        tank_info = tank_dir_parser(tank_dir)
        tank_dt = tank_info.get("tank_datetime")
        if isinstance(tank_dt, datetime):
            dt_str = tank_dt.strftime("%Y%m%d-%H%M%S")
        subject_ids = tank_info.get("subject_ids")
        if (
            isinstance(subject_ids, list)
            and all(isinstance(s, str) for s in subject_ids)
            and len(subject_ids) == num_subjects
        ):
            return subject_ids, dt_str
    except ValueError:
        pass

    return [f"subject{i}" for i in range(1, num_subjects + 1)], dt_str


def _build_subject_configs(
    args: argparse.Namespace, subject_ids: list[str]
) -> list[dict[str, str]]:
    cfgs = [
        {
            "subject_id": subject_ids[0],
            "iso_stream": args.first_iso,
            "exp_stream": args.first_exp,
            "ttl_stream": args.first_ttl,
        }
    ]
    if args.num_subjects == 2:
        cfgs.append(
            {
                "subject_id": subject_ids[1],
                "iso_stream": args.second_iso,
                "exp_stream": args.second_exp,
                "ttl_stream": args.second_ttl,
            }
        )
    return cfgs


def _validate_stream_names(
    *,
    subject_stream_cfgs: list[dict[str, str]],
    available_streams: list[str],
    ttl_filtering: bool,
) -> None:
    available_set = set(available_streams)
    for cfg in subject_stream_cfgs:
        subject_id = cfg["subject_id"]
        for key in ("iso_stream", "exp_stream"):
            value = cfg[key]
            if value not in available_set:
                raise ValueError(
                    f"Unknown {key} '{value}' for {subject_id}. "
                    f"Available streams: {available_streams}"
                )

        ttl_stream = cfg["ttl_stream"]
        if ttl_stream != "None" and ttl_stream not in available_set:
            raise ValueError(
                f"Unknown ttl_stream '{ttl_stream}' for {subject_id}. "
                f"Available streams: {available_streams}"
            )
        if ttl_filtering and ttl_stream == "None":
            raise ValueError(
                "TTL filtering requested but ttl stream is 'None' "
                f"for {subject_id}"
            )


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
    ttl_stream: str,
) -> None:
    info = row_data.info
    iso_df = stream_formatter(info, row_data["streams"][iso_stream])
    exp_df = stream_formatter(info, row_data["streams"][exp_stream])
    iso_df.to_csv(subject_dir / "iso_stream.csv", index=False)
    exp_df.to_csv(subject_dir / "exp_stream.csv", index=False)
    if ttl_stream != "None":
        ttl_df = stream_formatter(info, row_data["streams"][ttl_stream])
        ttl_df.to_csv(subject_dir / "ttl_stream.csv", index=False)


def _export_epoc_csv(
    *, row_data: Any, epoc_name: str, subject_dir: Path
) -> None:
    onset = row_data["epocs"][epoc_name]["onset"]
    offset = row_data["epocs"][epoc_name]["offset"]
    data = row_data["epocs"][epoc_name]["data"]
    epoc_df = pd.DataFrame({"index": data, "onset": onset, "offset": offset})
    epoc_df.to_csv(subject_dir / "epoc.csv", index=False)


def _export_ols_csv(
    *,
    row_data: Any,
    subject_dir: Path,
    iso_stream: str,
    exp_stream: str,
    ttl_stream: str,
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
    if ttl_stream != "None":
        ttl = row_data["streams"][ttl_stream]
        ttl_data = ttl["data"]
        ttl_fs = float(ttl["fs"])
        if not (iso_fs == exp_fs == ttl_fs):
            raise ValueError(
                "iso, exp, and ttl stream sampling rates must match "
                f"(got iso={iso_fs}, exp={exp_fs}, ttl={ttl_fs})"
            )

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
