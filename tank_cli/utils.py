from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import re

import numpy as np
from numpy.typing import NDArray
import pandas as pd

NUM_OF_POINTS: int = 128

def _calc_timepoints(
    stream: Any, num_of_points: int = NUM_OF_POINTS
) -> list[float]:
    """
    Calculate the timepoints for a stream.
    """
    calc_timepoints: list[float] = []
    for i in range(len(stream.data)):
        if i % num_of_points == 0:
            calc_timepoint = np.float64(i) * (np.float64(1) / stream.fs)
            calc_timepoints.append(calc_timepoint)
    return calc_timepoints

def _parse_datetime(datetime_str: str) -> datetime:
    """
    Parse datetime string in format 'YYMMDD-HHMMSS'.

    Args:
        datetime_str: String in format '250103-001644'

    Returns:
        datetime object

    Example:
        >>> parse_datetime('250103-001644')
        datetime.datetime(2025, 1, 3, 0, 16, 44)
    """
    return datetime.strptime(datetime_str, "%y%m%d-%H%M%S")

def tank_dir_parser(
    tank_dir: Path,
) -> dict[str, str | int | datetime | list[str]]:
    """
    Parse the tank directory name to get subject information.

    Args:
        tank_dir: The path to the tank directory.

    Returns:
        A dictionary containing the subject information.
    """
    tank_name: str = tank_dir.stem
    split_list: list[str] = tank_name.split("-")
    if len(split_list) != 3:
        raise ValueError(f"Invalid tank directory name: {tank_name}")

    subject_data: str = split_list[0]  # e.g. "123456_M1_F1" or "123456_M1"
    cage_number: str = subject_data.split("_")[0]

    try:
        tank_dt: datetime = _parse_datetime("-".join(split_list[1:]))
    except ValueError as e:
        raise ValueError(f"Invalid tank directory name: {tank_name}") from e

    # IMPORTANT: parse subject IDs from subject_data ONLY (not tank_name),
    # so we don't accidentally match pieces right before "-<date>-<time>".
    m = re.search(r"[MF]\d+_\d+$", subject_data)
    if m:
        # Example: "M1_2" -> ["M1", "M2"] (sex inferred for 2nd id)
        subjects_group: str = m.group()
        sex: Literal["M", "F"] = subjects_group[0]  # type: ignore[assignment]
        a, b = subjects_group.split("_")
        subject_ids: list[str] = [a, f"{sex}{b}"]
    else:
        m = re.search(r"[MF]\d+_[MF]\d+$", subject_data)
        if m:
            # Example: "M1_F1" -> ["M1", "F1"]
            subjects_group = m.group()
            subject_ids = subjects_group.split("_")
        else:
            m = re.search(r"[MF]\d+$", subject_data)
            if m:
                # Example: "M1" -> ["M1"]
                subject_ids = [m.group()]
            else:
                raise ValueError(
                    "Cannot parse subject ids."
                    + f" Invalid tank directory name: {tank_name}"
                )

    # add cage number to subject ids
    subject_ids = [f"{cage_number}_{subject_id}" for subject_id in subject_ids]
    num_subjects: int = len(subject_ids)

    return {
        "tank_name": tank_name,
        "cage_number": cage_number,
        "num_subjects": num_subjects,
        "subject_ids": subject_ids,
        "tank_datetime": tank_dt,
    }


def stream_formatter(
    data_info: Any, stream: Any, num_of_points: int = NUM_OF_POINTS
) -> pd.DataFrame:
    """
    Format the stream data to a pandas DataFrame.
    """
    block_name: str = data_info["blockname"]
    event_name: str = stream.name
    if "ts" not in stream.keys():
        time: NDArray[np.float64] = np.array(
            _calc_timepoints(stream, num_of_points)
        )
    else:
        time = stream.ts
    if len(stream.channel) == 1:
        channel: int = stream.channel[0]
    else:
        raise ValueError(
            f"Stream {event_name} has {len(stream.channel)} channels"
        )
    fs: np.float64 = stream.fs
    names: list[str] = [f"D{i}" for i in range(num_of_points)]
    stream_df: pd.DataFrame = pd.DataFrame(
        stream.data.reshape(-1, num_of_points),
        columns=names,
    )
    stream_df["BLOCK"] = block_name
    stream_df["EVENT"] = event_name
    stream_df["TIME"] = time
    stream_df["CHAN"] = channel
    stream_df["Sampling_Freq"] = fs
    stream_df["NumOfPoints"] = num_of_points
    reorder: list[str] = [
        "BLOCK",
        "EVENT",
        "TIME",
        "CHAN",
        "Sampling_Freq",
        "NumOfPoints",
    ] + names
    stream_df = stream_df[reorder]
    return stream_df
