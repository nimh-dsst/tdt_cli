from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore


def calculate_time_vector(stream: Any) -> NDArray[np.float64]:
    original_time: np.float64 = np.float64(len(stream["data"])) / stream["fs"]
    calc_time: NDArray[np.float64] = np.arange(
        0,
        original_time,
        1 / stream["fs"],
        dtype=np.float64,
    )
    return calc_time


def smooth_at_downsampled_points(
    data: NDArray[np.float32],
    fs_orig: np.float64,
    fs_target: np.float64,
    lowess_frac: np.float64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Smooth data using LOWESS, evaluating only at target downsampled time points.

    Parameters
    ----------
    data : NDArray[np.float64]
        Original high-frequency data.
    fs_orig : float
        Original sampling frequency.
    fs_target : float
        Target sampling frequency.
    lowess_frac : float
        Fraction for LOWESS smoother.

    Returns
    -------
    target_time_points : NDArray[np.float64]
        The time vector for the downsampled rate.
    smoothed_data_at_targets : NDArray[np.float64]
        Smoothed data evaluated only at target_time_points.
    """
    # 1. Create the original time vector
    n_orig: int = data.shape[0]
    original_time: NDArray[np.float64] = np.arange(n_orig) / fs_orig

    # 2. Determine the target time points for downsampling
    # We want points roughly every 1/fs_target seconds.
    # Find the indices in the original data that correspond most closely
    # to the desired downsampled times.
    target_dt: np.float64 = np.float64(1.0) / fs_target
    target_time_points: NDArray[np.float64] = np.arange(
        0, original_time[-1], target_dt, dtype=np.float64
    )

    # Alternatively, if using simple decimation by integer factor q:
    # q = int(np.floor(fs_orig / fs_target))
    # target_indices = np.arange(0, n_orig, q)
    # target_time_points = original_time[target_indices] # More direct for decimation

    print(f"Original points: {n_orig}")
    print(f"Evaluating LOWESS at {len(target_time_points)} points.")

    # 3. Apply LOWESS, evaluating only at target_time_points
    # NOTE: lowess still uses the FULL original_time and data
    #       to perform the local regressions around each target_time_point.
    smoothed_data_at_targets: NDArray[float] = lowess(  # type: ignore
        endog=data.astype(np.float64),
        exog=original_time.astype(np.float64),
        frac=lowess_frac,
        it=0,
        xvals=target_time_points.astype(np.float64),  # Evaluate here!
        return_sorted=False,  # xvals aren't necessarily sorted relative to exog
    )

    return target_time_points, smoothed_data_at_targets  # type: ignore


def downsample_then_smooth(
    data: NDArray[np.float32],
    fs_orig: np.float64,
    fs_target: np.float64,
    lowess_frac: np.float64 | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], np.float64]:
    """
    Downsample data first, optionally smooth using LOWESS.

    If `lowess_frac` is a float, the interpolated data is smoothed using LOWESS
    with that fraction.
    If `lowess_frac` is None, the raw interpolated data is returned.

    Parameters
    ----------
    data : NDArray[np.float32]
        Original high-frequency data.
    fs_orig : np.float64
        Original sampling frequency.
    fs_target : np.float64
        Target sampling frequency.
    lowess_frac : np.float64 | None
        Fraction for LOWESS smoother. If None, smoothing is skipped.

    Returns
    -------
    interpolated_time : NDArray[np.float64]
        Time vector for the (potentially) interpolated data.
    output_data : NDArray[np.float64]
        If `lowess_frac` is not None, data after interpolation and LOWESS smoothing.
        If `lowess_frac` is None, data after interpolation (or original data if no interpolation).
    fs_new : np.float64
        The actual sampling frequency after interpolation (or original fs if no interpolation).
    """
    n_orig = data.shape[0]
    original_duration = n_orig / fs_orig

    # Handle cases where downsampling is not needed or possible
    if fs_target >= fs_orig:
        print(
            f"Warning: Target frequency ({fs_target} Hz) is >= original frequency ({fs_orig} Hz). "
            "No downsampling performed. Smoothing original data."
        )
        fs_new = fs_orig
        # Use original time and data for smoothing
        interpolated_time = np.arange(n_orig, dtype=np.float64) / fs_orig
        interpolated_data: NDArray[np.float64] = data.astype(
            np.float64
        )  # Ensure float64

    else:
        # 1. Create time vectors and perform cubic spline interpolation
        original_time = np.arange(n_orig, dtype=np.float64) / fs_orig

        # Create target time points for interpolation
        dt_target = np.float64(1.0 / fs_target)
        # Use np.maximum for element-wise comparison/broadcast if needed, also handles scalars
        interpolated_time = np.arange(
            0,
            np.maximum(original_duration, dt_target),
            dt_target,
            dtype=np.float64,
        )

        # Ensure interpolated_time doesn't significantly exceed original_duration
        if (
            interpolated_time.size > 0
            and interpolated_time[-1] > original_duration
        ):
            interpolated_time = interpolated_time[
                interpolated_time <= original_duration
            ]

        if interpolated_time.size == 0:
            # Handle edge case where target frequency is too high for duration
            # Fallback to using just the first original point?
            # Or raise error? Let's raise for now.
            raise ValueError(
                f"Cannot interpolate: target frequency {fs_target} Hz results in zero time points "
                f"for duration {original_duration} s."
            )

        print(
            f"Interpolating from {n_orig} points ({fs_orig:.2f} Hz) to {interpolated_time.size} points "
            f"(target {fs_target:.2f} Hz)."
        )

        # Create spline (requires float64)
        spline = CubicSpline(original_time, data.astype(np.float64))

        # Interpolate (returns float64)
        # Ensure output is real and float64, spline can sometimes return complex
        interpolated_data_any = spline(interpolated_time)
        # Check if the interpolated data is complex
        if np.iscomplexobj(interpolated_data_any):
            raise ValueError(
                "CubicSpline interpolation resulted in complex data type!"
                f" dtype: {interpolated_data_any.dtype}"
            )
        # If not complex, assign and ensure float64.
        # The type hint is applied here. The linter might complain about redefinition
        # if it sees the definition in the other branch, but this assignment is
        # necessary within this 'else' block scope.
        interpolated_data: NDArray[np.float64] = interpolated_data_any.astype(  # noqa: F821
            np.float64
        )

        # 2. Calculate the actual new sampling frequency
        # Use np.mean(np.diff) for potentially uneven spacing, though arange should be even
        if interpolated_time.size > 1:
            # Cast result of np.mean to float64 for consistency
            fs_new = np.float64(1.0 / np.mean(np.diff(interpolated_time)))
        elif interpolated_time.size == 1:
            fs_new = np.float64(
                fs_target
            )  # Or np.nan? Or original fs? Default to target.
        else:  # Should be unreachable due to check above
            fs_new = np.float64(np.nan)

        print(f"Actual new sampling frequency: {fs_new:.4f} Hz")

        # Step 3 (time vector creation) is implicitly done above

    # Define output_data type hint before conditional assignment
    output_data: NDArray[np.float64]

    # 4. Optionally apply LOWESS to the (potentially interpolated) data
    if lowess_frac is not None:
        # Check if lowess_frac is within a reasonable range if provided
        if not 0 < lowess_frac <= 1:
            raise ValueError("lowess_frac must be between 0 and 1.")

        print(f"Applying LOWESS smoothing with frac={lowess_frac}...")
        # Ensure input to lowess is float64
        output_data = lowess(  # type: ignore
            endog=interpolated_data,  # Already float64
            exog=interpolated_time,
            frac=lowess_frac,
            it=0,
            return_sorted=False,  # interpolated_time is sorted, but check lowess reqs
        )
    else:
        print("Skipping LOWESS smoothing (lowess_frac is None).")
        output_data = interpolated_data  # Return the unsmoothed data

    return interpolated_time, output_data, fs_new


def parse_tank_rename_csv(
    csv_path: Path,
) -> dict[str, list[dict[str, str]]]:
    """Parse tank_rename.csv and return subject-to-tank mapping.

    The cage_numbers and subject_ids columns contain comma-separated values.
    Positional order determines subject order (First, Second).
    """
    rename_df = pd.read_csv(
        csv_path, dtype={"cage_numbers": str, "subject_ids": str}
    )
    subject_tank_map: dict[str, list[dict[str, str]]] = {}
    order_labels = ["First", "Second"]
    for _, row in rename_df.iterrows():
        num_subjects = int(row["num_subjects"])
        cage_numbers = [s.strip() for s in row["cage_numbers"].split(",")]
        subject_ids = [s.strip() for s in row["subject_ids"].split(",")]
        tank_dir = str(row["tank_dir"])
        if (
            len(cage_numbers) != num_subjects
            or len(subject_ids) != num_subjects
        ):
            continue
        for i in range(num_subjects):
            full_subject_id = f"{cage_numbers[i]}_{subject_ids[i]}"
            entry = {"order": order_labels[i], "tank_dir": tank_dir}
            if full_subject_id not in subject_tank_map:
                subject_tank_map[full_subject_id] = []
            subject_tank_map[full_subject_id].append(entry)
    return subject_tank_map


def extract_unique_sessions(
    subject_session_dicts: list[dict[str, Any]],
) -> list[str]:
    """Return sorted unique session_date_string values."""
    unique_sessions: list[str] = sorted(
        set(d["session_date_string"] for d in subject_session_dicts)
    )
    return unique_sessions


def build_subject_entry_map(
    subject_session_dicts: list[dict[str, Any]],
    tank_rename_map: dict[str, list[dict[str, str]]],
    selected_session: str,
) -> dict[str, dict[str, str]]:
    """For a given session, map each subject_id to its best-matching tank entry.

    When a subject appears in multiple tanks, match by tank_dir stem.
    Falls back to the first entry when no stem match is found.
    """
    subject_entry_map: dict[str, dict[str, str]] = {}
    if not tank_rename_map:
        return subject_entry_map
    for ssd in subject_session_dicts:
        if ssd["session_date_string"] != selected_session:
            continue
        sid = ssd["subject_id"]
        if sid not in tank_rename_map:
            continue
        entries = tank_rename_map[sid]
        if len(entries) == 1:
            subject_entry_map[sid] = entries[0]
        else:
            ssd_tank_stem = Path(ssd.get("tank_dir", "")).stem
            for entry in entries:
                if Path(entry["tank_dir"]).stem == ssd_tank_stem:
                    subject_entry_map[sid] = entry
                    break
            else:
                subject_entry_map[sid] = entries[0]
    return subject_entry_map


def build_subject_display_map(
    subject_entry_map: dict[str, dict[str, str]],
    truncate: bool,
) -> dict[str, str]:
    """Build display labels like 'First subject of tank TankName'.

    When truncate is True, shows only the directory name;
    otherwise the full path.
    """
    subject_display_map: dict[str, str] = {}
    for sid, entry in subject_entry_map.items():
        tank_label = (
            Path(entry["tank_dir"]).name if truncate else entry["tank_dir"]
        )
        subject_display_map[sid] = (
            f"{entry['order']} subject of tank {tank_label}"
        )
    return subject_display_map
