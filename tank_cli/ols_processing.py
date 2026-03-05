from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import zscore

from .preprocessing import (
    downsample_then_smooth,
    smooth_at_downsampled_points,
)


def run_ols_processing(
    *,
    iso_data: NDArray[np.floating[Any]],
    exp_data: NDArray[np.floating[Any]],
    iso_fs: float,
    exp_fs: float,
    smoothing_method: str = "Downsample then smooth",
    smoothing_fraction: float = 0.002,
    new_sampling_rate: float = 10.0,
    ttl_stream_data: NDArray[np.floating[Any]] | None = None,
    ttl_fs: float | None = None,
    ttl_filtering: bool = False,
    ttl_start_offset: int = 0,
) -> dict[str, Any]:
    """Run smoothing + OLS dF/F correction and return processed arrays."""
    if ttl_filtering and ttl_stream_data is not None and ttl_fs is None:
        raise ValueError("ttl_fs is required when ttl_filtering is enabled")

    time_array, smoothed_exp, smoothed_iso = _apply_smoothing(
        iso_data=iso_data,
        exp_data=exp_data,
        iso_fs=iso_fs,
        exp_fs=exp_fs,
        smoothing_method=smoothing_method,
        smoothing_fraction=smoothing_fraction,
        new_sampling_rate=new_sampling_rate,
    )

    if ttl_filtering and ttl_stream_data is not None:
        _, new_ttl, _ = downsample_then_smooth(
            data=ttl_stream_data.astype(np.float32),
            fs_orig=np.float64(ttl_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=None,
        )
        ttl_int: NDArray[np.int32] = new_ttl.astype(np.int32)
        if not np.all(np.isin(ttl_int, [0, 1])):
            raise ValueError(
                "TTL values must be either 0 or 1 after downsampling"
            )
        ttl_index: NDArray[np.bool_] = ttl_int == 1
        if not np.any(ttl_index):
            raise ValueError(
                "No TTL==1 samples found after downsampling; cannot trim"
            )
        first_true_idx = int(np.argmax(ttl_index))
        start = max(0, first_true_idx - ttl_start_offset)
        ttl_index[start:first_true_idx] = True
        smoothed_exp = smoothed_exp[ttl_index]
        smoothed_iso = smoothed_iso[ttl_index]
        time_array = time_array[ttl_index]
    elif ttl_start_offset > 0:
        smoothed_exp = smoothed_exp[ttl_start_offset:]
        smoothed_iso = smoothed_iso[ttl_start_offset:]
        time_array = time_array[ttl_start_offset:]

    if smoothed_exp.size < 2 or smoothed_iso.size < 2:
        raise ValueError("Not enough data points for OLS dF/F computation")

    coefs = np.polyfit(smoothed_iso, smoothed_exp, 1)
    p = np.poly1d(coefs)
    y_fitted = p(smoothed_iso)
    delta_f = smoothed_exp - y_fitted
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_f_norm = np.divide(
            delta_f,
            y_fitted,
            out=np.full_like(delta_f, np.nan, dtype=np.float64),
            where=y_fitted != 0,
        )
    delta_f_zscore = zscore(delta_f_norm, nan_policy="omit")

    exp_mean = float(smoothed_exp.mean())
    iso_mean = float(smoothed_iso.mean())
    signal_ratio = exp_mean / iso_mean if iso_mean != 0 else float("inf")

    return {
        "time_array": time_array,
        "smoothed_exp": smoothed_exp,
        "smoothed_iso": smoothed_iso,
        "delta_f": delta_f,
        "delta_f_norm": delta_f_norm,
        "delta_f_zscore": delta_f_zscore,
        "coefs": coefs,
        "signal_ratio": signal_ratio,
        "exp_mean": exp_mean,
        "iso_mean": iso_mean,
    }


def _apply_smoothing(
    *,
    iso_data: NDArray[np.floating[Any]],
    exp_data: NDArray[np.floating[Any]],
    iso_fs: float,
    exp_fs: float,
    smoothing_method: str,
    smoothing_fraction: float,
    new_sampling_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if smoothing_method == "Smooth at downsampled points":
        time_array, smoothed_exp = smooth_at_downsampled_points(
            data=exp_data.astype(np.float32),
            fs_orig=np.float64(exp_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=np.float64(smoothing_fraction),
        )
        _, smoothed_iso = smooth_at_downsampled_points(
            data=iso_data.astype(np.float32),
            fs_orig=np.float64(iso_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=np.float64(smoothing_fraction),
        )
    elif smoothing_method == "Downsample then smooth":
        time_array, smoothed_exp, _ = downsample_then_smooth(
            data=exp_data.astype(np.float32),
            fs_orig=np.float64(exp_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=np.float64(smoothing_fraction),
        )
        _, smoothed_iso, _ = downsample_then_smooth(
            data=iso_data.astype(np.float32),
            fs_orig=np.float64(iso_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=np.float64(smoothing_fraction),
        )
    elif smoothing_method == "Downsampled only":
        time_array, smoothed_exp, _ = downsample_then_smooth(
            data=exp_data.astype(np.float32),
            fs_orig=np.float64(exp_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=None,
        )
        _, smoothed_iso, _ = downsample_then_smooth(
            data=iso_data.astype(np.float32),
            fs_orig=np.float64(iso_fs),
            fs_target=np.float64(new_sampling_rate),
            lowess_frac=None,
        )
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")

    return time_array, smoothed_exp, smoothed_iso
