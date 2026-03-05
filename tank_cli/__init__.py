"""Public package surface for the TDT tank CLI."""

from .cli import (
    DEFAULT_NEW_SAMPLING_RATE,
    DEFAULT_SMOOTHING_FRACTION,
    DEFAULT_SMOOTHING_METHOD,
    DEFAULT_TTL_START_OFFSET,
    build_parser,
    main,
)

__all__ = [
    "main",
    "build_parser",
    "DEFAULT_SMOOTHING_METHOD",
    "DEFAULT_SMOOTHING_FRACTION",
    "DEFAULT_NEW_SAMPLING_RATE",
    "DEFAULT_TTL_START_OFFSET",
]
