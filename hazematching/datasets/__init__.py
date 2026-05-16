from .config import (
    DATASET_DESCRIPTIONS,
    SUBSETS,
    canonical_subset,
    format_subset_options,
)
from .data_norm import normalize, denormalize, STATS

__all__ = [
    "DATASET_DESCRIPTIONS",
    "SUBSETS",
    "STATS",
    "HazeDataset",
    "canonical_subset",
    "denormalize",
    "format_subset_options",
    "normalize",
]


def __getattr__(name: str):
    if name == "HazeDataset":
        from .haze import HazeDataset

        return HazeDataset

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
