"""Per-dataset, per-channel normalisation statistics for HazeMatching.

Each entry maps a dataset key to ([mean_ch0, mean_ch1], [std_ch0, std_ch1]).
Channel 0 = confocal target; channel 1 = widefield input.
Zebrafish uses filename-derived source channels (ch0/ch1/ch2).
"""

import re
from pathlib import Path
from typing import Optional, Union

from .config import canonical_subset


ZEBRAFISH_STATS_BY_SOURCE_CHANNEL: dict[int, tuple[list[float], list[float]]] = {
    0: ([87.74069, 176.03], [51.070114, 39.342876]),
    1: ([53.154064, 155.55043], [30.28169, 23.21549]),
    2: ([62.870373, 162.67065], [32.791805, 27.638577]),
}

STATS: dict[str, tuple[list[float], list[float]]] = {
    "organoids1": ([493.80344, 1325.1595], [311.96014, 155.96614]),
    "organoids2": ([2480.7273, 7911.6943], [2576.4062, 6652.056]),
    "microtubule": ([50.924667, 145.08662], [31.688, 27.819897]),
    "neuron": ([37881.363, 101.10135], [116733.914, 2.7681978]),
}


def zebrafish_source_channel_from_path(path: Union[str, Path]) -> int:
    name = Path(path).name
    match = re.search(r"ch([0-2])", name)
    if match is None:
        raise ValueError(
            f"Could not infer zebrafish source channel from filename {name!r}; "
            "expected a token like 'ch0', 'ch1', or 'ch2'."
        )
    return int(match.group(1))


def _stats_for(
    dataset: str,
    source_channel: Optional[int] = None,
    path: Optional[Union[str, Path]] = None,
):
    dataset = canonical_subset(dataset)
    if dataset == "zebrafish":
        if source_channel is None:
            if path is None:
                raise ValueError(
                    "Zebrafish normalization requires a filename/path or "
                    "source_channel so the ch0/ch1/ch2-specific stats can be used."
                )
            source_channel = zebrafish_source_channel_from_path(path)
        try:
            return ZEBRAFISH_STATS_BY_SOURCE_CHANNEL[source_channel]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported zebrafish source channel {source_channel!r}; "
                "expected 0, 1, or 2."
            ) from exc
    return STATS[dataset]


def normalize(
    image,
    dataset: str,
    channel: int,
    source_channel: Optional[int] = None,
    path: Optional[Union[str, Path]] = None,
):
    mean, std = _stats_for(dataset, source_channel=source_channel, path=path)
    return (image - mean[channel]) / std[channel]


def denormalize(
    image,
    dataset: str,
    channel: int,
    source_channel: Optional[int] = None,
    path: Optional[Union[str, Path]] = None,
):
    mean, std = _stats_for(dataset, source_channel=source_channel, path=path)
    return image * std[channel] + mean[channel]
