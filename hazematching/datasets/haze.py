from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tifffile import imread

from .config import SUBSETS, canonical_subset, format_subset_options
from .data_norm import normalize


class HazeDataset(data.Dataset):
    """HazeMatching dataset for paired widefield / confocal microscopy images.

    Args:
        subset: One of the HazeMatching dataset keys.
        folder: Directory containing image crops.
        returns: Channel indices to include in each sample, e.g. ``[0, 1]``
            for confocal target and widefield input.
    """

    def __init__(
        self,
        subset: str,
        folder: Path,
        returns: list[int] = [0, 1],
    ):
        super().__init__()
        try:
            subset = canonical_subset(subset)
        except ValueError as exc:
            raise ValueError(
                f"subset must be one of {format_subset_options()}, got {subset!r}"
            ) from exc

        self.subset = subset
        self.returns = returns

        self.paths = sorted(Path(folder).glob("*.tif"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = torch.from_numpy(imread(path).astype(np.float32))
        channels = [
            normalize(img[ch : ch + 1], self.subset, ch, path=path)
            for ch in self.returns
        ]
        return torch.cat(channels, dim=0)
