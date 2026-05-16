"""Compute and plot calibration curves for HazeMatching inference results.

Usage:
    uv run python scripts/calibrate.py zebrafish --results-dir data/zebrafish
    uv run python scripts/calibrate.py microtubule --results-dir data/microtubule --output calibration.pdf
"""

from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tifffile import imread, TiffFile
from tqdm import tqdm

from hazematching.calibration import Calibration, plot_calibration
from hazematching.datasets import canonical_subset, format_subset_options
from hazematching.datasets.data_norm import normalize

app = typer.Typer()


def _load_split(
    results_dir: Path,
    data_dir: Path,
    subset: str,
    folder: str,
    n_samples: int,
    device: torch.device,
):
    """Load predictions and confocal targets for one split (val or test).

    Returns pred (N, H, W, 1), std (N, H, W, 1), target (N, H, W, 1).
    """
    results_path = results_dir / f"{folder}_results"
    data_path = data_dir / folder

    image_files = sorted(f for f in results_path.iterdir() if f.suffix == ".tif")

    pred_list, std_list, target_list = [], [], []
    for result_file in tqdm(image_files, desc=folder, leave=False):
        # Predictions: shape (n_samples, num_steps, H, W)
        with TiffFile(result_file) as tif:
            image = tif.asarray()
        samples = image[:n_samples, -1]  # (n_samples, H, W) — last ODE step

        # Confocal target: channel 0 of the original tif, normalized.
        raw_path = data_path / result_file.name
        raw = imread(raw_path).astype("float32")
        target_image = normalize(raw[0:1], subset, channel=0, path=raw_path).squeeze(0)

        mmse = np.mean(samples, axis=0, keepdims=True)  # (1, H, W)

        samples_t = torch.tensor(samples, device=device)
        std = torch.std(samples_t, dim=0, keepdim=True).cpu().numpy()  # (1, H, W)

        pred_list.append(mmse)
        std_list.append(std)
        target_list.append(target_image[np.newaxis])  # (1, H, W)

    # Stack and add channel dim for Calibration (expects shape N, H, W, C)
    pred = np.concatenate(pred_list, axis=0)[..., np.newaxis]
    std = np.concatenate(std_list, axis=0)[..., np.newaxis]
    target = np.concatenate(target_list, axis=0)[..., np.newaxis]
    return pred, std, target


@app.command()
def calibrate(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    results_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing val_results/ and test_results/ from infer.py."
        ),
    ],
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory containing subset folders.")
    ] = Path("data"),
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Path for the output PDF. Defaults to <results_dir>/calibration.pdf."
        ),
    ] = None,
    n_samples: Annotated[
        int, typer.Option(help="Number of posterior samples to use for MMSE/std.")
    ] = 50,
    num_bins: Annotated[
        int, typer.Option(help="Number of bins for calibration stats.")
    ] = 50,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if output is None:
        output = results_dir / "calibration.pdf"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_dir = data_dir / subset

    typer.echo("Loading val split...")
    pred_val, std_val, target_val = _load_split(
        results_dir, subset_dir, subset, "val", n_samples, device
    )

    typer.echo("Loading test split...")
    pred_test, std_test, target_test = _load_split(
        results_dir, subset_dir, subset, "test", n_samples, device
    )

    # Fit calibration factors on val
    typer.echo("Fitting calibration on val...", nl=False)
    calib = Calibration(num_bins=num_bins)
    _, factors = calib.get_calibrated_factor_for_stdev(pred_val, std_val, target_val)
    typer.echo(" done")

    scaled_std_test = np.clip(
        std_test * factors["scalar"] + factors["offset"], a_min=0, a_max=None
    )

    typer.echo("Computing test stats...", nl=False)
    calib_test = Calibration(num_bins=num_bins)
    stats_test = calib_test.compute_stats(pred_test, scaled_std_test, target_test)
    stats_test_unscaled = Calibration(num_bins=num_bins).compute_stats(
        pred_test, std_test, target_test
    )
    typer.echo(" done")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"{subset.upper()} Calibration", fontsize=18, fontweight="bold")
    plot_calibration(
        ax,
        "HazeMatching (scaled)",
        stats_test,
        show_identity=True,
        scaling_factor=factors["scalar"].item(),
        offset=factors["offset"].item(),
    )
    plot_calibration(ax, "HazeMatching (unscaled)", stats_test_unscaled)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved calibration plot to {output}")


if __name__ == "__main__":
    app()
