import os
import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
from microssim import MicroMS3IM
from tifffile import imread, TiffFile
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import typer

from hazematching.datasets import canonical_subset, format_subset_options
from hazematching.ra_psnr import RangeInvariantPsnr
from hazematching.utils import (
    lpips,
    fid_score,
    FSIM,
    extract_patches_inner_metrics,
    GMSD,
    entropy,
)

warnings.filterwarnings("ignore")

app = typer.Typer()


@app.command()
def compute_metrics(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    results_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory containing inference .tif results. Defaults to <data_dir>/<subset>/test_results/."
        ),
    ] = None,
    fid_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory of FID reference crops. Defaults to <data_dir>/<subset>/train_crops_fid (train_crop for neuron data)/."
        ),
    ] = None,
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory (used to resolve defaults).")
    ] = Path("data"),
    n_samples: Annotated[
        int, typer.Option(help="Number of samples to average for MMSE prediction.")
    ] = 50,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    subset_dir = data_dir / subset
    if results_dir is None:
        results_dir = subset_dir / "test_results"
    if fid_dir is None:
        # The 'neuron' dataset uses a different folder name for FID references
        if subset == "neuron":
            fid_dir = subset_dir / "train_crop"
        else:
            fid_dir = subset_dir / "train_crops_fid"

    micros_ms3im = MicroMS3IM()

    # ── Load FID reference crops ─────────────────────────────────────────────
    fid_files = sorted(f for f in os.listdir(fid_dir) if f.endswith(".tif"))
    fid_crops = []
    for fid_file in tqdm(fid_files, desc="Loading FID crops", leave=False):
        with TiffFile(fid_dir / fid_file) as tif:
            fid_crops.append(tif.asarray())
    fid_crops = np.concatenate(fid_crops, axis=0)
    fid_crops_gts = torch.from_numpy(fid_crops).unsqueeze(1)
    typer.echo(f"Using {fid_crops.shape[0]} crops for FID.")

    image_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".tif"))

    psnr_values, ms_ssim_scores, micro3_ssim_scores = [], [], []
    gts, outputs, gts_full, outputs_full = [], [], [], []
    ind_fsims, ind_lpips, ind_fids, ind_gmsd = [], [], [], []

    typer.echo(
        f"Computing metrics over {len(image_files)} images (MMSE n={n_samples})..."
    )

    gt_dir = subset_dir / "test"

    for image_file in tqdm(image_files, desc="Images", leave=False):
        with TiffFile(results_dir / image_file) as tif:
            image = tif.asarray()

        image_pred = image[:n_samples, -1]  # (n_samples, H, W)
        image_gt = imread(gt_dir / image_file).astype("float32")[0:1]  # (1, H, W)
        mmse_pred = np.mean(image_pred, axis=0, keepdims=True)

        # PSNR + MS-SSIM
        psnr_values.append(RangeInvariantPsnr(image_gt, mmse_pred))
        ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=3, data_range=1.0, betas=(0.0448, 0.2856, 0.3001)
        )
        ms_ssim_scores.append(
            ms_ssim_metric(
                torch.from_numpy(mmse_pred).unsqueeze(0),
                torch.from_numpy(image_gt).unsqueeze(0),
            )
        )

        mmse_patches, _ = extract_patches_inner_metrics(mmse_pred, 64)
        gt_patches, _ = extract_patches_inner_metrics(image_gt, 64)
        gts.append(torch.from_numpy(gt_patches))
        outputs.append(torch.from_numpy(mmse_patches))
        gts_full.append(torch.from_numpy(image_gt).unsqueeze(1))
        outputs_full.append(torch.from_numpy(mmse_pred).unsqueeze(1))

        # Per-sample perceptual metrics
        torch_gt_patches = torch.from_numpy(gt_patches)
        valid = [
            i for i in range(torch_gt_patches.shape[0]) if torch_gt_patches[i].max() > 0
        ]
        torch_gt_patches = torch_gt_patches[valid]

        image_fsims, image_lpips_, image_fids_, image_gmsd_ = [], [], [], []
        for j in range(image_pred.shape[0]):
            pred_patches, _ = extract_patches_inner_metrics(image_pred[j : j + 1], 64)
            torch_pred = torch.from_numpy(pred_patches)[valid]
            image_fsims.append(FSIM(torch_pred, torch_gt_patches))
            image_lpips_.append(lpips(torch_gt_patches, torch_pred))
            image_fids_.append(fid_score(fid_crops_gts, torch_pred))
            image_gmsd_.append(GMSD(torch_pred, torch_gt_patches))

        if image_fsims:
            ind_fsims.append(torch.mean(torch.stack(image_fsims)))
            ind_lpips.append(torch.mean(torch.tensor(image_lpips_)))
            ind_fids.append(torch.mean(torch.tensor(image_fids_)))
            ind_gmsd.append(torch.mean(torch.stack(image_gmsd_)))

    # ── Aggregate ────────────────────────────────────────────────────────────
    gts = torch.cat(gts, dim=0)
    outputs = torch.cat(outputs, dim=0)
    gts_full = torch.cat(gts_full, dim=0)
    outputs_full = torch.cat(outputs_full, dim=0)

    average_psnr = sum(psnr_values) / len(psnr_values)
    std_psnr = torch.std(torch.stack(psnr_values))
    average_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    std_ms_ssim = torch.std(torch.stack(ms_ssim_scores))

    fsim_scores = FSIM(outputs, gts)
    fsim_mean = torch.mean(fsim_scores)
    lpips_score = lpips(gts, outputs)
    fid = fid_score(fid_crops_gts, outputs)
    gmsd_scores = GMSD(outputs, gts)
    gmsd_mean = torch.mean(gmsd_scores)
    entropy_scores = entropy(outputs)

    average_ind_fsim = torch.mean(torch.stack(ind_fsims))
    std_ind_fsim = torch.std(torch.stack(ind_fsims))
    average_ind_lpips = torch.mean(torch.tensor(ind_lpips))
    std_ind_lpips = torch.std(torch.tensor(ind_lpips))
    average_ind_fid = torch.mean(torch.tensor(ind_fids))
    std_ind_fid = torch.std(torch.tensor(ind_fids))
    average_ind_gmsd = torch.mean(torch.stack(ind_gmsd))
    std_ind_gmsd = torch.std(torch.stack(ind_gmsd))

    # MicroMS3IM
    gts_np = gts_full.numpy()
    outs_np = outputs_full.numpy()
    micros_ms3im.fit(gts_np[:, 0], outs_np[:, 0])
    micro3_ssim_scores = [
        micros_ms3im.score(gts_np[i, 0], outs_np[i, 0], betas=(0.0448, 0.2856, 0.3001))
        for i in range(gts_np.shape[0])
    ]
    average_micro3_ssim = np.mean(micro3_ssim_scores)
    std_micro3_ssim = np.std(micro3_ssim_scores)

    # ── Print ────────────────────────────────────────────────────────────────
    typer.echo(f"\n=== {subset.upper()} (n={n_samples}) ===")
    typer.echo(f"PSNR:         {average_psnr.item():.4f} ± {std_psnr.item():.4f}")
    typer.echo(
        f"MS-SSIM:      {average_ms_ssim.item():.4f} ± {std_ms_ssim.item():.4f}"
    )
    typer.echo(f"MicroMS3IM:   {average_micro3_ssim:.4f} ± {std_micro3_ssim:.4f}")
    typer.echo(f"FSIM  (MMSE): {fsim_mean.item():.4f}")
    typer.echo(
        f"FSIM  (Ind):  {average_ind_fsim.item():.4f} ± {std_ind_fsim.item():.4f}"
    )
    typer.echo(f"GMSD  (MMSE): {gmsd_mean.item():.4f}")
    typer.echo(
        f"GMSD  (Ind):  {average_ind_gmsd.item():.4f} ± {std_ind_gmsd.item():.4f}"
    )
    typer.echo(f"LPIPS (MMSE): {lpips_score:.4f}")
    typer.echo(
        f"LPIPS (Ind):  {average_ind_lpips.item():.4f} ± {std_ind_lpips.item():.4f}"
    )
    typer.echo(f"FID   (MMSE): {fid:.4f}")
    typer.echo(f"FID   (Ind):  {average_ind_fid.item():.4f} ± {std_ind_fid.item():.4f}")

    # LaTeX rows
    name = "\\textbf{HazeMatching}"
    typer.echo("\n--- LaTeX (MMSE + Ind, supplemental) ---")
    typer.echo(
        f"& {name} & "
        f"\\makecell{{{average_psnr.item():.2f} \\\\ {std_psnr.item():.3f}}} & "
        f"\\makecell{{{average_micro3_ssim:.3f} \\\\ {std_micro3_ssim:.4f}}} & "
        f"\\makecell{{{lpips_score:.3f}}} & "
        f"\\makecell{{{fid:.3f}}} & "
        f"\\makecell{{{average_ind_lpips.item():.3f} \\\\ {std_ind_lpips.item():.4f}}} & "
        f"\\makecell{{{average_ind_fid.item():.3f} \\\\ {std_ind_fid.item():.4f}}} \\\\ \\cline{{2-7}}"
    )


if __name__ == "__main__":
    app()
