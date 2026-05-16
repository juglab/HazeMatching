import os
import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
from tifffile import imread, imwrite
from tqdm import tqdm
import typer

from hazematching import CCFMUNet, odeint
from hazematching.datasets import canonical_subset, format_subset_options
from hazematching.datasets.data_norm import normalize, denormalize
from hazematching.utils import extract_patches_inner, reconstruct_image_inner

warnings.filterwarnings("ignore")

app = typer.Typer()

MICROTUBULE_CENTER_CROP = slice(56, -56)
NEURON_PATCH_SIZE = 64
NEURON_CROP_SIZE = 32


@app.command()
def infer(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    checkpoint: Annotated[Path, typer.Option(help="Path to model .pth checkpoint.")],
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory containing subset folders.")
    ] = Path("data"),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to write results. Defaults to <data_dir>/<subset_folder>/<folder>_results/."
        ),
    ] = None,
    folders: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Which split folders to run (e.g. test val). Default: test val."
        ),
    ] = None,
    n_samples: Annotated[
        int, typer.Option(help="Number of stochastic samples per image.")
    ] = 50,
    num_steps: Annotated[int, typer.Option(help="Number of ODE time steps.")] = 20,
    max_batch_size: Annotated[
        int, typer.Option(help="Max patches per ODE batch.")
    ] = 256,
    patch_size: Annotated[int, typer.Option(help="Patch size for tiling.")] = 128,
    crop_size: Annotated[int, typer.Option(help="Inner crop size per patch.")] = 64,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if folders is None:
        folders = ["test", "val"]

    if subset == "neuron":
        patch_size = NEURON_PATCH_SIZE
        crop_size = NEURON_CROP_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CCFMUNet(
        dim=(2, patch_size, patch_size),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    typer.echo(f"Loaded checkpoint: {checkpoint}")

    subset_dir = data_dir / subset
    ts = torch.linspace(0.0, 1.0, num_steps).to(device)

    for folder in folders:
        folder_dir = subset_dir / folder
        image_files = sorted(f for f in os.listdir(folder_dir) if f.endswith(".tif"))

        if output_dir is None:
            out = subset_dir / f"{folder}_results"
        else:
            out = output_dir / f"{folder}_results"
        out.mkdir(parents=True, exist_ok=True)

        for image_file in tqdm(image_files, desc=f"{folder}", unit="image"):
            img_path = folder_dir / image_file
            raw = imread(img_path).astype("float32")
            if subset == "microtubule":
                raw = raw[
                    :,
                    MICROTUBULE_CENTER_CROP,
                    MICROTUBULE_CENTER_CROP,
                ]
            noisy_norm = normalize(raw[1:2], subset, channel=1, path=img_path)
            patches, coords = extract_patches_inner(
                noisy_norm, patch_size=patch_size, crop_size=crop_size
            )
            condition = torch.from_numpy(patches).to(device)

            samples = []

            for _ in tqdm(range(n_samples), desc="samples", leave=False):
                image_tensor = np.zeros(
                    (num_steps, max_batch_size, 2, patch_size, patch_size),
                    dtype=np.float32,
                )
                noise = torch.randn_like(condition)
                input_tensor = torch.cat([noise, condition], dim=1)
                num_batches = (
                    input_tensor.size(0) + max_batch_size - 1
                ) // max_batch_size

                with torch.no_grad():
                    for i in range(num_batches):
                        batch = input_tensor[
                            i * max_batch_size : (i + 1) * max_batch_size
                        ]
                        traj, _ = odeint(
                            lambda t, x: model(t, x),
                            batch,
                            ts,
                            atol=1e-4,
                            rtol=1e-4,
                            method="euler",
                            condition=1,
                        )
                        image_tensor = np.concatenate(
                            [image_tensor, traj.cpu().numpy()], axis=1
                        )

                image_tensor = image_tensor[:, max_batch_size:]
                samples.append(image_tensor)

            samples = np.stack(samples, axis=0)
            full = reconstruct_image_inner(
                samples,
                coords,
                noisy_norm.shape,
                patch_size=patch_size,
                crop_size=crop_size,
            )

            if subset == "neuron":
                prediction = full[:, :, 0]
            else:
                prediction = denormalize(
                    full[:, :, 0], subset, channel=0, path=img_path
                )

            imwrite(
                out / image_file, prediction, imagej=True, metadata={"axes": "TZYX"}
            )


if __name__ == "__main__":
    app()
