import random
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from hazematching import CCFMFlowMatcher, CCFMUNet
from hazematching.datasets import (
    HazeDataset,
    canonical_subset,
    format_subset_options,
)

app = typer.Typer()

NEURON_IMAGE_SIZE = 64
DEFAULT_IMAGE_SIZE = 128


@app.command()
def train(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory containing subset folders.")
    ] = Path("data"),
    save_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save checkpoints. Defaults to ./checkpoints/<subset>."
        ),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 16,
    n_epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 200,
    lr: Annotated[float, typer.Option(help="Adam learning rate.")] = 1e-4,
    no_wandb: Annotated[
        bool, typer.Option("--no-wandb", help="Disable Weights & Biases logging.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = not no_wandb

    subset_dir = data_dir / subset
    if save_dir is None:
        save_dir = Path("checkpoints") / subset
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_set = HazeDataset(subset, subset_dir / "train_crop")
    val_set = HazeDataset(subset, subset_dir / "val_crop")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # ── Model ────────────────────────────────────────────────────────────────
    image_size = NEURON_IMAGE_SIZE if subset == "neuron" else DEFAULT_IMAGE_SIZE
    model = CCFMUNet(
        dim=(2, image_size, image_size),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"Parameters: {n_params / 1e6:.2f}M")

    FM = CCFMFlowMatcher(sigma=0.0)
    ts = torch.linspace(0.0, 1.0, 20).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    # ── W&B ─────────────────────────────────────────────────────────────────
    if use_wandb:
        experiment = wandb.init(
            project=subset,
            name="HazeMatching",
            resume="allow",
            anonymous="must",
            mode="online",
            reinit=True,
            save_code=True,
        )
        experiment.config.update(
            dict(
                subset=subset,
                learning_rate=lr,
                max_epochs=n_epochs,
                batch_size=batch_size,
                seed=seed,
            )
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # ── Training loop ────────────────────────────────────────────────────────
    try:
        for epoch in range(n_epochs):
            model.train()
            with tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch"
            ) as pbar:
                for data in pbar:
                    optimizer.zero_grad()
                    x0 = data[:, 1:2].to(device)
                    x1 = data[:, 0:1].to(device)
                    x0_noise = torch.randn_like(x0)
                    t = ts[torch.randint(0, len(ts), (x0.shape[0],))]
                    t, xt, ut = FM.sample_location_and_conditional_flow(
                        x0_noise, x1, t=t
                    )
                    xt = torch.cat([xt, x0], dim=1)
                    loss = criterion(model(t, xt), ut)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    if use_wandb:
                        wandb.log({"loss": loss.item()})

            model.eval()
            avg_val_loss = 0.0
            with (
                torch.no_grad(),
                tqdm(
                    val_loader, desc=f"  Val {epoch+1}/{n_epochs}", unit="batch"
                ) as pbar,
            ):
                for data in pbar:
                    x0 = data[:, 1:2].to(device)
                    x1 = data[:, 0:1].to(device)
                    x0_noise = torch.randn_like(x0)
                    t = ts[torch.randint(0, len(ts), (x0.shape[0],))]
                    t, xt, ut = FM.sample_location_and_conditional_flow(
                        x0_noise, x1, t=t
                    )
                    xt = torch.cat([xt, x0], dim=1)
                    loss = criterion(model(t, xt), ut)
                    avg_val_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    if use_wandb:
                        wandb.log({"val_loss": loss.item()})

            avg_val_loss /= len(val_loader)
            if round(avg_val_loss, 4) < round(best_loss, 4):
                typer.echo(
                    f"  Best model: val_loss={avg_val_loss:.4f} (was {best_loss:.4f}), epoch {epoch+1}"
                )
                best_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir / "best_model.pth")

    except KeyboardInterrupt:
        typer.echo("Interrupted — saving last checkpoint.")

    torch.save(model.state_dict(), save_dir / "last_model.pth")
    typer.echo(f"Done. Checkpoints saved to {save_dir}/")


if __name__ == "__main__":
    app()
