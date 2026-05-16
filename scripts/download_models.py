"""Download pre-trained HazeMatching checkpoints.

Usage:
    uv run python scripts/download_models.py               # download all pre-trained models
    uv run python scripts/download_models.py --checkpoint-dir /my/path
    uv run python scripts/download_models.py --subset zebrafish --subset organoids1
"""

from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

from hazematching.datasets import DATASET_DESCRIPTIONS, SUBSETS, canonical_subset

BASE_URL = "https://download.fht.org/jug/hazematching/checkpoints/"

# Fill the hash field when the final checkpoints are published.
MODELS: dict[str, dict[str, str | None]] = {
    key: {"best_model.pth": None} for key in SUBSETS
}

app = typer.Typer()


def _download_subset(key: str, checkpoint_dir: Path) -> None:
    key = canonical_subset(key)
    dest = checkpoint_dir / key
    dest.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {DATASET_DESCRIPTIONS[key]} checkpoints -> {dest}/")
    for filename, known_hash in MODELS[key].items():
        pooch.retrieve(
            url=BASE_URL + f"{key}/{filename}",
            known_hash=known_hash,
            fname=filename,
            path=dest,
            progressbar=True,
        )
        typer.echo(f"  {filename}")


@app.command()
def main(
    checkpoint_dir: Annotated[
        Path, typer.Option(help="Directory to download checkpoints into.")
    ] = Path("checkpoints"),
    subset: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    try:
        keys = [canonical_subset(s) for s in subset] if subset else list(MODELS)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    for key in keys:
        _download_subset(key, checkpoint_dir)

    typer.echo("\nAll done. Checkpoint layout:")
    for key in keys:
        typer.echo(f"  {checkpoint_dir / key}/")
        for filename in MODELS[key]:
            typer.echo(f"    {filename}")


if __name__ == "__main__":
    app()
