import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/hazematching_walkthrough.ipynb")


def _lines(text: str) -> list[str]:
    text = dedent(text).strip("\n")
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
            # HazeMatching Walkthrough for Biologists and Microscopists

            This notebook is a guided, beginner-friendly example of the full **HazeMatching** workflow.
            It is designed for readers who may be new to Python but want to understand what the model is doing at each step.

            We will do four practical things:

            1. Look at the microscopy data and understand what the model sees.
            2. Run a **tiny training demo** for only 2 to 3 epochs, just to watch the loss change.
            3. Load a **pre-trained model** and run inference on a small number of validation and test images.
            4. Compute simple quality metrics, make a calibration plot, and visualize the input, ground truth, MMSE reconstruction, and posterior samples.

            This notebook is intentionally light.
            It is **not** meant to reproduce the full paper numbers.
            Instead, it is meant to make the workflow understandable and easy to adapt.
            """
        ),
        md(
            """
            ## Before You Start

            This example assumes that the repository dependencies are installed and that you have access to the HazeMatching data and the HazeMatching checkpoints.

            A few important notes:

            - In this repository's actual training and inference code, **channel 1 is used as the observed noisy / low-resolution input**.
            - **Channel 0 is used as the clean / high-resolution target**.
            - That convention is what we follow in this notebook, even if some older comments in the code suggest the opposite.

            Throughout the notebook, we also include a few teaching figures from `notebooks/figures/`.
            These are there to give a visual explanation before readers look at the code.
            """
        ),
        md(
            """
            ## Step 1: Set Up the Notebook

            The next cell does the basic notebook setup.

            It:

            - finds the repository root automatically,
            - adds the repository to Python's import path,
            - defines all user-editable settings in one place,
            - chooses CPU or GPU,
            - and fixes random seeds so repeated runs are easier to compare.

            If you only want to change one thing, the most important setting is `SUBSET`.
            """
        ),
        code(
            """
            from __future__ import annotations

            import copy
            import random
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            from IPython.display import display
            from microssim import MicroMS3IM
            from tifffile import imread
            from torch.utils.data import DataLoader
            from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
            from tqdm.auto import tqdm


            def find_repo_root(start: Path) -> Path:
                current = start.resolve()
                for candidate in [current, *current.parents]:
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                raise FileNotFoundError("Could not find the repository root.")


            REPO_ROOT = find_repo_root(Path.cwd())
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from hazematching import CCFMFlowMatcher, CCFMUNet, odeint
            from hazematching.calibration import Calibration, plot_calibration
            from hazematching.datasets import HazeDataset
            from hazematching.datasets.data_norm import denormalize, normalize
            from hazematching.ra_psnr import RangeInvariantPsnr
            from hazematching.utils import extract_patches_inner, reconstruct_image_inner

            DATA_DIR = REPO_ROOT / "data"
            CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
            NOTEBOOK_OUTPUT_DIR = REPO_ROOT / "notebooks" / "outputs"
            NOTEBOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            SUBSET = "zebrafish"
            SEED = 42

            TRAIN_EPOCHS = 3
            TRAIN_BATCH_SIZE = 8
            TINY_TRAIN_MAX_BATCHES = 20
            TINY_VAL_MAX_BATCHES = 8

            NUM_INFERENCE_IMAGES = 3
            NUM_POSTERIOR_SAMPLES = 8
            NUM_ODE_STEPS = 20
            PATCH_SIZE = 128
            CROP_SIZE = 64
            MAX_PATCH_BATCH = 256

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            print(f"Repository root : {REPO_ROOT}")
            print(f"Dataset subset  : {SUBSET}")
            print(f"Compute device  : {DEVICE}")
            print(f"Data folder     : {DATA_DIR / SUBSET}")
            print(f"Checkpoint file : {CHECKPOINT_DIR / SUBSET / 'best_model.pth'}")
            """
        ),
        md(
            """
            ### Visual Intuition: Convex Combination During Training

            The training objective is based on intermediate states between random noise and the target image.
            If you are presenting this to a non-technical audience, it often helps to show the picture first and only then discuss the training cell.

            ![Convex combination overview](figures/convex_combination.png)
            """
        ),
        md(
            """
            ## Step 2: Make Sure the Data and Checkpoint Exist

            The next cell is a convenience cell.

            By default it **does not download anything**.
            It simply checks whether the expected data folder and pre-trained checkpoint are already present.

            If you want the notebook to download them for you, change either flag to `True`.
            """
        ),
        code(
            """
            DOWNLOAD_DATA_IF_MISSING = False
            DOWNLOAD_MODEL_IF_MISSING = False

            subset_dir = DATA_DIR / SUBSET
            checkpoint_path = CHECKPOINT_DIR / SUBSET / "best_model.pth"

            if DOWNLOAD_DATA_IF_MISSING and not subset_dir.exists():
                subprocess.run(
                    [sys.executable, "scripts/download_data.py", "--subset", SUBSET],
                    cwd=REPO_ROOT,
                    check=True,
                )

            if DOWNLOAD_MODEL_IF_MISSING and not checkpoint_path.exists():
                subprocess.run(
                    [sys.executable, "scripts/download_models.py", "--subset", SUBSET],
                    cwd=REPO_ROOT,
                    check=True,
                )

            print(f"Data available       : {subset_dir.exists()}")
            print(f"Checkpoint available : {checkpoint_path.exists()}")

            if not subset_dir.exists():
                raise FileNotFoundError(
                    f"Could not find {subset_dir}. Download the dataset or set the correct path."
                )

            if not checkpoint_path.exists():
                print(
                    "Pre-trained checkpoint not found yet. That is okay if you only want to inspect the data "
                    "or run the tiny training demo first."
                )
            """
        ),
        md(
            """
            ## Step 3: Look at the Data

            Before touching the model, it helps to see what one example actually looks like.

            The next cell does two things:

            - it loads one **training crop**, which is the small patch used during optimization,
            - and it loads one **full test image**, which is the larger image used for inference.

            We show the observed noisy input and the clean target side by side.
            This is often the most useful first picture for collaborators who are new to the project.
            """
        ),
        code(
            """
            train_set = HazeDataset(SUBSET, DATA_DIR / SUBSET / "train_crop")
            val_set = HazeDataset(SUBSET, DATA_DIR / SUBSET / "val_crop")

            train_example = train_set[0].numpy()
            first_test_file = sorted((DATA_DIR / SUBSET / "test").glob("*.tif"))[0]
            full_example = imread(first_test_file).astype(np.float32)

            fig, axes = plt.subplots(2, 2, figsize=(10, 9))

            axes[0, 0].imshow(train_example[1], cmap="magma")
            axes[0, 0].set_title("Training crop: observed noisy input")
            axes[0, 1].imshow(train_example[0], cmap="magma")
            axes[0, 1].set_title("Training crop: clean target")

            axes[1, 0].imshow(full_example[1], cmap="magma")
            axes[1, 0].set_title(f"Full image input: {first_test_file.name}")
            axes[1, 1].imshow(full_example[0], cmap="magma")
            axes[1, 1].set_title("Full image target")

            for ax in axes.ravel():
                ax.axis("off")

            plt.tight_layout()
            plt.show()

            print(f"Number of training crops : {len(train_set)}")
            print(f"Number of validation crops: {len(val_set)}")
            print(f"Full image shape          : {full_example.shape}")
            """
        ),
        md(
            """
            ## Step 4: Build the Model and the Tiny Training Demo

            The goal of the next cell is not to train the final scientific model.
            Instead, it prepares a **small educational training loop** that closely follows `scripts/train.py`.

            This helper code:

            - creates the HazeMatching U-Net,
            - prepares a simple training loop,
            - runs only a limited number of mini-batches per epoch to keep things light,
            - records train and validation loss after each epoch,
            - and saves the best tiny-demo checkpoint inside `notebooks/outputs/`.

            If you are presenting the project to non-coders, this is usually the key cell to explain slowly.
            """
        ),
        md(
            """
            ### Visual Intuition: What Happens During Training

            This figure can be read as a companion to the tiny training loop below.
            It helps explain what the model receives as input and what it is asked to predict.

            ![Training overview](figures/training.png)
            """
        ),
        code(
            """
            def build_model() -> CCFMUNet:
                model = CCFMUNet(
                    dim=(2, 128, 128),
                    num_channels=32,
                    out_channels=1,
                    num_res_blocks=1,
                ).to(DEVICE)
                return model


            def make_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
                train_loader = DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_set,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )
                return train_loader, val_loader


            def training_step(model, matcher, criterion, batch, ts):
                target = batch[:, 0:1].to(DEVICE)
                condition = batch[:, 1:2].to(DEVICE)
                noise = torch.randn_like(target)
                t = ts[torch.randint(0, len(ts), (target.shape[0],), device=DEVICE)]
                t, xt, ut = matcher.sample_location_and_conditional_flow(noise, target, t=t)
                xt = torch.cat([xt, condition], dim=1)
                prediction = model(t, xt)
                loss = criterion(prediction, ut)
                return loss


            def run_tiny_training_demo(
                epochs: int = TRAIN_EPOCHS,
                batch_size: int = TRAIN_BATCH_SIZE,
                lr: float = 1e-4,
                max_train_batches: int = TINY_TRAIN_MAX_BATCHES,
                max_val_batches: int = TINY_VAL_MAX_BATCHES,
            ):
                model = build_model()
                matcher = CCFMFlowMatcher(sigma=0.0)
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                ts = torch.linspace(0.0, 1.0, NUM_ODE_STEPS, device=DEVICE)
                train_loader, val_loader = make_loaders(batch_size=batch_size)

                history = {"train_loss": [], "val_loss": []}
                best_val = float("inf")
                best_state = copy.deepcopy(model.state_dict())

                for epoch in range(epochs):
                    model.train()
                    running_train = []
                    for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - train")):
                        if batch_index >= max_train_batches:
                            break
                        optimizer.zero_grad()
                        loss = training_step(model, matcher, criterion, batch, ts)
                        loss.backward()
                        optimizer.step()
                        running_train.append(loss.item())

                    model.eval()
                    running_val = []
                    with torch.no_grad():
                        for batch_index, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - val")):
                            if batch_index >= max_val_batches:
                                break
                            loss = training_step(model, matcher, criterion, batch, ts)
                            running_val.append(loss.item())

                    mean_train = float(np.mean(running_train))
                    mean_val = float(np.mean(running_val))
                    history["train_loss"].append(mean_train)
                    history["val_loss"].append(mean_val)

                    if mean_val < best_val:
                        best_val = mean_val
                        best_state = copy.deepcopy(model.state_dict())

                    print(
                        f"Epoch {epoch + 1:02d} | train loss = {mean_train:.4f} | "
                        f"val loss = {mean_val:.4f}"
                    )

                model.load_state_dict(best_state)
                tiny_checkpoint = NOTEBOOK_OUTPUT_DIR / f"{SUBSET}_tiny_demo_model.pth"
                torch.save(model.state_dict(), tiny_checkpoint)
                return model, history, tiny_checkpoint


            def plot_training_history(history: dict[str, list[float]]) -> None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(history["train_loss"], marker="o", label="train loss")
                ax.plot(history["val_loss"], marker="o", label="validation loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE loss")
                ax.set_title("Tiny training demo")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Step 5: Optionally Run a Tiny 2 to 3 Epoch Training Demo

            This is the cell to run if you want people to **see training happen**.

            A few things to keep in mind:

            - this is intentionally small and fast,
            - the result is only for demonstration,
            - and the pre-trained checkpoint will still give better reconstructions.

            If you want to skip training and move directly to inference, set `RUN_TINY_TRAINING = False`.
            """
        ),
        code(
            """
            RUN_TINY_TRAINING = True

            tiny_model = None
            tiny_history = None
            tiny_checkpoint = NOTEBOOK_OUTPUT_DIR / f"{SUBSET}_tiny_demo_model.pth"

            if RUN_TINY_TRAINING:
                tiny_model, tiny_history, tiny_checkpoint = run_tiny_training_demo()
                plot_training_history(tiny_history)
                print(f"Tiny demo checkpoint saved to: {tiny_checkpoint}")
            else:
                print("Tiny training was skipped.")
            """
        ),
        md(
            """
            ## Step 6: Prepare Inference Helpers

            The next cell defines helper functions for posterior inference.

            In plain language, the code below:

            - loads a checkpoint,
            - cuts a large microscopy image into overlapping patches,
            - starts each patch from random noise,
            - uses the conditioning image to guide the denoising / reconstruction process,
            - samples multiple possible outputs from the posterior,
            - and stitches those patch predictions back into a full image.

            This is the most important cell for understanding how HazeMatching generates both a prediction and an uncertainty estimate.
            """
        ),
        md(
            """
            ### Visual Intuition: Iterative Inference

            In inference, the model does not jump directly from noise to the final prediction.
            Instead, it takes many small steps, gradually refining the reconstruction while being guided by the observed input image.

            ![Iterative inference overview](figures/iterative_inference.png)
            """
        ),
        code(
            """
            def load_checkpoint_model(path: Path) -> CCFMUNet:
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {path}")
                model = build_model()
                state = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state)
                model.eval()
                return model


            def predict_posterior_for_image(
                model: CCFMUNet,
                raw_image: np.ndarray,
                subset: str,
                image_path = None,
                n_samples: int = NUM_POSTERIOR_SAMPLES,
                num_steps: int = NUM_ODE_STEPS,
                patch_size: int = PATCH_SIZE,
                crop_size: int = CROP_SIZE,
                max_patch_batch: int = MAX_PATCH_BATCH,
            ) -> dict:
                noisy_raw = raw_image[1:2].astype(np.float32)
                target_raw = raw_image[0:1].astype(np.float32)
                noisy_norm = normalize(noisy_raw, subset, channel=1, path=image_path)
                target_norm = normalize(target_raw, subset, channel=0, path=image_path)

                patches, coords = extract_patches_inner(
                    noisy_norm,
                    patch_size=patch_size,
                    crop_size=crop_size,
                )
                condition = torch.from_numpy(patches).float().to(DEVICE)
                ts = torch.linspace(0.0, 1.0, num_steps, device=DEVICE)

                posterior_trajectories = []
                with torch.no_grad():
                    for _ in tqdm(range(n_samples), desc="Posterior samples", leave=False):
                        sample_batches = []
                        model_input = torch.cat([torch.randn_like(condition), condition], dim=1)

                        for start in range(0, model_input.size(0), max_patch_batch):
                            batch = model_input[start : start + max_patch_batch]
                            traj, _ = odeint(
                                lambda t, x: model(t, x),
                                batch,
                                ts,
                                atol=1e-4,
                                rtol=1e-4,
                                method="euler",
                                condition=1,
                            )
                            sample_batches.append(traj.cpu().numpy())

                        posterior_trajectories.append(np.concatenate(sample_batches, axis=1))

                posterior_trajectories = np.stack(posterior_trajectories, axis=0)
                posterior_full = reconstruct_image_inner(
                    posterior_trajectories,
                    coords,
                    noisy_norm.shape,
                    patch_size=patch_size,
                    crop_size=crop_size,
                )

                posterior_norm = posterior_full[:, :, 0]
                posterior_raw = denormalize(
                    posterior_norm,
                    subset,
                    channel=0,
                    path=image_path,
                )

                mmse_norm = posterior_norm[:, -1].mean(axis=0)
                mmse_raw = posterior_raw[:, -1].mean(axis=0)
                std_norm = posterior_norm[:, -1].std(axis=0)

                return {
                    "input_raw": noisy_raw[0],
                    "gt_raw": target_raw[0],
                    "input_norm": noisy_norm[0],
                    "gt_norm": target_norm[0],
                    "posterior_norm": posterior_norm,
                    "posterior_raw": posterior_raw,
                    "mmse_norm": mmse_norm,
                    "mmse_raw": mmse_raw,
                    "std_norm": std_norm,
                }


            def run_demo_inference(
                model: CCFMUNet,
                subset: str,
                splits: tuple[str, ...] = ("val", "test"),
                max_images: int = NUM_INFERENCE_IMAGES,
            ) -> dict[str, list[dict]]:
                results = {}
                for split in splits:
                    split_dir = DATA_DIR / subset / split
                    image_paths = sorted(split_dir.glob("*.tif"))[:max_images]
                    split_results = []

                    for image_path in tqdm(image_paths, desc=f"{split} images"):
                        raw_image = imread(image_path).astype(np.float32)
                        result = predict_posterior_for_image(
                            model,
                            raw_image,
                            subset,
                            image_path=image_path,
                        )
                        result["image_name"] = image_path.name
                        split_results.append(result)

                    results[split] = split_results
                return results
            """
        ),
        md(
            """
            ## Step 7: Choose Which Checkpoint to Use and Run Small Inference

            This cell lets you choose between:

            - the **tiny demo checkpoint** you just trained,
            - or the **pre-trained checkpoint** shipped for the selected subset.

            For teaching, it is often nice to compare both.
            For image quality, the pre-trained checkpoint is usually the better choice.
            """
        ),
        code(
            """
            USE_TINY_CHECKPOINT_FOR_INFERENCE = False

            if USE_TINY_CHECKPOINT_FOR_INFERENCE:
                selected_checkpoint = tiny_checkpoint
            else:
                selected_checkpoint = checkpoint_path

            inference_model = load_checkpoint_model(selected_checkpoint)
            posterior_results = run_demo_inference(
                inference_model,
                subset=SUBSET,
                splits=("val", "test"),
                max_images=NUM_INFERENCE_IMAGES,
            )

            print(f"Checkpoint used for inference: {selected_checkpoint}")
            print(
                "Images processed per split:",
                {split: len(items) for split, items in posterior_results.items()},
            )
            """
        ),
        md(
            """
            ## Step 8: Compute Quick Metrics on the Small Inference Set

            The paper uses a larger evaluation protocol.
            Here we keep things intentionally simple and light.

            The next cell computes a few easy-to-explain quantities on the **MMSE prediction**:

            - **Range-Invariant PSNR**: a reconstruction quality score,
            - **MS-SSIM**: a structural similarity score,
            - **MicroMS3IM**: a microscopy-focused perceptual similarity score.

            Because we only infer on a few images here, these numbers are best understood as a demonstration rather than a final benchmark.
            """
        ),
        code(
            """
            def to_unit_interval(image: np.ndarray) -> np.ndarray:
                image = np.asarray(image, dtype=np.float32)
                span = float(image.max() - image.min())
                if span < 1e-8:
                    return np.zeros_like(image)
                return (image - image.min()) / span


            def compute_quick_metrics(results: dict[str, list[dict]]):
                rows = []
                gt_bank = []
                pred_bank = []
                ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
                    kernel_size=3,
                    data_range=1.0,
                    betas=(0.0448, 0.2856, 0.3001),
                )

                for split, items in results.items():
                    for item in items:
                        gt = item["gt_raw"][None, :, :]
                        pred = item["mmse_raw"][None, :, :]

                        psnr_value = float(RangeInvariantPsnr(gt, pred))
                        gt_tensor = torch.from_numpy(to_unit_interval(gt)[None, :, :, :])
                        pred_tensor = torch.from_numpy(to_unit_interval(pred)[None, :, :, :])
                        ms_ssim_value = float(ms_ssim_metric(pred_tensor, gt_tensor))

                        rows.append(
                            {
                                "split": split,
                                "image_name": item["image_name"],
                                "psnr": psnr_value,
                                "ms_ssim": ms_ssim_value,
                            }
                        )
                        gt_bank.append(item["gt_raw"])
                        pred_bank.append(item["mmse_raw"])

                gt_bank = np.stack(gt_bank, axis=0)
                pred_bank = np.stack(pred_bank, axis=0)

                micros = MicroMS3IM()
                micros.fit(gt_bank, pred_bank)
                for row, gt_image, pred_image in zip(rows, gt_bank, pred_bank):
                    row["micro_ms3im"] = float(
                        micros.score(
                            gt_image,
                            pred_image,
                            betas=(0.0448, 0.2856, 0.3001),
                        )
                    )

                return rows


            quick_metric_rows = compute_quick_metrics(posterior_results)

            for row in quick_metric_rows:
                print(
                    f"{row['split']:>4} | {row['image_name']} | "
                    f"PSNR={row['psnr']:.3f} | "
                    f"MS-SSIM={row['ms_ssim']:.3f} | "
                    f"MicroMS3IM={row['micro_ms3im']:.3f}"
                )

            print("\\nMean values by split")
            for split in sorted(posterior_results):
                split_rows = [row for row in quick_metric_rows if row["split"] == split]
                print(
                    f"{split:>4} | "
                    f"PSNR={np.mean([row['psnr'] for row in split_rows]):.3f} | "
                    f"MS-SSIM={np.mean([row['ms_ssim'] for row in split_rows]):.3f} | "
                    f"MicroMS3IM={np.mean([row['micro_ms3im'] for row in split_rows]):.3f}"
                )
            """
        ),
        md(
            """
            ## Step 9: Create a Calibration Plot

            One of the strengths of HazeMatching is that it produces a **distribution of possible reconstructions**, not just one image.

            That lets us estimate uncertainty.

            The next cell uses:

            - the **validation images** to fit a simple linear calibration of the predicted standard deviation,
            - and the **test images** to visualize how well that uncertainty matches the actual reconstruction error.

            If you only use a very small number of images, the curve may look noisy.
            That is expected for a lightweight demo.
            """
        ),
        md(
            """
            ### Visual Intuition: Posterior Sampling

            HazeMatching produces multiple plausible reconstructions for the same input image.
            This figure is useful when introducing the idea that uncertainty is estimated from variation across those posterior samples.

            <img src="figures/posterior_sampling.png" alt="Posterior sampling overview" width="60%">
            """
        ),
        code(
            """
            def stack_for_calibration(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                pred = np.stack([item["mmse_norm"] for item in items], axis=0)[..., np.newaxis]
                std = np.stack([item["std_norm"] for item in items], axis=0)[..., np.newaxis]
                target = np.stack([item["gt_norm"] for item in items], axis=0)[..., np.newaxis]
                return pred, std, target


            pred_val, std_val, target_val = stack_for_calibration(posterior_results["val"])
            pred_test, std_test, target_test = stack_for_calibration(posterior_results["test"])

            calibration = Calibration(num_bins=25)
            _, factors = calibration.get_calibrated_factor_for_stdev(pred_val, std_val, target_val)
            scaled_std_test = np.clip(
                std_test * factors["scalar"] + factors["offset"],
                a_min=0.0,
                a_max=None,
            )

            stats_scaled = Calibration(num_bins=25).compute_stats(pred_test, scaled_std_test, target_test)
            stats_unscaled = Calibration(num_bins=25).compute_stats(pred_test, std_test, target_test)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_title(f"{SUBSET.upper()} calibration demo")
            plot_calibration(
                ax,
                "HazeMatching (scaled)",
                stats_scaled,
                show_identity=True,
                scaling_factor=factors["scalar"].item(),
                offset=factors["offset"].item(),
            )
            plot_calibration(ax, "HazeMatching (unscaled)", stats_unscaled)
            plt.show()

            print("Learned scalar :", float(factors["scalar"].item()))
            print("Learned offset :", float(factors["offset"].item()))
            """
        ),
        md(
            """
            ## Step 10: Visualize One Example in the Most Intuitive Way

            The next cell creates the main summary figure that many readers care about most.

            It shows:

            - the noisy input image,
            - the ground-truth target,
            - the MMSE reconstruction,
            - and two posterior samples.

            This is a very useful figure for talks, lab meetings, and manuscript supplements because it shows both the average prediction and the variability of the posterior.
            """
        ),
        code(
            """
            EXAMPLE_SPLIT = "test"
            EXAMPLE_INDEX = 0
            POSTERIOR_SAMPLE_A = 0
            POSTERIOR_SAMPLE_B = 1

            example = posterior_results[EXAMPLE_SPLIT][EXAMPLE_INDEX]
            vmin = min(example["gt_raw"].min(), example["mmse_raw"].min())
            vmax = max(example["gt_raw"].max(), example["mmse_raw"].max())

            panels = [
                ("Input image", example["input_raw"]),
                ("Ground truth", example["gt_raw"]),
                ("MMSE", example["mmse_raw"]),
                (f"Posterior {POSTERIOR_SAMPLE_A + 1}", example["posterior_raw"][POSTERIOR_SAMPLE_A, -1]),
                (f"Posterior {POSTERIOR_SAMPLE_B + 1}", example["posterior_raw"][POSTERIOR_SAMPLE_B, -1]),
            ]

            fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
            for ax, (title, image) in zip(axes, panels):
                ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")

            plt.suptitle(f"{EXAMPLE_SPLIT} example: {example['image_name']}", y=1.02)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Where to Extend This Notebook

            This notebook was written to be understandable first and exhaustive second.

            Good next extensions would be:

            - increase `NUM_INFERENCE_IMAGES` and `NUM_POSTERIOR_SAMPLES`,
            - switch to another subset such as `microtubule` or `organoids1`,
            - insert your custom explanatory figures between the markdown sections,
            - compare the tiny checkpoint and the pre-trained checkpoint side by side,
            - and, for full paper-style evaluation, run the repository scripts on the complete dataset.

            If you want, this notebook can also be turned into a more polished teaching document with callout boxes, saved output figures, and a short glossary for microscopy users.
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
