from typing import Union, Optional

import numpy as np
from scipy import stats


def _get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


def _get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return len(normalized_cumsum) - 1


class Calibration:
    """Calibrate uncertainty estimates by fitting a linear mapping from
    pixel-wise predicted std to actual prediction error (RMSE).

    Args:
        num_bins: Number of bins to use when computing calibration stats.
    """

    def __init__(self, num_bins: int = 15):
        self._bins = num_bins

    def compute_bin_boundaries(self, predict_std: np.ndarray) -> np.ndarray:
        min_std = np.min(predict_std)
        max_std = np.max(predict_std)
        return np.linspace(min_std, max_std, self._bins + 1)

    def compute_stats(
        self, pred: np.ndarray, pred_std: np.ndarray, target: np.ndarray
    ) -> dict[int, dict[str, Union[np.ndarray, list]]]:
        """Compute bin-wise RMSE and RMV per channel.

        Parameters
        ----------
        pred : np.ndarray, shape (n, h, w, c)
        pred_std : np.ndarray, shape (n, h, w, c)
        target : np.ndarray, shape (n, h, w, c)
        """
        self.stats_dict = {}
        for ch_idx in range(pred.shape[-1]):
            self.stats_dict[ch_idx] = {
                "bin_count": [],
                "rmv": [],
                "rmse": [],
                "bin_boundaries": None,
                "bin_matrix": [],
                "rmse_err": [],
            }
            pred_ch = pred[..., ch_idx]
            std_ch = pred_std[..., ch_idx]
            target_ch = target[..., ch_idx]
            boundaries = self.compute_bin_boundaries(std_ch)
            self.stats_dict[ch_idx]["bin_boundaries"] = boundaries
            bin_matrix = np.digitize(std_ch.reshape(-1), boundaries).reshape(
                std_ch.shape
            )
            self.stats_dict[ch_idx]["bin_matrix"] = bin_matrix
            error = (pred_ch - target_ch) ** 2
            for bin_idx in range(1, 1 + self._bins):
                bin_mask = bin_matrix == bin_idx
                bin_size = np.sum(bin_mask)
                if bin_size > 0:
                    bin_error = np.sqrt(np.sum(error[bin_mask]) / bin_size)
                    stderr = np.std(error[bin_mask]) / np.sqrt(bin_size)
                    rmse_stderr = np.sqrt(stderr)
                    bin_var = np.mean(std_ch[bin_mask] ** 2)
                else:
                    bin_error = rmse_stderr = None
                    bin_var = 0.0
                self.stats_dict[ch_idx]["rmse"].append(bin_error)
                self.stats_dict[ch_idx]["rmse_err"].append(rmse_stderr)
                self.stats_dict[ch_idx]["rmv"].append(np.sqrt(bin_var))
                self.stats_dict[ch_idx]["bin_count"].append(bin_size)
        return self.stats_dict

    def get_calibrated_factor_for_stdev(
        self,
        pred: Optional[np.ndarray] = None,
        pred_std: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        q_s: float = 0.00001,
        q_e: float = 0.99999,
    ) -> tuple[dict, dict]:
        """Fit a linear scalar mapping std → RMSE via linear regression on bins.

        Returns (per-channel dict of slope/intercept, array-form factors dict).
        """
        if not hasattr(self, "stats_dict"):
            if any(v is None for v in [pred, pred_std, target]):
                raise ValueError("pred, pred_std, and target must be provided.")
            self.compute_stats(pred=pred, pred_std=pred_std, target=target)

        outputs = {}
        for ch_idx, ch_stats in self.stats_dict.items():
            count = np.asarray(ch_stats["bin_count"])
            mask = count > 0
            x = np.asarray(ch_stats["rmv"])[mask]
            y = np.asarray(ch_stats["rmse"])[mask]
            count = count[mask]
            first_idx = _get_first_index(count, q_s)
            last_idx = _get_last_index(count, q_e)
            x = x[first_idx:-last_idx]
            y = y[first_idx:-last_idx]
            slope, intercept, *_ = stats.linregress(x.tolist(), y.tolist())
            outputs[ch_idx] = {"scalar": slope, "offset": intercept}

        factors = self._factors_as_arrays(outputs)
        return outputs, factors

    def _factors_as_arrays(self, factors_dict: dict) -> dict:
        scalars = np.array(
            [factors_dict[i]["scalar"] for i in range(len(factors_dict))]
        ).reshape(1, 1, 1, -1)
        offsets = np.array(
            [factors_dict[i].get("offset", 0.0) for i in range(len(factors_dict))]
        ).reshape(1, 1, 1, -1)
        if np.any(scalars < 0):
            scalars = np.full_like(scalars, 1e-9)
        return {"scalar": scalars, "offset": offsets}


def plot_calibration(
    ax, method, calibration_stats, show_identity=False, scaling_factor=1.0, offset=0.0
):
    """Plot binned RMSE vs RMV calibration curve for a single method."""
    color_map = {
        "HazeMatching": "#1f77b4",
        "SIFM": "#2ca02c",
        "LVAE": "#d62728",
    }
    color = color_map.get(method, "#1f77b4")
    scaling_factor = round(scaling_factor, 4)
    offset = round(offset, 4)

    first_idx = _get_first_index(calibration_stats[0]["bin_count"], 0.0001)
    last_idx = _get_last_index(calibration_stats[0]["bin_count"], 0.9999)
    rmv = np.array(calibration_stats[0]["rmv"][first_idx:-last_idx])
    rmse = np.array(calibration_stats[0]["rmse"][first_idx:-last_idx])
    rmse_err = np.array(calibration_stats[0]["rmse_err"][first_idx:-last_idx])

    ax.plot(rmv, rmse, "--o", label=method, color=color)
    ax.fill_between(rmv, rmse - rmse_err, rmse + rmse_err, color=color, alpha=0.3)

    if show_identity:
        ax.plot([rmv.min(), rmv.max()], [rmv.min(), rmv.max()], "k--", label=r"$y=x$")

    ax.set_xlabel("RMV", fontsize=16, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=13)
    ax.tick_params(axis="both", labelsize=14)
    ax.figure.tight_layout()
