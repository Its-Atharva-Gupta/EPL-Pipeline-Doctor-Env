"""Plot reward curves from TRL trainer_state.json files."""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_from_trainer_state(checkpoint_dir: str, output: str = "reward_curve.png") -> None:
    """Plot rewards from TRL checkpoint trainer_state.json."""
    checkpoint_path = Path(checkpoint_dir) / "trainer_state.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {checkpoint_path}")

    with open(checkpoint_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        logger.error("No log_history found in trainer_state.json")
        return

    steps = []
    rewards = []
    for entry in log_history:
        if "step" in entry and "reward" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])

    if not steps:
        logger.error("No reward data found in log_history")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot individual steps
        ax.plot(steps, rewards, alpha=0.5, color="steelblue", marker="o", markersize=4, label="Step reward")

        # Smoothed moving average
        if len(rewards) > 1:
            window = min(10, len(rewards) // 3 or 1)
            kernel = np.ones(window) / window
            smoothed = np.convolve(rewards, kernel, mode="valid")
            smooth_steps = steps[window - 1:]
            ax.plot(smooth_steps, smoothed, color="darkblue", linewidth=2.5, label=f"Moving Avg (window={window})")

        # Reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-0.3, color="red", linestyle=":", linewidth=0.8, alpha=0.7, label="Penalty reward (-0.3)")

        # Statistics
        mean_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)

        ax.text(
            0.02, 0.98,
            f"Mean: {mean_reward:.3f}\nMax: {max_reward:.3f}\nMin: {min_reward:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Reward")
        ax.set_title("ETL Pipeline Doctor — GRPO Reward Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output, dpi=150, bbox_inches="tight")
        logger.info("✓ Reward curve saved to %s", output)
        logger.info(f"  Steps: {len(steps)} | Mean: {mean_reward:.4f} | Final: {rewards[-1]:.4f}")

    except ImportError:
        logger.warning("matplotlib/numpy not installed — printing stats instead")
        mean_r = sum(rewards) / len(rewards)
        logger.info(
            "Steps: %d | Mean: %.4f | Final: %.4f | Min: %.4f | Max: %.4f",
            len(rewards), mean_r, rewards[-1], min(rewards), max(rewards)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO reward curves from trainer_state.json")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory (containing trainer_state.json)")
    parser.add_argument("--output", default="reward_curve.png", help="Output PNG file")
    args = parser.parse_args()
    plot_from_trainer_state(args.checkpoint_dir, args.output)


if __name__ == "__main__":
    main()
