"""Visualise reward curves from a GRPO training metrics.jsonl file."""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot(metrics_path: str, output: str = "reward_curve.png") -> None:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    steps, rewards = [], []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "step" in entry and "reward" in entry:
                    steps.append(entry["step"])
                    rewards.append(entry["reward"])
            except json.JSONDecodeError:
                continue

    if not steps:
        logger.error("No reward data found in %s", metrics_path)
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rewards, alpha=0.4, color="steelblue", label="Step reward")

        # Smoothed moving average
        window = min(20, len(rewards) // 4 or 1)
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_steps = steps[window - 1 :]
        ax.plot(smooth_steps, smoothed, color="darkblue", linewidth=2, label=f"MA-{window}")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.set_title("ETL Pipeline Doctor — GRPO Reward Curve")
        ax.legend()
        fig.savefig(output, dpi=150, bbox_inches="tight")
        logger.info("Reward curve saved to %s", output)
    except ImportError:
        logger.warning("matplotlib/numpy not installed — printing stats instead")
        mean_r = sum(rewards) / len(rewards)
        logger.info(
            "Steps: %d  Mean reward: %.3f  Final reward: %.3f", len(rewards), mean_r, rewards[-1]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO reward curves")
    parser.add_argument("metrics_file", help="Path to metrics.jsonl")
    parser.add_argument("--output", default="reward_curve.png")
    args = parser.parse_args()
    plot(args.metrics_file, args.output)


if __name__ == "__main__":
    main()
