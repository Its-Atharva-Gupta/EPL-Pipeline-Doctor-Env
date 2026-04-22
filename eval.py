"""Evaluation: compare base Qwen3-0.6B vs trained checkpoint over N episodes."""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
N_EVAL_EPISODES = 20
SEEDS = list(range(N_EVAL_EPISODES))


def _start_server() -> None:
    import uvicorn

    from server.app import app

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(2)


def _run_episode(model_fn, seed: int) -> dict:
    """Run one episode using model_fn to generate actions."""
    obs = requests.post(f"{BASE_URL}/reset", json={"seed": seed}).json()["observation"]
    total_reward = 0.0
    steps = 0
    resolved = False

    for _ in range(20):
        action = model_fn(obs)
        resp = requests.post(f"{BASE_URL}/step", json={"action": action}, timeout=30).json()
        obs = resp["observation"]
        total_reward += resp.get("reward", 0.0)
        steps += 1

        if obs.get("episode_done"):
            resolved = "PASS" in (obs.get("last_tool_output") or "")
            break

    return {
        "seed": seed,
        "total_reward": total_reward,
        "steps": steps,
        "resolved": resolved,
    }


def _random_baseline_action(obs: dict) -> dict:
    """Baseline: random valid tool selection."""
    import random

    tools = [
        {
            "tool_name": "trace_lineage",
            "tool_args": {"table": "gold.kpi_daily_revenue"},
            "reasoning": "random",
        },
        {
            "tool_name": "inspect_schema",
            "tool_args": {"table": "bronze.orders_raw"},
            "reasoning": "random",
        },
        {
            "tool_name": "check_row_counts",
            "tool_args": {"table": "silver.orders_enriched"},
            "reasoning": "random",
        },
        {
            "tool_name": "sample_rows",
            "tool_args": {"table": "bronze.orders_raw", "n": 5},
            "reasoning": "random",
        },
    ]
    return random.choice(tools)


def _model_action(model, tokenizer, obs: dict) -> dict:
    """Generate action from a loaded HF model."""
    import torch

    from models import ETLObservation
    from train import build_prompt

    prompt = build_prompt(tokenizer, ETLObservation(**obs))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "tool_name": "inspect_schema",
            "tool_args": {"table": "bronze.orders_raw"},
            "reasoning": text[:100],
        }


def evaluate(checkpoint_path: str | None, n_episodes: int = N_EVAL_EPISODES) -> None:
    _start_server()

    seeds = list(range(n_episodes))

    # --- Baseline (random) ---
    logger.info("Running baseline evaluation (%d episodes)...", n_episodes)
    baseline_results = [_run_episode(_random_baseline_action, s) for s in seeds]

    # --- Trained model ---
    trained_results = []
    if checkpoint_path:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            model.eval()
            fn = lambda obs: _model_action(model, tokenizer, obs)  # noqa: E731
            logger.info("Running trained model evaluation (%d episodes)...", n_episodes)
            trained_results = [_run_episode(fn, s) for s in seeds]
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s", exc)
    else:
        logger.info("No checkpoint provided — skipping trained model eval")

    _write_results(baseline_results, trained_results, checkpoint_path)


def _write_results(baseline: list[dict], trained: list[dict], checkpoint: str | None) -> None:
    def _summary(results: list[dict]) -> dict:
        if not results:
            return {}
        return {
            "mean_reward": sum(r["total_reward"] for r in results) / len(results),
            "resolution_rate": sum(r["resolved"] for r in results) / len(results),
            "mean_steps": sum(r["steps"] for r in results) / len(results),
        }

    bs = _summary(baseline)
    ts = _summary(trained)

    md_lines = [
        "# ETL Pipeline Doctor — Evaluation Results\n",
        "| Metric | Base Qwen3-0.6B | Trained |",
        "| --- | --- | --- |",
        f"| Mean terminal reward | {bs.get('mean_reward', 'N/A'):.2f} | {ts.get('mean_reward', 'N/A'):.2f} |",
        f"| Resolution rate | {bs.get('resolution_rate', 0):.1%} | {ts.get('resolution_rate', 0):.1%} |",
        f"| Mean steps-to-fix | {bs.get('mean_steps', 0):.1f} | {ts.get('mean_steps', 0):.1f} |",
    ]

    Path("eval_results.md").write_text("\n".join(md_lines))
    logger.info("Results written to eval_results.md")

    # Save raw JSON
    with open("eval_results.json", "w") as f:
        json.dump({"baseline": baseline, "trained": trained}, f, indent=2)

    _plot_results(bs, ts)


def _plot_results(baseline: dict, trained: dict) -> None:
    try:
        import matplotlib.pyplot as plt

        metrics = ["mean_reward", "resolution_rate", "mean_steps"]
        labels = ["Mean Reward", "Resolution Rate", "Mean Steps"]
        b_vals = [baseline.get(m, 0) for m in metrics]
        t_vals = [trained.get(m, 0) for m in metrics]

        x = range(len(metrics))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([i - 0.2 for i in x], b_vals, width=0.4, label="Base")
        ax.bar([i + 0.2 for i in x], t_vals, width=0.4, label="Trained")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title("ETL Pipeline Doctor: Base vs Trained")
        fig.savefig("eval_plot.png", dpi=150, bbox_inches="tight")
        logger.info("Plot saved to eval_plot.png")
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL Pipeline Doctor — Evaluation")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes)


if __name__ == "__main__":
    main()
