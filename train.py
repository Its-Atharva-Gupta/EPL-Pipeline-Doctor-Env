"""GRPO training entry point for ETL Pipeline Doctor."""

import argparse
import csv
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import os
import requests
from dotenv import load_dotenv
from client import ETLPipelineDoctorEnv
from models import ETLAction, ProviderConfig

# GPU Memory optimization (critical for TRL + colocate mode on 80GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)
# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

TRAIN_CONFIG = {
    "model_id": "Qwen/Qwen3-0.6B",
    "algorithm": "GRPO",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "num_generations": 4,
    "max_steps": 200,
    "per_device_train_batch_size": 1,  # Keep 1, use gradient_accumulation for effective batch
    "gradient_accumulation_steps": 8,  # Effective batch = 8 (Kube best practice)
    "learning_rate": 5e-6,
    "max_prompt_length": 1024,
    "max_completion_length": 256,
    "save_steps": 50,
    "env_base_url": "http://localhost:8000",
    "max_turns": 20,  # Max env.step() calls per episode
}

SYSTEM_PROMPT = """You are a data engineer diagnosing a broken ETL pipeline.

Output ONE command per turn. No explanations, no markdown, no prefixes.

Available commands:
- SELECT ... FROM ...              — read-only SQL queries
- INSPECT TABLE <table>            — get schema + null counts
- CHECK <table> / CHECK ROWS <table> — row counts by date
- TRACE <table> / TRACE LINEAGE <table> — upstream/downstream dependencies
- SAMPLE <table> [n]               — sample n random rows
- UPDATE ... WHERE ...             — apply a fix (mutation)
- INSERT INTO ... SELECT ...       — insert data (mutation)
- VERIFY                           — compare KPI to expected value

IMPORTANT: If a command fails, try a different approach. Do NOT repeat the same command more than once."""


def format_observation(obs) -> str:
    """Format observation into agent-readable text."""
    alert = getattr(obs, "alert", "") or ""
    last_output = getattr(obs, "last_tool_output", "") or "(none)"
    step = getattr(obs, "step", 0)
    max_steps = getattr(obs, "max_steps", 20)

    return f"""{alert}

Last command output:
{last_output}

Step {step}/{max_steps}. Diagnose and fix this issue."""


def format_history(history: list[dict]) -> str:
    """Format action history into condensed summary."""
    if not history:
        return ""
    lines = ["PREVIOUS COMMANDS AND RESULTS:"]
    for entry in history:
        cmd = entry.get("command", "")
        output = entry.get("output", "")
        reward = entry.get("reward", 0.0)
        if len(output) > 200:
            output = output[:200] + "... (truncated)"
        lines.append(f"$ {cmd}")
        lines.append(f"  Output: {output}")
        lines.append(f"  Reward: {reward:.2f}")
    return "\n".join(lines)


def apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback for older tokenizers."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def parse_commands(text: str) -> list[str]:
    """Extract commands from agent response (max 2 per turn to prevent spam)."""
    commands = []
    seen = set()
    for line in text.strip().split("\n"):
        line = line.strip()
        # Accept SQL, trace, inspect, check, verify, sample commands
        if line.upper().startswith(("SELECT ", "UPDATE ", "INSERT ", "VERIFY", "TRACE ", "INSPECT ", "CHECK ", "SAMPLE ")):
            if line not in seen:
                commands.append(line)
                seen.add(line)
        if len(commands) >= 2:
            break
    return commands


def _build_prompt(obs_dict: dict) -> str:
    """Convert an ETLObservation dict into a chat-formatted prompt."""
    alert = obs_dict.get("alert", "")
    tables = obs_dict.get("available_tables", [])
    kpis = obs_dict.get("available_kpis", [])
    history = obs_dict.get("action_history", [])
    last_output = obs_dict.get("last_tool_output") or "(none)"

    history_text = "\n".join(history[-5:]) if history else "(none)"
    commands_doc = """Available commands (raw SQL):
- SELECT ... FROM ...              — read-only SQL queries
- INSPECT TABLE <table>            — get schema + null counts
- CHECK <table> / CHECK ROWS <table> — row counts by date
- TRACE <table> / TRACE LINEAGE <table> — upstream/downstream dependencies
- SAMPLE <table> [n]               — sample n random rows
- UPDATE ... WHERE ...             — apply a fix (mutation)
- INSERT INTO ... SELECT ...       — insert data (mutation)
- VERIFY                           — compare KPI to expected value

Examples:
  SELECT * FROM gold_kpi_daily_revenue LIMIT 5
  INSPECT TABLE silver.orders_enriched
  CHECK ROWS silver.daily_sales
  TRACE LINEAGE gold.kpi_daily_revenue
  SAMPLE bronze.orders_raw 10
  UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL
  VERIFY"""

    return (
        f"You are a data engineer diagnosing a broken ETL pipeline.\n\n"
        f"ALERT: {alert}\n\n"
        f"Available tables: {tables}\n"
        f"KPIs to monitor: {kpis}\n\n"
        f"{commands_doc}\n\n"
        f"Recent action history:\n{history_text}\n\n"
        f"Last tool output:\n{last_output}\n\n"
        "OUTPUT ONLY ONE COMMAND. NO JSON. NO EXPLANATION. JUST THE COMMAND.\n"
        "Examples: SELECT * FROM gold_kpi_daily_revenue\n"
        "         TRACE LINEAGE gold.kpi_daily_revenue\n"
        "         UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL"
    )


def _start_server_background() -> threading.Thread:
    """Start the env server in a background thread (for smoke tests)."""
    import uvicorn

    from server.app import app

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    import time

    time.sleep(2)
    return t


def configure_llm_provider(base_url: str = "http://localhost:8000") -> None:
    """Configure the LLM provider (OpenAI by default)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set. Judge will not be available. "
            "Set OPENAI_API_KEY environment variable to enable LLM judge."
        )
        return

    config = ProviderConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )

    try:
        response = requests.post(
            f"{base_url}/configure",
            json=config.model_dump(),
            timeout=15,
        )
        response.raise_for_status()
        logger.info("✓ LLM provider configured: Groq (llama-3.3-70b-versatile)")
    except requests.RequestException as exc:
        logger.warning("Failed to configure LLM provider: %s", exc)
        logger.warning("Training will continue but judgment rewards will not be available.")


def run_dry_run() -> None:
    """Smoke test: start server, run one episode without GPU."""
    logger.info("Starting dry-run smoke test...")
    _start_server_background()

    base_url = TRAIN_CONFIG["env_base_url"]
    logger.info("Configuring LLM provider...")
    configure_llm_provider(base_url)

    env = ETLPipelineDoctorEnv(base_url=base_url)

    try:
        obs = env.reset()
        logger.info("Reset OK. Alert: %s", obs.alert[:60] if obs.alert else "(no alert)")

        action = ETLAction(
            command="TRACE LINEAGE gold.kpi_daily_revenue",
        )
        obs = env.step(action)
        logger.info("Step OK. Reward: %.2f", obs.step_reward)
        logger.info("Dry-run passed.")
    finally:
        env.close()


def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Compute rewards for completions by executing on environment.

    This is called by GRPOTrainer for each batch of completions.
    We execute on the environment and return per-completion rewards.
    """
    rewards = []
    env = kwargs.get("env")

    # Reset environment for fresh episode before evaluating completions
    try:
        env.reset()
    except Exception as exc:
        logger.warning(f"Failed to reset environment for batch: {exc}")
        return [-0.3] * len(completions)

    for i, completion in enumerate(completions):
        try:
            # Extract command from completion
            command = completion.strip().split('\n')[0].strip()

            # Filter out artifacts
            if not command or command.lower() in ('yes', 'no', 'ok', 'done', 'unknown'):
                logger.debug(f"  Completion {i}: skipped empty/artifact")
                rewards.append(-0.3)
                continue

            # Execute on environment
            action = ETLAction(command=command)
            obs = env.step(action)
            reward = float(obs.step_reward or 0.0)

            cmd_short = command[:60] + "..." if len(command) > 60 else command
            logger.debug(f"  Completion {i}: '{cmd_short}' → reward={reward:.3f}")
            rewards.append(reward)

        except Exception as exc:
            logger.warning(f"Reward error on completion {i}: {exc}")
            rewards.append(-0.3)

    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info(f"Batch rewards: count={len(rewards)} avg={avg_r:.3f} min={min(rewards):.3f} max={max(rewards):.3f}")
    return rewards


def plot_rewards(csv_path: Path, out_path: Path = None):
    """Plot reward curves from CSV log (no tensorboard needed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available — skipping plot")
        return

    episodes, rewards = [], []
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row:
                    episodes.append(int(row[0]))
                    rewards.append(float(row[1]))
    except (FileNotFoundError, ValueError):
        logger.warning("No valid reward data to plot")
        return

    if not episodes:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Per-episode rewards
    ax.plot(episodes, rewards, alpha=0.5, color="steelblue", marker="o", markersize=4, label="Per episode")

    # Rolling average
    if len(rewards) > 1:
        window = min(10, len(rewards) // 3 or 1)
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_steps = episodes[window - 1:]
        ax.plot(smooth_steps, smoothed, color="darkblue", linewidth=2.5, label=f"Rolling avg (window={window})")

    # Trend line
    z = np.polyfit(episodes, rewards, 1)
    trend = np.poly1d(z)
    ax.plot(episodes, trend(episodes), color="red", linewidth=1.5, linestyle="--",
            label=f"Trend ({'↑' if z[0] > 0 else '↓'} {abs(z[0]):.4f}/ep)")

    mean_r = sum(rewards) / len(rewards)
    max_r = max(rewards)
    min_r = min(rewards)

    ax.text(
        0.02, 0.98,
        f"Episodes: {len(episodes)} | Mean: {mean_r:.3f} | Max: {max_r:.3f} | Min: {min_r:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("ETL Pipeline Doctor — GRPO Reward Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = out_path or csv_path.with_suffix(".png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


def run_training(args: argparse.Namespace) -> None:
    """Full GRPO training run (requires GPU)."""
    global torch  # Make torch available to rollout_once
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        logger.error("Training dependencies missing. Install with: uv sync --extra train")
        raise SystemExit(1) from exc

    output_dir = Path("training/grpo-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id or TRAIN_CONFIG["model_id"]
    logger.info("Loading model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).cuda()

    lora_config = LoraConfig(
        r=TRAIN_CONFIG["lora_r"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        lora_dropout=TRAIN_CONFIG["lora_dropout"],
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Start env server
    _start_server_background()
    base_url = TRAIN_CONFIG["env_base_url"]

    # Configure LLM provider
    logger.info("Configuring LLM provider...")
    configure_llm_provider(base_url)

    # Create persistent client
    env = ETLPipelineDoctorEnv(base_url=base_url)

    # ---- Simple reusable dataset (no pre-computed resets) ----
    # Dataset size should be larger than max_steps to ensure samples don't run out
    dataset_prompt = "Diagnose and fix this broken ETL pipeline."
    dataset_size = max(args.max_steps * 2, 100)  # At least 2x max_steps or 100 samples
    train_ds = Dataset.from_dict({"prompt": [dataset_prompt] * dataset_size})

    # ---- CSV-based reward logging ----
    # Note: GRPOTrainer doesn't provide per-episode callbacks, so per-episode logging
    # would require custom trainer modifications. Rewards are tracked in TRL's built-in logs.

    # ---- Wrap reward function with environment access ----
    def reward_func_with_env(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Reward function that has access to the environment."""
        kwargs['env'] = env
        return reward_fn(completions, prompts, **kwargs)

    # ---- GRPO Config with optimizations ----
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        max_completion_length=TRAIN_CONFIG["max_completion_length"],
        temperature=0.2,
        save_steps=TRAIN_CONFIG["save_steps"],
        logging_steps=10,
        report_to="none",
        # GPU Memory Optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func_with_env,
        train_dataset=train_ds,
        processing_class=tokenizer,
        # Note: peft_config is not passed here since we already applied LoRA above
    )

    logger.info("Starting GRPO training for %d steps...", args.max_steps)
    try:
        trainer.train()
    finally:
        env.close()

    model.save_pretrained(str(output_dir / f"checkpoint-{args.max_steps}"))
    tokenizer.save_pretrained(str(output_dir / f"checkpoint-{args.max_steps}"))
    logger.info("Training complete. Checkpoint saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL Pipeline Doctor — GRPO Training")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test without GPU")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--num-generations", type=int, default=TRAIN_CONFIG["num_generations"])
    parser.add_argument("--max-steps", type=int, default=TRAIN_CONFIG["max_steps"])
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run_training(args)


if __name__ == "__main__":
    main()
