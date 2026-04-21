"""GRPO training entry point for ETL Pipeline Doctor."""

import argparse
import logging
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CONFIG = {
    "model_id": "Qwen/Qwen3-0.6B",
    "algorithm": "GRPO",
    "lora_r": 16,
    "lora_alpha": 32,
    "num_generations": 4,
    "max_steps": 200,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "max_prompt_length": 1024,
    "max_completion_length": 256,
    "save_steps": 50,
    "env_base_url": "http://localhost:8000",
}


def _build_prompt(obs_dict: dict) -> str:
    """Convert an ETLObservation dict into a chat-formatted prompt."""
    alert = obs_dict.get("alert", "")
    tables = obs_dict.get("available_tables", [])
    kpis = obs_dict.get("available_kpis", [])
    history = obs_dict.get("action_history", [])
    last_output = obs_dict.get("last_tool_output") or "(none)"

    history_text = "\n".join(history[-5:]) if history else "(none)"
    tools_doc = """Available tools:
- run_query(sql)           — read-only SQL query
- inspect_schema(table)    — column names, types, null counts
- check_row_counts(table)  — row count + date breakdown
- trace_lineage(table)     — upstream/downstream tables
- sample_rows(table, n)    — random sample rows
- apply_fix(fix_type, target, params) — apply a repair
- verify_output(kpi_name)  — compare KPI to expected value

Fix types: rename_column, backfill_partition, coalesce_column, deduplicate, cast_column, custom_sql"""

    return (
        f"You are a data engineer diagnosing a broken ETL pipeline.\n\n"
        f"ALERT: {alert}\n\n"
        f"Available tables: {tables}\n"
        f"KPIs to monitor: {kpis}\n\n"
        f"{tools_doc}\n\n"
        f"Recent action history:\n{history_text}\n\n"
        f"Last tool output:\n{last_output}\n\n"
        "Respond with a JSON action:\n"
        '{"tool_name": "<tool>", "tool_args": {<args>}, "reasoning": "<your reasoning>"}'
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


def run_dry_run() -> None:
    """Smoke test: start server, run one episode without GPU."""
    import requests

    logger.info("Starting dry-run smoke test...")
    _start_server_background()

    base = TRAIN_CONFIG["env_base_url"]

    obs = requests.post(f"{base}/reset", json={}).json()["observation"]
    logger.info("Reset OK. Alert: %s", obs["alert"][:60])

    action = {
        "action": {
            "tool_name": "trace_lineage",
            "tool_args": {"table": "gold.kpi_daily_revenue"},
            "reasoning": "Dry-run test — tracing lineage from KPI",
        }
    }
    step_obs = requests.post(f"{base}/step", json=action).json()
    logger.info("Step OK. Reward: %s", step_obs.get("reward"))
    logger.info("Dry-run passed.")


def run_training(args: argparse.Namespace) -> None:
    """Full GRPO training run (requires H100 / GPU)."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        logger.error("Training dependencies missing. Install with: uv sync --extra train")
        raise SystemExit(1) from exc

    output_dir = Path("training/grpo-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id or TRAIN_CONFIG["model_id"]
    logger.info("Loading model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=TRAIN_CONFIG["lora_r"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Start env server
    _start_server_background()
    base_url = TRAIN_CONFIG["env_base_url"]
    import requests

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Call the env server to get rewards for each completion."""
        rewards = []
        for completion in completions:
            try:
                import json as _json

                action = _json.loads(completion)
                obs_resp = requests.post(
                    f"{base_url}/step",
                    json={"action": action},
                    timeout=15,
                ).json()
                rewards.append(float(obs_resp.get("reward", 0.0)))
            except Exception as exc:
                logger.warning("Reward fn error: %s", exc)
                rewards.append(-0.3)
        return rewards

    # Build dataset from env resets
    import datasets

    def _make_dataset(n_samples: int = 200) -> datasets.Dataset:
        rows = []
        for _ in range(n_samples):
            obs = requests.post(f"{base_url}/reset", json={}).json()["observation"]
            prompt = _build_prompt(obs)
            rows.append({"prompt": prompt})
        return datasets.Dataset.from_list(rows)

    logger.info("Generating training prompts...")
    train_ds = _make_dataset(args.max_steps)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        max_prompt_length=TRAIN_CONFIG["max_prompt_length"],
        max_completion_length=TRAIN_CONFIG["max_completion_length"],
        save_steps=TRAIN_CONFIG["save_steps"],
        logging_steps=10,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training for %d steps...", args.max_steps)
    trainer.train()

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
