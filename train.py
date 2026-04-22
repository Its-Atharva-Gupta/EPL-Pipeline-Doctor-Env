"""GRPO training entry point for ETL Pipeline Doctor.

This trains the *agent* (HF model) to output a single valid command per step.
The environment server uses a separate provider-configured LLM for judging
reasoning quality and (optionally) adversarial curriculum design.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

from client import ETLPipelineDoctorEnv
from models import ETLAction, ETLObservation, ProviderConfig

load_dotenv()

# GPU Memory / allocator tuning (helpful for TRL + LoRA on large models).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

# LLM judge/provider rate-limit protection (server-side provider calls).
os.environ.setdefault("LLM_MIN_INTERVAL_S", "0.25")  # 250ms minimum spacing between calls
os.environ.setdefault("LLM_MAX_CONCURRENCY", "2")
os.environ.setdefault("LLM_MAX_RETRIES", "6")
os.environ.setdefault("LLM_BACKOFF_BASE_S", "0.5")
os.environ.setdefault("LLM_BACKOFF_JITTER_MAX_S", "0.25")
os.environ.setdefault("LLM_MAX_BACKOFF_S", "30")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("train")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class TrainDefaults:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_generations: int = 4
    max_steps: int = 200
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6

    max_prompt_length: int = 1024
    max_completion_length: int = 256

    save_steps: int = 50
    env_base_url: str = "http://localhost:8000"

    # Environment episode max steps (server-side constant is also 20).
    max_turns: int = 20


DEFAULTS = TrainDefaults()


SYSTEM_INSTRUCTIONS = """You are a data engineer diagnosing a broken ETL pipeline.

Output EXACTLY ONE command per turn. No explanations, no markdown, no prefixes.

Valid commands:
- SELECT ... FROM ...                 (read-only SQL query)
- INSPECT TABLE <table>               (schema + null counts)
- CHECK <table> / CHECK ROWS <table>  (row counts by date)
- TRACE <table> / TRACE LINEAGE <table> (dependencies)
- SAMPLE <table> [n]                  (sample rows)
- UPDATE ... WHERE ...                (mutation)
- INSERT INTO ... SELECT ...           (mutation)
- VERIFY                              (compare KPI to expected value)

If a command fails, try a different approach. Do NOT repeat the same command more than once.
"""


# ---------------------------------------------------------------------------
# Prompting + parsing
# ---------------------------------------------------------------------------


def _format_obs_for_user(obs: ETLObservation) -> str:
    history = obs.action_history[-5:] if obs.action_history else []
    history_text = "\n".join(history) if history else "(none)"
    last_output = obs.last_tool_output or "(none)"

    return (
        f"EPISODE_ID: {obs.episode_id}\n"
        f"ALERT: {obs.alert}\n\n"
        f"Available tables: {obs.available_tables}\n"
        f"KPIs to monitor: {obs.available_kpis}\n\n"
        f"Recent action history:\n{history_text}\n\n"
        f"Last tool output:\n{last_output}\n\n"
        "OUTPUT ONLY ONE COMMAND. NO JSON. NO EXPLANATION. JUST THE COMMAND.\n"
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply model-specific chat template with a safe fallback."""
    if hasattr(tokenizer, "apply_chat_template"):
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

    # Fallback for non-chat tokenizers.
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def build_prompt(tokenizer: Any, obs: ETLObservation) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": _format_obs_for_user(obs)},
    ]
    return _apply_chat_template(tokenizer, messages)


def extract_first_command(text: str) -> str | None:
    """Extract the first valid command line from a model completion."""
    if not text:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith(
            (
                "SELECT ",
                "UPDATE ",
                "INSERT ",
                "VERIFY",
                "TRACE ",
                "INSPECT ",
                "CHECK ",
                "SAMPLE ",
            )
        ):
            return line
    return None


# ---------------------------------------------------------------------------
# Server management + provider configuration
# ---------------------------------------------------------------------------


def _is_server_healthy(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_server_running(base_url: str) -> None:
    """Ensure the FastAPI/OpenEnv server is running (starts uvicorn if needed)."""
    if _is_server_healthy(base_url):
        return

    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    port = parsed.port or 8000

    logger.info("Starting env server in a background thread...")
    import uvicorn

    from server.app import app

    def _run() -> None:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()

    deadline = time.time() + 25.0
    while time.time() < deadline:
        if _is_server_healthy(base_url):
            logger.info("✓ Env server is healthy at %s", base_url)
            return
        time.sleep(0.25)

    raise RuntimeError(f"Env server failed health check at {base_url}")


def configure_judge_provider(
    base_url: str,
    *,
    provider: str,
    model: str,
    api_key_env: str,
    provider_base_url: str = "",
) -> None:
    """Configure the server-side judge/adversary provider via POST /configure."""
    api_key = os.getenv(api_key_env, "")
    if not api_key and provider not in ("ollama",):
        logger.warning(
            "%s not set; judge/designer LLM may fail (provider=%s).", api_key_env, provider
        )

    config = ProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        api_key=api_key,
        base_url=provider_base_url,
    )

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/configure",
            json=config.model_dump(),
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("✓ Judge provider configured: %s (%s)", provider, model)
    except requests.RequestException as exc:
        logger.warning("Failed to configure judge provider: %s", exc)
        logger.warning("Training may fail if the server cannot call its judge/designer LLM.")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptMeta:
    seed: int
    fault_type: str
    tier: str


def _cycle(items: list[str]) -> Iterable[str]:
    idx = 0
    while True:
        yield items[idx % len(items)]
        idx += 1


def build_prompt_dataset(
    env: ETLPipelineDoctorEnv,
    tokenizer: Any,
    *,
    n_samples: int,
    seed0: int,
    fault_types: list[str],
    tier: str,
) -> tuple[list[str], dict[str, PromptMeta]]:
    """Build a dataset of prompts by resetting the env to deterministic scenarios.

    We bypass curriculum here by forcing `fault_type` and `tier` in reset kwargs so:
      - prompts are actually conditioned on env observations,
      - reward evaluation can reset to the same scenario for fair comparison,
      - scoring does not skew curriculum via repeated record_outcome calls.
    """
    prompts: list[str] = []
    meta: dict[str, PromptMeta] = {}

    ft_iter = _cycle(fault_types)

    for i in range(n_samples):
        seed = seed0 + i
        ft = next(ft_iter)
        obs = env.reset(seed=seed, fault_type=ft, tier=tier)
        prompt = build_prompt(tokenizer, obs)
        prompts.append(prompt)
        meta[prompt] = PromptMeta(seed=seed, fault_type=ft, tier=tier)

    return prompts, meta


# ---------------------------------------------------------------------------
# Reward function (GRPO)
# ---------------------------------------------------------------------------


def make_reward_fn(
    env: ETLPipelineDoctorEnv,
    prompt_meta: dict[str, PromptMeta],
    *,
    penalty_invalid: float = -0.5,
) -> Any:
    """Create a reward function compatible with TRL GRPOTrainer."""

    lock = threading.Lock()

    def _reward(completions: list[str], prompts: list[str], **_: Any) -> list[float]:
        rewards: list[float] = []

        # Guard against TRL calling reward in parallel threads.
        with lock:
            for i, (completion, prompt) in enumerate(zip(completions, prompts, strict=False)):
                meta = prompt_meta.get(prompt)
                if meta is None:
                    # Unknown prompt: treat as invalid and keep going.
                    logger.warning("Missing prompt metadata for completion %d; assigning penalty.", i)
                    rewards.append(penalty_invalid)
                    continue

                # Reset to the exact same deterministic scenario for each completion so
                # rewards are comparable within a GRPO generation batch.
                try:
                    env.reset(seed=meta.seed, fault_type=meta.fault_type, tier=meta.tier)
                except Exception as exc:
                    logger.warning("Env reset failed for completion %d: %s", i, exc)
                    rewards.append(penalty_invalid)
                    continue

                command = extract_first_command(completion)
                if not command:
                    rewards.append(penalty_invalid)
                    continue

                try:
                    obs = env.step(ETLAction(command=command))
                    step_reward = float(getattr(obs, "step_reward", 0.0) or 0.0)
                    terminal_reward = float(getattr(obs, "terminal_reward", 0.0) or 0.0)
                    r = step_reward + terminal_reward
                    rewards.append(r)
                except Exception as exc:
                    logger.warning("Env step failed for completion %d: %s", i, exc)
                    rewards.append(penalty_invalid)

        return rewards

    return _reward


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace) -> None:
    base_url = args.env_base_url
    ensure_server_running(base_url)

    if args.configure_judge:
        configure_judge_provider(
            base_url,
            provider=args.judge_provider,
            model=args.judge_model,
            api_key_env=args.judge_api_key_env,
            provider_base_url=args.judge_base_url,
        )

    env = ETLPipelineDoctorEnv(base_url=base_url)
    try:
        obs = env.reset(seed=args.seed, fault_type=args.fault_types[0], tier=args.tier)
        logger.info("Dry-run reset OK. episode_id=%s alert=%s", obs.episode_id, (obs.alert or "")[:80])

        obs2 = env.step(ETLAction(command="TRACE LINEAGE gold.kpi_daily_revenue"))
        logger.info("Dry-run step OK. step_reward=%.4f terminal_reward=%.4f done=%s",
                    obs2.step_reward, obs2.terminal_reward, obs2.episode_done)
    finally:
        env.close()


def run_training(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        logger.error("Training dependencies missing. Install with: uv sync --extra train")
        raise SystemExit(1) from exc

    base_url = args.env_base_url
    ensure_server_running(base_url)

    if args.configure_judge:
        configure_judge_provider(
            base_url,
            provider=args.judge_provider,
            model=args.judge_model,
            api_key_env=args.judge_api_key_env,
            provider_base_url=args.judge_base_url,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id
    logger.info("Loading model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        raise RuntimeError("CUDA is required for training in this script.")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Persistent env connection for prompt generation + reward evaluation.
    env = ETLPipelineDoctorEnv(base_url=base_url)
    try:
        # Build prompt dataset with real env observations.
        n_samples = max(args.max_steps * 2, 200)
        prompts, meta = build_prompt_dataset(
            env,
            tokenizer,
            n_samples=n_samples,
            seed0=args.seed,
            fault_types=args.fault_types,
            tier=args.tier,
        )
        train_ds = Dataset.from_dict({"prompt": prompts})

        reward_fn = make_reward_fn(
            env,
            meta,
            penalty_invalid=args.penalty_invalid,
        )

        grpo_cfg = GRPOConfig(
            output_dir=str(output_dir),
            num_generations=args.num_generations,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=max(2, int(0.02 * args.max_steps)),
            max_grad_norm=1.0,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            save_steps=args.save_steps,
            logging_steps=10,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_cfg,
            reward_funcs=reward_fn,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )

        logger.info("Starting GRPO training for %d steps...", args.max_steps)
        trainer.train()

        # Save final checkpoint.
        final_dir = output_dir / f"checkpoint-{args.max_steps}"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("Training complete. Saved to %s", final_dir)
    finally:
        env.close()


def _parse_csv_list(value: str) -> list[str]:
    items = [v.strip() for v in (value or "").split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty list")
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="ETL Pipeline Doctor — GRPO Training")

    p.add_argument("--dry-run", action="store_true", help="Smoke test server + env interaction")

    p.add_argument("--env-base-url", default=DEFAULTS.env_base_url)
    p.add_argument("--output-dir", default="training/grpo-output")

    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended on Ampere+)")

    p.add_argument("--lora-r", type=int, default=DEFAULTS.lora_r)
    p.add_argument("--lora-alpha", type=int, default=DEFAULTS.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=DEFAULTS.lora_dropout)

    p.add_argument("--num-generations", type=int, default=DEFAULTS.num_generations)
    p.add_argument("--max-steps", type=int, default=DEFAULTS.max_steps)
    p.add_argument("--per-device-train-batch-size", type=int, default=DEFAULTS.per_device_train_batch_size)
    p.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULTS.gradient_accumulation_steps)
    p.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-prompt-length", type=int, default=DEFAULTS.max_prompt_length)
    p.add_argument("--max-completion-length", type=int, default=DEFAULTS.max_completion_length)
    p.add_argument("--save-steps", type=int, default=DEFAULTS.save_steps)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fault-types",
        type=_parse_csv_list,
        default=_parse_csv_list("schema_drift,stale_partition,null_explosion,fanout_join,type_mismatch"),
        help="Comma-separated fault types to train on (bypasses curriculum).",
    )
    p.add_argument("--tier", default="warmup", help="Tier to use with forced fault types.")

    p.add_argument("--penalty-invalid", type=float, default=-0.5, help="Reward for invalid/unparseable outputs")

    # Judge provider config (server-side LLM)
    p.add_argument("--configure-judge", action="store_true", help="POST /configure before training")
    p.add_argument("--judge-provider", default="groq", choices=["anthropic", "openai", "groq", "openrouter", "ollama"])
    p.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    p.add_argument("--judge-api-key-env", default="GROQ_API_KEY")
    p.add_argument("--judge-base-url", default="", help="Optional base_url for openai-compatible providers")

    args = p.parse_args()

    if args.dry_run:
        run_dry_run(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
