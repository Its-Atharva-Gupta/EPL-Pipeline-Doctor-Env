"""
Enhanced training script with full workflow tracing.

Shows for each episode:
  1. Reset payload → observation received
  2. Curriculum state (mastery EMA, tier)
  3. Fault type selected (from curriculum)
  4. For each step:
     - Action sent (tool + args + reasoning)
     - Tool result
     - Reward breakdown (outcome, reasoning, efficiency, penalties)
     - Judge score and brief
     - Cumulative reward so far
  5. Episode end: terminal reward, outcome (success/fail)
  6. Curriculum update: EMA after episode

Run this to see everything happening during training.
Usage: python train_traced.py --episodes 3 --dry-run
"""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingTracer:
    """Instruments training loop with detailed logging."""

    def __init__(self, base_url: str = "http://localhost:8000", use_websocket: bool = False):
        self.base_url = base_url.rstrip("/")
        self.use_websocket = use_websocket
        self.episode_count = 0
        self.total_episodes = 0
        self.ws = None

    

    def format_reward_breakdown(self, breakdown: dict) -> str:
        """Format reward breakdown for display."""
        lines = ["Breakdown:"]
        for key, val in breakdown.items():
            if isinstance(val, float):
                lines.append(f"      {key:30s} = {val:+.4f}")
            else:
                lines.append(f"      {key:30s} = {val}")
        return "\n".join(lines)

    def run_episode(self, max_steps: int = 20, seed: int | None = None) -> dict:
        """
        Run a single episode with full tracing.

        Returns:
          {
            "episode_id": str,
            "fault_type": str,
            "resolved": bool,
            "steps": int,
            "cumulative_reward": float,
            "rewards_per_step": [float, ...],
          }
        """
        self.episode_count += 1
        progress = f"[{self.episode_count}/{self.total_episodes}]" if self.total_episodes else f"[Episode {self.episode_count}]"

        print("\n" + "=" * 100)
        print(f"{progress} RESET")
        print("=" * 100)

        # ────────────────────────────────────────────────────────────────
        # 1. RESET
        # ────────────────────────────────────────────────────────────────
        reset_payload = {}
        if seed is not None:
            reset_payload["seed"] = seed

        logger.info(f"{progress} Calling /reset with seed={seed}")

        try:
            reset_resp = requests.post(
                f"{self.base_url}/reset",
                json=reset_payload,
                timeout=10,
            )
            reset_resp.raise_for_status()
        except Exception as e:
            logger.error(f"{progress} Reset failed: {e}")
            return {}

        obs = reset_resp.json().get("observation", {})
        episode_id = obs.get("episode_id", "unknown")

        # ────────────────────────────────────────────────────────────────
        # Get hidden state (fault type, etc.)
        try:
            state_resp = requests.get(f"{self.base_url}/state", timeout=10)
            state = state_resp.json()
            fault_type = state.get("fault_type", "unknown")
            difficulty = state.get("difficulty", 0)
        except Exception:
            fault_type = "unknown"
            difficulty = 0

        print(f"\n📍 Episode ID: {episode_id}")
        print(f"🔧 Fault type: {fault_type}")
        print(f"📊 Difficulty: {difficulty}")
        print(f"🚨 Alert: {obs.get('alert', 'N/A')[:80]}...")

        # ────────────────────────────────────────────────────────────────
        # 2. STEP LOOP
        # ────────────────────────────────────────────────────────────────
        step_num = 0
        cumulative_reward = 0.0
        rewards_per_step = []
        resolved = False

        for step_num in range(1, max_steps + 1):
            print(f"\n{'-' * 100}")
            print(f"  STEP {step_num}")
            print(f"{'-' * 100}")

            # ────────────────────────────────────────────────────────────────
            # Simulate agent action (in real training, this comes from model)
            # For tracing, we use a simple heuristic: trace → inspect → sample → fix
            # ────────────────────────────────────────────────────────────────
            action = self._choose_action(step_num, obs)

            print(f"\n  📤 Action:")
            print(f"      Tool: {action['tool_name']}")
            if action['tool_args']:
                print(f"      Args: {json.dumps(action['tool_args'], indent=12)}")
            if action['reasoning']:
                print(f"      Reasoning: {action['reasoning'][:70]}...")

            # ────────────────────────────────────────────────────────────────
            # Call /step
            # ────────────────────────────────────────────────────────────────
            try:
                # OpenEnv expects action wrapped in "action" field
                payload = {"action": action}
                step_resp = requests.post(
                    f"{self.base_url}/step",
                    json=payload,
                    timeout=30,
                )
                step_resp.raise_for_status()
            except Exception as e:
                logger.error(f"  Step {step_num} failed: {e}")
                break

            obs = step_resp.json().get("observation", {})

            # ────────────────────────────────────────────────────────────────
            # Tool result
            # ────────────────────────────────────────────────────────────────
            tool_output = obs.get("last_tool_output", "N/A")
            output_preview = tool_output[:80] + "..." if len(tool_output) > 80 else tool_output
            print(f"\n  📥 Tool result:")
            print(f"      {output_preview}")

            # ────────────────────────────────────────────────────────────────
            # Reward
            # ────────────────────────────────────────────────────────────────
            step_reward = obs.get("step_reward", 0.0)
            cumulative_reward += step_reward
            rewards_per_step.append(step_reward)

            breakdown = obs.get("step_reward_breakdown", {})
            print(f"\n  💰 Reward: {step_reward:+.4f}")
            if breakdown:
                for key, val in breakdown.items():
                    if isinstance(val, (int, float)):
                        print(f"      {key:30s} = {val:+.4f}")

            print(f"      ──────────────────────────────────")
            print(f"      Cumulative: {cumulative_reward:+.4f}")

            # ────────────────────────────────────────────────────────────────
            # Episode end check
            # ────────────────────────────────────────────────────────────────
            episode_done = obs.get("episode_done", False)
            if episode_done:
                resolved = step_reward > 2.0  # Terminal reward threshold
                print(f"\n  ✅ Episode ended at step {step_num}")
                break

        # ────────────────────────────────────────────────────────────────
        # 3. EPISODE SUMMARY
        # ────────────────────────────────────────────────────────────────
        print("\n" + "=" * 100)
        print(f"{progress} EPISODE SUMMARY")
        print("=" * 100)
        print(f"  Fault type: {fault_type}")
        print(f"  Steps taken: {step_num}")
        print(f"  Resolved: {'✓ YES' if resolved else '✗ NO'}")
        print(f"  Cumulative reward: {cumulative_reward:+.4f}")
        print(f"  Rewards per step: {[f'{r:+.2f}' for r in rewards_per_step]}")

        return {
            "episode_id": episode_id,
            "fault_type": fault_type,
            "resolved": resolved,
            "steps": step_num,
            "cumulative_reward": cumulative_reward,
            "rewards_per_step": rewards_per_step,
        }

    def _choose_action(self, step_num: int, obs: dict) -> dict:
        """
        Simple action selection heuristic for tracing.

        In real training, the agent model would choose this.
        For demonstration, we use a fixed sequence.
        """
        action_sequence = [
            {
                "tool_name": "trace_lineage",
                "tool_args": {"table": obs.get("available_kpis", ["gold.kpi_daily_revenue"])[0]},
                "reasoning": "Trace KPI upstream to understand data lineage",
            },
            {
                "tool_name": "inspect_schema",
                "tool_args": {"table": "silver.orders_enriched"},
                "reasoning": "Inspect schema to identify schema drift or type issues",
            },
            {
                "tool_name": "sample_rows",
                "tool_args": {"table": "silver.orders_enriched", "n": 5},
                "reasoning": "Sample rows to spot NULL values or data quality issues",
            },
            {
                "tool_name": "check_row_counts",
                "tool_args": {"table": "silver.orders_enriched"},
                "reasoning": "Check for stale partitions or missing data",
            },
            {
                "tool_name": "apply_fix",
                "tool_args": {
                    "fix_type": "coalesce_column",
                    "target": "silver.orders_enriched",
                    "params": {"column": "region", "default": "UNKNOWN"},
                },
                "reasoning": "Apply fix for NULL explosion (most common issue)",
            },
            {
                "tool_name": "verify_output",
                "tool_args": {"kpi_name": "gold.kpi_daily_revenue"},
                "reasoning": "Verify the KPI is now correct",
            },
        ]

        if step_num <= len(action_sequence):
            return action_sequence[step_num - 1]
        else:
            return {
                "tool_name": "verify_output",
                "tool_args": {"kpi_name": "gold.kpi_daily_revenue"},
                "reasoning": "Final verification",
            }


def main():
    parser = argparse.ArgumentParser(
        description="Training tracer for ETL Pipeline Doctor"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start server, run episodes, show traces",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Starting server for dry-run...")
        from server.app import app

        def run_server():
            import uvicorn

            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)
        logger.info("Server ready")

    print("\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 98 + "║")
    print("║" + "ETL PIPELINE DOCTOR — TRAINING TRACE".center(98) + "║")
    print("║" + f"Running {args.episodes} episodes with detailed workflow logging".center(98) + "║")
    print("║" + " " * 98 + "║")
    print("╚" + "=" * 98 + "╝")

    tracer = TrainingTracer()
    tracer.total_episodes = args.episodes

    # Run episodes
    results = []
    for ep in range(args.episodes):
        result = tracer.run_episode(max_steps=args.max_steps, seed=42 + ep)
        results.append(result)
        time.sleep(0.5)

    # ════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 98 + "║")
    print("║" + "TRAINING SUMMARY".center(98) + "║")
    print("║" + " " * 98 + "║")
    print("╚" + "=" * 98 + "╝")

    print(f"\n  Total episodes: {len(results)}")
    resolved_count = sum(1 for r in results if r.get("resolved"))
    print(f"  Resolved: {resolved_count}/{len(results)} ({100*resolved_count//len(results)}%)")

    print(f"\n  Per-fault-type performance:")
    fault_groups = {}
    for r in results:
        ft = r.get("fault_type", "unknown")
        if ft not in fault_groups:
            fault_groups[ft] = []
        fault_groups[ft].append(r)

    for ft, episodes in fault_groups.items():
        resolved_ft = sum(1 for e in episodes if e.get("resolved"))
        avg_reward = sum(e.get("cumulative_reward", 0) for e in episodes) / len(episodes)
        print(f"      {ft:20s}: {resolved_ft}/{len(episodes)} resolved, avg reward = {avg_reward:+.2f}")

    total_reward = sum(r.get("cumulative_reward", 0) for r in results)
    avg_reward = total_reward / len(results) if results else 0
    print(f"\n  Average cumulative reward: {avg_reward:+.4f}")
    print(f"  Total reward across all episodes: {total_reward:+.4f}")

    print("\n[DONE] Training trace complete.\n")


if __name__ == "__main__":
    main()
