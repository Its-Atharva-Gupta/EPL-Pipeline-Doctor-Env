"""
Detailed workflow tracer for ETL Pipeline Doctor.

Shows the complete flow:
  1. What is sent to /reset
  2. What observation is returned
  3. Curriculum state after reset
  4. What action is sent to /step
  5. Reward breakdown received
  6. Judge score details
  7. Next observation state
  8. Curriculum state updates
"""

import json
import logging
import threading
import time
from pathlib import Path

import requests

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WorkflowTracer:
    """Traces complete ETL Pipeline Doctor workflow with detailed logging."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.episode_num = 0
        self.step_num = 0

    def format_json(self, obj: dict, indent: int = 2) -> str:
        """Pretty-print JSON with indentation."""
        return json.dumps(obj, indent=indent, default=str)

    # ========================================================================
    # RESET WORKFLOW
    # ========================================================================
    def reset(self, seed: int | None = None, episode_id: str | None = None) -> dict:
        """
        Step 1: Call /reset endpoint.

        Captures:
          - Request payload
          - Curriculum state before
          - Response observation
          - Fault type selected (hidden in state)
          - Alert text chosen
        """
        self.episode_num += 1
        self.step_num = 0

        print("\n" + "=" * 80)
        print(f"EPISODE {self.episode_num}: RESET WORKFLOW")
        print("=" * 80)

        payload = {}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        print(f"\n[REQUEST] POST /reset")
        print(f"Payload:\n{self.format_json(payload)}")

        try:
            resp = requests.post(
                f"{self.base_url}/reset",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            return {}

        data = resp.json()
        obs = data.get("observation", {})

        # ────────────────────────────────────────────────────────────────
        print(f"\n[RESPONSE] Observation returned:")
        print(f"  Alert: {obs.get('alert', 'N/A')[:70]}...")
        print(f"  Available tables: {obs.get('available_tables', [])}")
        print(f"  Available KPIs: {obs.get('available_kpis', [])}")
        print(f"  Step: {obs.get('step')}")
        print(f"  Difficulty: {obs.get('difficulty')}")
        print(f"  Episode done: {obs.get('episode_done')}")

        # ────────────────────────────────────────────────────────────────
        # Get environment state (includes hidden fault_type, curriculum mastery)
        print(f"\n[SERVER STATE] Fetching internal state...")
        try:
            state_resp = requests.get(f"{self.base_url}/state", timeout=10)
            state_resp.raise_for_status()
            state = state_resp.json()

            print(f"  Episode ID: {state.get('episode_id')}")
            print(f"  Fault type (HIDDEN from agent): {state.get('fault_type')}")
            print(f"  Max steps: {state.get('max_steps')}")
            print(f"  Cumulative reward: {state.get('cumulative_reward')}")
        except Exception as e:
            print(f"  ⚠ Could not fetch state: {e}")
            state = {}

        self._print_section_divider()
        return obs

    # ========================================================================
    # STEP WORKFLOW
    # ========================================================================
    def step(
        self,
        tool_name: str,
        tool_args: dict,
        reasoning: str = "",
    ) -> dict:
        """
        Step 2: Call /step with an action.

        Captures:
          - Action sent (tool, args, reasoning)
          - Reward breakdown (outcome, reasoning, efficiency)
          - Judge score and brief
          - Tool result
          - Next observation
          - Whether episode ended
        """
        self.step_num += 1

        print("\n" + "-" * 80)
        print(f"STEP {self.step_num}: TOOL EXECUTION")
        print("-" * 80)

        # ────────────────────────────────────────────────────────────────
        # Prepare and send action
        action = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reasoning": reasoning or f"Executing {tool_name}",
        }

        print(f"\n[ACTION] Sending to /step")
        print(f"  Tool: {tool_name}")
        print(f"  Args: {self.format_json(tool_args)}")
        print(f"  Reasoning: {reasoning[:70]}..." if len(reasoning) > 70 else f"  Reasoning: {reasoning}")

        try:
            # OpenEnv expects action wrapped in "action" field
            payload = {"action": action}
            resp = requests.post(
                f"{self.base_url}/step",
                json=payload,
                timeout=30,  # Judge calls may take time
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            return {}

        data = resp.json()
        obs = data.get("observation", {})

        # ────────────────────────────────────────────────────────────────
        # Tool result
        print(f"\n[TOOL RESULT]")
        print(f"  Output: {obs.get('last_tool_output', 'N/A')[:150]}...")

        # ────────────────────────────────────────────────────────────────
        # Reward breakdown
        print(f"\n[REWARD BREAKDOWN]")
        breakdown = obs.get("step_reward_breakdown", {})
        for key, value in breakdown.items():
            print(f"  {key}: {value:.3f}")

        total_reward = obs.get("step_reward", 0.0)
        print(f"  ──────────────")
        print(f"  TOTAL: {total_reward:.3f}")

        # ────────────────────────────────────────────────────────────────
        # Episode status
        print(f"\n[EPISODE STATUS]")
        print(f"  Step: {obs.get('step')}")
        print(f"  Episode done: {obs.get('episode_done')}")
        print(f"  Action history: {len(obs.get('action_history', []))} actions so far")

        # ────────────────────────────────────────────────────────────────
        # Get detailed state
        try:
            state_resp = requests.get(f"{self.base_url}/state", timeout=10)
            state_resp.raise_for_status()
            state = state_resp.json()
            print(f"  Cumulative reward: {state.get('cumulative_reward', 0.0):.3f}")
        except Exception as e:
            pass

        self._print_section_divider()
        return obs

    # ========================================================================
    # CURRICULUM OBSERVER
    # ========================================================================
    def show_curriculum_state(self) -> None:
        """Fetch and display curriculum mastery state."""
        print(f"\n[CURRICULUM STATE]")
        try:
            state_resp = requests.get(f"{self.base_url}/state", timeout=10)
            state_resp.raise_for_status()
            state = state_resp.json()
            # Note: curriculum state is not exposed via /state, so we just show what we can
            print(f"  (Curriculum state is internal to environment)")
            print(f"  Episode count: {state.get('episode_id')}")
        except Exception as e:
            print(f"  ⚠ Could not fetch: {e}")

    # ========================================================================
    # HELPERS
    # ========================================================================
    def _print_section_divider(self) -> None:
        """Print a visual divider."""
        print("\n" + "." * 80)

    def print_summary(self) -> None:
        """Print episode summary."""
        print("\n" + "=" * 80)
        print(f"EPISODE {self.episode_num}: SUMMARY")
        print("=" * 80)
        print(f"Total steps executed: {self.step_num}")


def main():
    """Run a complete workflow trace with one episode."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "ETL PIPELINE DOCTOR — WORKFLOW TRACER".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Start server in background
    print("\n[STARTUP] Starting environment server...")
    from server.app import app

    def run_server():
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    print("[STARTUP] Server ready at http://localhost:8000")

    tracer = WorkflowTracer()

    # ════════════════════════════════════════════════════════════════════════
    # EPISODE 1: Trace a complete reset + 3 steps
    # ════════════════════════════════════════════════════════════════════════

    obs = tracer.reset(seed=42)

    if obs:
        # Step 1: Trace lineage from KPI
        obs = tracer.step(
            tool_name="trace_lineage",
            tool_args={"table": "gold.kpi_daily_revenue"},
            reasoning="Starting diagnosis: trace the KPI upstream to identify data dependencies",
        )

        # Step 2: Inspect schema of a key table
        if not obs.get("episode_done"):
            obs = tracer.step(
                tool_name="inspect_schema",
                tool_args={"table": "silver.orders_enriched"},
                reasoning="Check schema to identify data type or column naming issues",
            )

        # Step 3: Sample rows to see actual data
        if not obs.get("episode_done"):
            obs = tracer.step(
                tool_name="sample_rows",
                tool_args={"table": "silver.orders_enriched", "n": 5},
                reasoning="Sample actual data to spot NULL values or type mismatches",
            )

        # Step 4: Run a query to aggregate
        if not obs.get("episode_done"):
            obs = tracer.step(
                tool_name="run_query",
                tool_args={"sql": "SELECT COUNT(*) as cnt, COUNT(DISTINCT order_id) as uniq FROM silver.orders_enriched;"},
                reasoning="Count rows to detect if there's missing or duplicated data",
            )

        tracer.print_summary()

    # ════════════════════════════════════════════════════════════════════════
    # EPISODE 2: Show how curriculum state persists
    # ════════════════════════════════════════════════════════════════════════

    print("\n\n")
    obs = tracer.reset(seed=99)

    if obs:
        obs = tracer.step(
            tool_name="check_row_counts",
            tool_args={"table": "silver.orders_enriched"},
            reasoning="Check if row counts match expected values (stale partition detection)",
        )

        tracer.print_summary()

    print("\n\n[DONE] Workflow trace complete.\n")


if __name__ == "__main__":
    main()
