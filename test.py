"""Reward signal sanity check — run BEFORE committing to GRPO training.

This script probes three independent questions that GRPO absolutely requires answers to:

  1. DOES THE JUDGE DISCRIMINATE?
     Feed the judge hand-crafted good / neutral / bad actions. Check that scores
     separate cleanly. If a textbook diagnostic action and a random action both
     score near 0, the judge is giving you no signal and GRPO will flatline.

  2. DO TEXTBOOK FIXES ACTUALLY PASS verify_output?
     Inject each fault, apply the canonical fix, check that verify_output reports
     success. If the "correct" fix fails verification, your ground-truth tolerance
     is too tight or the fix logic is wrong — either way, the agent can never win.

  3. IS END-TO-END LATENCY SURVIVABLE?
     Time one full episode. GRPO training = (rollouts × steps × latency × gradient steps).
     If one episode takes >90 seconds, 200 training steps × 4 rollouts is untenable
     on Colab and you need to degrade (fewer rollouts, smaller model, cloud judge).

Run:
    uv run python sanity_check_reward.py

Exit code 0 = safe to train. Non-zero = fix the flagged issue first.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

from server.etl_pipeline_doctor_environment import EtlPipelineDoctorEnvironment
from server.llm_judge import LLMJudge
from models import ETLAction, ToolName


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"{GREEN}PASS{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}WARN{RESET}  {msg}")


def _header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")
    print("-" * len(msg))


# -----------------------------------------------------------------------------
# CHECK 1 — Judge discrimination
# -----------------------------------------------------------------------------

@dataclass
class JudgeProbe:
    """A synthetic action with a category label we expect the judge to reward."""
    label: str                  # "good" | "neutral" | "bad"
    alert: str
    compact_history: list[str]
    action: ETLAction
    tool_output: str


def build_judge_probes() -> list[JudgeProbe]:
    """12 probes: 4 good, 4 neutral, 4 bad. Judge scores should cluster by label."""
    alert = "KPI gold.kpi_daily_revenue is 0 for the last 3 days"

    good = [
        JudgeProbe(
            "good", alert, [],
            ETLAction(
                tool_name=ToolName.TRACE_LINEAGE,
                tool_args={"table": "gold.kpi_daily_revenue"},
                reasoning="KPI is zero. Before inspecting data, I need to map which upstream tables feed this KPI so I can systematically narrow down where the fault sits.",
            ),
            "Upstream: ['silver.daily_sales']. Downstream: []",
        ),
        JudgeProbe(
            "good", alert,
            ["[1] trace_lineage(gold.kpi_daily_revenue) -> Upstream: silver.daily_sales"],
            ETLAction(
                tool_name=ToolName.CHECK_ROW_COUNTS,
                tool_args={"table": "silver.daily_sales"},
                reasoning="Lineage points to silver.daily_sales. A zero KPI could mean rows are missing for recent dates. Row counts by partition will confirm staleness vs content issue.",
            ),
            "Row count: 870. Last partition: 2026-04-18 (3 days ago). Expected most recent: 2026-04-21.",
        ),
        JudgeProbe(
            "good", alert,
            ["[1] trace_lineage(silver.daily_sales)", "[2] check_row_counts -> stale, last partition 3 days old"],
            ETLAction(
                tool_name=ToolName.INSPECT_SCHEMA,
                tool_args={"table": "silver.orders_enriched"},
                reasoning="Silver is stale. Checking schema of its upstream source to see if a schema change broke the refresh job rather than assuming it's a pure partition issue.",
            ),
            "Columns: order_id, customer_id, order_date, order_total (was total_amount), product_count, region",
        ),
        JudgeProbe(
            "good", alert,
            ["[1-3] lineage + row counts + schema inspect -> schema drift total_amount -> order_total"],
            ETLAction(
                tool_name=ToolName.APPLY_FIX,
                tool_args={"fix_type": "rename_column", "target": "silver.orders_enriched", "params": {"old": "total_amount", "new": "order_total"}},
                reasoning="Confirmed: column was renamed upstream. Applying rename to downstream transform so aggregation references the new column name.",
            ),
            "Fix applied. silver.orders_enriched now uses order_total.",
        ),
    ]

    neutral = [
        JudgeProbe(
            "neutral", alert, [],
            ETLAction(
                tool_name=ToolName.INSPECT_SCHEMA,
                tool_args={"table": "gold.kpi_daily_revenue"},
                reasoning="Looking at the schema of the KPI table.",
            ),
            "Columns: date, revenue, yoy_growth_pct",
        ),
        JudgeProbe(
            "neutral", alert, [],
            ETLAction(
                tool_name=ToolName.SAMPLE_ROWS,
                tool_args={"table": "bronze.orders_raw", "n": 5},
                reasoning="Sampling rows from bronze.",
            ),
            "5 rows sampled.",
        ),
        JudgeProbe(
            "neutral", alert,
            ["[1] inspect_schema(gold.kpi_daily_revenue)"],
            ETLAction(
                tool_name=ToolName.RUN_QUERY,
                tool_args={"sql": "SELECT COUNT(*) FROM bronze.orders_raw"},
                reasoning="Counting rows in bronze.",
            ),
            "count: 6200",
        ),
        JudgeProbe(
            "neutral", alert, [],
            ETLAction(
                tool_name=ToolName.RUN_QUERY,
                tool_args={"sql": "SELECT * FROM bronze.products_raw LIMIT 3"},
                reasoning="Looking at products.",
            ),
            "3 rows.",
        ),
    ]

    bad = [
        JudgeProbe(
            "bad", alert, [],
            ETLAction(
                tool_name=ToolName.APPLY_FIX,
                tool_args={"fix_type": "custom_sql", "target": "bronze.orders_raw", "params": {"sql": "DELETE FROM bronze.orders_raw"}},
                reasoning="Let me just clear the table.",
            ),
            "Fix applied.",
        ),
        JudgeProbe(
            "bad", alert,
            ["[1] inspect_schema(gold.kpi_daily_revenue)", "[2] inspect_schema(gold.kpi_daily_revenue)"],
            ETLAction(
                tool_name=ToolName.INSPECT_SCHEMA,
                tool_args={"table": "gold.kpi_daily_revenue"},
                reasoning="Inspecting the schema again.",
            ),
            "Columns: date, revenue, yoy_growth_pct (same as before)",
        ),
        JudgeProbe(
            "bad", alert,
            ["[1] check_row_counts(silver.daily_sales) -> stale 3 days"],
            ETLAction(
                tool_name=ToolName.APPLY_FIX,
                tool_args={"fix_type": "coalesce_column", "target": "silver.daily_sales", "params": {"column": "region", "default": "X"}},
                reasoning="Applying a coalesce fix.",
            ),
            "Fix applied.",
        ),
        JudgeProbe(
            "bad", alert, [],
            ETLAction(
                tool_name=ToolName.APPLY_FIX,
                tool_args={"fix_type": "rename_column", "target": "bronze.orders_raw", "params": {"old": "x", "new": "y"}},
                reasoning="Try a fix.",
            ),
            "Error: column x does not exist.",
        ),
    ]

    return good + neutral + bad


def check_judge_discrimination() -> tuple[bool, dict[str, list[float]]]:
    """Score each probe 3 times (stochasticity test) and return scores by label."""
    _header("Check 1 of 3 — Judge discrimination")
    probes = build_judge_probes()
    judge = LLMJudge()
    scores_by_label: dict[str, list[float]] = {"good": [], "neutral": [], "bad": []}

    for i, probe in enumerate(probes, 1):
        trials = []
        for _ in range(3):
            try:
                score = judge.score(
                    alert=probe.alert,
                    compact_history=probe.compact_history,
                    action=probe.action,
                    tool_result=type("TR", (), {"success": True, "output": probe.tool_output, "data": None})(),
                )
                trials.append(float(score))
            except Exception as e:
                _fail(f"Judge raised on probe {i} ({probe.label}): {e}")
                trials.append(0.0)

        mean = statistics.mean(trials)
        stdev = statistics.stdev(trials) if len(trials) > 1 else 0.0
        scores_by_label[probe.label].append(mean)
        print(f"  probe {i:>2} [{probe.label:>7}] mean={mean:+.2f} stdev={stdev:.2f} trials={trials}")

    good_mean = statistics.mean(scores_by_label["good"])
    neutral_mean = statistics.mean(scores_by_label["neutral"])
    bad_mean = statistics.mean(scores_by_label["bad"])

    print(f"\n  good mean:    {good_mean:+.2f}")
    print(f"  neutral mean: {neutral_mean:+.2f}")
    print(f"  bad mean:     {bad_mean:+.2f}")

    passed = True

    if good_mean - bad_mean < 0.4:
        _fail(f"Good and bad actions should be separated by >=0.4. Got {good_mean - bad_mean:.2f}.")
        _fail("Judge is not discriminating enough — GRPO will not get a usable gradient.")
        passed = False
    else:
        _ok(f"Good-vs-bad separation is {good_mean - bad_mean:.2f} — usable.")

    if good_mean < 0.3:
        _warn(f"Good actions average only {good_mean:.2f}. Judge may be too harsh; check prompt.")

    if bad_mean > -0.2:
        _warn(f"Bad actions average only {bad_mean:.2f}. Judge may be too lenient; check prompt.")

    return passed, scores_by_label


# -----------------------------------------------------------------------------
# CHECK 2 — Textbook fixes pass verify_output
# -----------------------------------------------------------------------------

TEXTBOOK_FIXES = {
    "schema_drift": {
        "fix_type": "rename_column",
        "params_hint": "use the old_column / new_column from the fault spec",
    },
    "stale_partition": {
        "fix_type": "backfill_partition",
        "params_hint": "use the most recent date missing from the stale table",
    },
    "null_explosion": {
        "fix_type": "coalesce_column",
        "params_hint": "coalesce the affected column with a non-null default",
    },
    "fanout_join": {
        "fix_type": "deduplicate",
        "params_hint": "deduplicate on the natural primary key",
    },
    "type_mismatch": {
        "fix_type": "cast_column",
        "params_hint": "cast back to DECIMAL or the original numeric type",
    },
}


def check_textbook_fixes() -> bool:
    _header("Check 2 of 3 — Textbook fixes resolve each fault")
    env = EtlPipelineDoctorEnvironment()
    all_pass = True

    for fault_type in TEXTBOOK_FIXES:
        try:
            obs = env.reset_with_fault(fault_type=fault_type)
        except Exception as e:
            _fail(f"Could not force fault={fault_type}: {e}")
            _fail("Add a test-only `reset_with_fault(fault_type=...)` method to the env for deterministic probing.")
            all_pass = False
            continue

        state = env.get_state()
        correct_fix = env.get_canonical_fix_for_current_fault()

        fix_action = ETLAction(
            tool_name=ToolName.APPLY_FIX,
            tool_args=correct_fix,
            reasoning=f"Applying the canonical fix for {fault_type}.",
        )

        step_result = env.step(fix_action)
        resolved = step_result.episode_done and state.resolved

        if resolved:
            _ok(f"{fault_type}: canonical fix resolved the fault.")
        else:
            _fail(f"{fault_type}: canonical fix did NOT resolve. Check tolerance or fix logic.")
            print(f"    state.resolved={state.resolved}, episode_done={step_result.episode_done}")
            all_pass = False

    return all_pass


# -----------------------------------------------------------------------------
# CHECK 3 — End-to-end latency
# -----------------------------------------------------------------------------

def check_episode_latency() -> bool:
    _header("Check 3 of 3 — End-to-end episode latency")
    env = EtlPipelineDoctorEnvironment()

    scripted_actions = [
        ETLAction(
            tool_name=ToolName.TRACE_LINEAGE,
            tool_args={"table": "gold.kpi_daily_revenue"},
            reasoning="Start with lineage.",
        ),
        ETLAction(
            tool_name=ToolName.CHECK_ROW_COUNTS,
            tool_args={"table": "silver.daily_sales"},
            reasoning="Check counts on upstream.",
        ),
        ETLAction(
            tool_name=ToolName.INSPECT_SCHEMA,
            tool_args={"table": "silver.orders_enriched"},
            reasoning="Inspect schema.",
        ),
        ETLAction(
            tool_name=ToolName.SAMPLE_ROWS,
            tool_args={"table": "bronze.orders_raw", "n": 5},
            reasoning="Sample bronze.",
        ),
        ETLAction(
            tool_name=ToolName.RUN_QUERY,
            tool_args={"sql": "SELECT COUNT(*) FROM bronze.orders_raw"},
            reasoning="Count.",
        ),
    ]

    start = time.monotonic()
    env.reset()
    reset_time = time.monotonic() - start

    step_times = []
    for action in scripted_actions:
        t0 = time.monotonic()
        env.step(action)
        step_times.append(time.monotonic() - t0)

    total = reset_time + sum(step_times)
    mean_step = statistics.mean(step_times)

    print(f"\n  reset: {reset_time:.2f}s")
    print(f"  step times: {[f'{t:.2f}s' for t in step_times]}")
    print(f"  mean per step: {mean_step:.2f}s")
    print(f"  total 5-step episode: {total:.2f}s")

    projected_20_step_episode = reset_time + mean_step * 20
    projected_200_training_steps = projected_20_step_episode * 4 * 200 / 3600  # 4 rollouts, convert to hours

    print(f"\n  projected 20-step episode: {projected_20_step_episode:.1f}s")
    print(f"  projected full 200-step training run (4 rollouts): {projected_200_training_steps:.1f} hours")

    if mean_step > 5.0:
        _fail(f"Mean step latency {mean_step:.1f}s is too slow. Training will take >{projected_200_training_steps:.0f}h.")
        _fail("Options: drop rollouts to 2, cap episodes at 10 steps, or switch judge to Claude Sonnet.")
        return False
    elif mean_step > 2.0:
        _warn(f"Mean step latency {mean_step:.1f}s is borderline. Training will take ~{projected_200_training_steps:.0f}h.")
        return True
    else:
        _ok(f"Mean step latency {mean_step:.1f}s — training run should finish in ~{projected_200_training_steps:.1f}h.")
        return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print(f"{BOLD}ETL Pipeline Doctor — reward signal sanity check{RESET}")
    print("Run this BEFORE committing to a full GRPO training run.\n")

    results: dict[str, bool] = {}

    try:
        passed, scores = check_judge_discrimination()
        results["judge_discrimination"] = passed
        with open("judge_scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\n  score distribution saved to judge_scores.json")
    except Exception as e:
        _fail(f"Check 1 crashed: {e}")
        results["judge_discrimination"] = False

    try:
        results["textbook_fixes"] = check_textbook_fixes()
    except Exception as e:
        _fail(f"Check 2 crashed: {e}")
        results["textbook_fixes"] = False

    try:
        results["latency"] = check_episode_latency()
    except Exception as e:
        _fail(f"Check 3 crashed: {e}")
        results["latency"] = False

    _header("Summary")
    for check, passed in results.items():
        label = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {label}  {check}")

    if all(results.values()):
        print(f"\n{GREEN}{BOLD}All checks passed. Safe to run GRPO training.{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}One or more checks failed. Fix these before training.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())