import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ETLAction, ETLObservation, ETLState, ToolResult

from .adversarial_designer import AdversarialDesigner
from .constants import KPI_TABLES, MAX_STEPS, TIERS, WAREHOUSE_TABLES
from .curriculum import CurriculumController
from .fault_catalogue import FaultCatalogue
from .fault_injector import FaultInjector
from .llm_judge import LLMJudge
from .reward import DenseProgress, compute_step_reward, compute_terminal_reward
from .tool_handlers import ToolHandlers
from .warehouse import Warehouse

logger = logging.getLogger(__name__)

try:
    from .provider_state import get_provider_config
except ImportError:
    from server.provider_state import get_provider_config

_KPI_ALERTS: dict[str, list[str]] = {
    "gold.kpi_daily_revenue": [
        "ALERT: Daily revenue KPI dropped unexpectedly. Expected ~$40,000/day, seeing anomalous values.",
        "ALERT: Revenue KPI shows data quality issues. Recent day revenue looks incorrect.",
    ],
    "gold.kpi_category_mix": [
        "ALERT: Category revenue mix KPI is showing unexpected distribution. Share values look wrong.",
        "ALERT: Product category mix anomaly detected. Revenue shares do not sum to 1.0.",
    ],
}


class EtlPipelineDoctorEnvironment(Environment):
    """RL environment for diagnosing and repairing broken ETL pipelines."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, provider_config: Any = None) -> None:
        super().__init__()
        self._warehouse = Warehouse()
        self._fault_injector = FaultInjector(self._warehouse)
        self._tool_handlers = ToolHandlers(self._warehouse)
        self._judge = LLMJudge()
        self._catalogue = FaultCatalogue()
        self._designer = AdversarialDesigner()
        self._rng = random.Random()
        self._curriculum = CurriculumController(self._catalogue, self._designer, seed=self._rng.randint(0, 2**31))
        # If the server has a configured provider, pick it up automatically.
        self._provider_config: Any = provider_config or get_provider_config()  # ProviderConfig | None
        self._debug_judge_raw: bool = os.environ.get("ETL_DEBUG_JUDGE_RAW", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._debug_judge_prompts: bool = os.environ.get("ETL_DEBUG_JUDGE_PROMPTS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        # Per-episode state
        self._etl_state = ETLState()
        self._action_history: list[str] = []
        self._prev_kpi_ok: bool = False
        self._called_trace_lineage: bool = False
        self._tool_call_counts: dict[str, int] = {}
        self._current_alert: str = ""
        self._cumulative_reward: float = 0.0

        # Dense progress tracking — reset each episode
        self._prev_kpi_proximity: float = 0.0
        self._prev_fault_progress: float = 0.0
        self._baseline_null_count: int = 0
        self._baseline_corrupt_count: int = 0
        self._baseline_total_count: int = 0

        # Per-session episode registry for reproducibility. Keyed by episode_id.
        # Allows clients to reset() with the same episode_id to reproduce the
        # exact same scenario (seed + fault spec + alert).
        self._episode_registry: dict[str, dict[str, Any]] = {}

    def set_provider(self, config: Any) -> None:
        """Set the LLM provider configuration used by the judge and designer."""
        self._provider_config = config

    def _infer_touched_tables(self, sql: str) -> list[str]:
        """Best-effort table detection for UPDATE/INSERT statements."""
        s = sql.upper()
        candidates = [
            "silver.orders_enriched",
            "silver.daily_sales",
            "bronze.products_raw",
        ]
        touched: list[str] = []
        for q in candidates:
            if q.upper() in s:
                touched.append(q)
                continue
            sql_name = self._warehouse.table_sql_name(q).upper()
            if sql_name in s:
                touched.append(q)
        return touched

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> ETLObservation:
        # Allow configuring the judge once per WebSocket session by passing a
        # ProviderConfig (or dict) in reset kwargs. Reset does NOT close the WS
        # connection, so the config persists across episodes.
        if "provider_config" in kwargs and kwargs["provider_config"] is not None:
            pc = kwargs["provider_config"]
            try:
                # Accept either ProviderConfig or a dict matching ProviderConfig schema.
                if hasattr(pc, "provider") and hasattr(pc, "model"):
                    self.set_provider(pc)
                elif isinstance(pc, dict):
                    from models import ProviderConfig

                    self.set_provider(ProviderConfig(**pc))
            except Exception:
                logger.warning("Invalid provider_config in reset kwargs; ignoring")

        # If server-level config changed after this env instance was created,
        # refresh our local view at reset boundaries.
        if self._provider_config is None:
            self._provider_config = get_provider_config()

        ep_id = episode_id or str(uuid4())

        # Reproducibility: if episode_id is re-used within the same session,
        # restore the same seed + fault + alert regardless of passed seed/kwargs.
        if ep_id in self._episode_registry:
            rec = self._episode_registry[ep_id]
            ep_seed = int(rec["seed"])
            fault_spec = rec["fault_spec"]
            self._current_alert = str(rec["alert"])
        else:
            ep_seed = seed if seed is not None else self._rng.randint(0, 2**31)

            # Optional: allow forcing a specific fault type/tier for deterministic eval.
            forced_fault_type = kwargs.get("fault_type")
            forced_tier = kwargs.get("tier")
            if forced_fault_type is not None:
                tier = str(forced_tier) if forced_tier is not None else "warmup"
                if tier not in TIERS:
                    tier = "warmup"
                fault_spec = self._catalogue.pick(str(forced_fault_type), tier, random.Random(ep_seed))
            else:
                # Pick fault from curriculum (may use adversarial designer + judge provider).
                fault_spec = self._curriculum.pick_fault(self._provider_config)

        # Set up fresh warehouse
        self._warehouse.setup(seed=ep_seed)

        # Inject fault
        self._fault_injector.inject([fault_spec])

        # Reset per-episode bookkeeping
        self._tool_handlers.reset()
        self._action_history = []
        self._called_trace_lineage = False
        self._tool_call_counts = {}
        self._cumulative_reward = 0.0
        self._prev_kpi_ok = False
        self._prev_kpi_proximity = 0.0
        self._prev_fault_progress = 0.0
        self._baseline_null_count = 0
        self._baseline_corrupt_count = 0
        self._baseline_total_count = 0

        # Propagate the fault into downstream KPI tables.
        self._warehouse.recompute_downstream_from(fault_spec["target_table"])

        # Capture dense signal baselines (must be after fault injection + downstream recompute)
        self._capture_dense_baselines(fault_spec["fault_type"])

        tier_name = str(fault_spec.get("tier", "warmup"))
        if tier_name not in TIERS:
            tier_name = self._curriculum.state_summary()["tiers"].get(
                fault_spec["fault_type"], "warmup"
            )
        diff_idx = TIERS.index(tier_name) if tier_name in TIERS else 0

        self._etl_state = ETLState(
            episode_id=ep_id,
            step=0,
            max_steps=MAX_STEPS,
            cumulative_reward=0.0,
            difficulty=diff_idx,
            fault_type=fault_spec["fault_type"],
            resolved=False,
        )

        # Pick alert text
        kpi = fault_spec["affected_kpi"]
        if ep_id not in self._episode_registry:
            alert_options = _KPI_ALERTS.get(
                kpi,
                [f"ALERT: KPI anomaly detected in {kpi}."],
            )
            # Make alert selection deterministic per episode seed for stable judge prompts.
            self._current_alert = random.Random(ep_seed).choice(alert_options)

            self._episode_registry[ep_id] = {
                "seed": ep_seed,
                "fault_spec": fault_spec,
                "alert": self._current_alert,
            }

        logger.info(
            "Episode %s started: fault=%s kpi=%s seed=%d",
            ep_id,
            fault_spec["fault_type"],
            kpi,
            ep_seed,
        )

        return ETLObservation(
            alert=self._current_alert,
            last_tool_output=None,
            action_history=[],
            available_kpis=list(KPI_TABLES),
            available_tables=list(WAREHOUSE_TABLES),
            step=0,
            step_reward=0.0,
            step_reward_breakdown={},
            episode_done=False,
            difficulty=diff_idx,
            judge_provider=getattr(self._provider_config, "provider", None) if self._provider_config else None,
            judge_model=getattr(self._provider_config, "model", None) if self._provider_config else None,
            done=False,
            reward=0.0,
            episode_id=ep_id,
            cumulative_reward=0.0,
            terminal_reward=0.0,
            episode_return=0.0,
        )

    # ------------------------------------------------------------------
    # Dense progress helpers
    # ------------------------------------------------------------------
    def _capture_dense_baselines(self, fault_type: str) -> None:
        """Capture fault-specific baseline metrics immediately after fault injection."""
        conn = self._warehouse.conn
        try:
            if fault_type == "null_explosion":
                row = conn.execute(
                    "SELECT COUNT(*) FROM silver_orders_enriched WHERE region IS NULL"
                ).fetchone()
                self._baseline_null_count = int(row[0]) if row and row[0] is not None else 0
            elif fault_type == "type_mismatch":
                corrupt = conn.execute(
                    "SELECT COUNT(*) FROM bronze_orders_raw WHERE total_amount LIKE '%_corrupted'"
                ).fetchone()
                total = conn.execute(
                    "SELECT COUNT(*) FROM bronze_orders_raw"
                ).fetchone()
                self._baseline_corrupt_count = int(corrupt[0]) if corrupt and corrupt[0] is not None else 0
                self._baseline_total_count = int(total[0]) if total and total[0] is not None else 1
        except Exception:
            logger.warning("Failed to capture dense baselines for fault_type=%s", fault_type)

        # Initialize prev state to post-fault baseline so step 1 delta measures
        # the agent's marginal contribution, not the gap from zero to broken state
        self._prev_kpi_proximity = self._compute_kpi_proximity(conn)
        self._prev_fault_progress = self._compute_fault_progress(conn, fault_type)

    def _compute_kpi_proximity(self, conn: sqlite3.Connection) -> float:
        """
        Compute how close the current gold.kpi_daily_revenue revenue sum is
        to the pre-fault ground truth.

        Returns 0.0 when fully broken, 1.0 when matching ground truth exactly.
        """
        try:
            row = conn.execute(
                "SELECT SUM(revenue) FROM gold_kpi_daily_revenue"
            ).fetchone()
            cur_sum = float(row[0]) if row and row[0] is not None else 0.0

            gt_rows = self._warehouse.ground_truth("gold.kpi_daily_revenue")
            gt_sum = sum(float(r.get("revenue", 0.0)) for r in gt_rows)

            if gt_sum == 0.0:
                return 1.0 if cur_sum == 0.0 else 0.0

            proximity = 1.0 - min(abs(cur_sum - gt_sum) / gt_sum, 1.0)
            return float(proximity)
        except Exception:
            logger.warning("KPI proximity query failed — defaulting to 0.0")
            return 0.0

    def _compute_fault_progress(self, conn: sqlite3.Connection, fault_type: str) -> float:
        """
        Compute a fault-type-specific progress metric.

        For each fault type, returns 0.0 at fault injection state and 1.0 at resolution.
        Returns 0.0 on any query error (fail-safe, non-fatal).
        """
        try:
            match fault_type:

                case "null_explosion":
                    row = conn.execute(
                        "SELECT COUNT(*) FROM silver_orders_enriched WHERE region IS NULL"
                    ).fetchone()
                    current_nulls = int(row[0]) if row and row[0] is not None else 0
                    if self._baseline_null_count == 0:
                        return 1.0
                    return 1.0 - min(current_nulls / self._baseline_null_count, 1.0)

                case "stale_partition":
                    silver_max = conn.execute(
                        "SELECT MAX(order_date) FROM silver_orders_enriched"
                    ).fetchone()
                    bronze_max = conn.execute(
                        "SELECT MAX(order_date) FROM bronze_orders_raw"
                    ).fetchone()
                    s = silver_max[0] if silver_max and silver_max[0] else None
                    b = bronze_max[0] if bronze_max and bronze_max[0] else None
                    return 1.0 if (s is not None and s == b) else 0.0

                case "type_mismatch":
                    row = conn.execute(
                        "SELECT COUNT(*) FROM bronze_orders_raw "
                        "WHERE total_amount LIKE '%_corrupted'"
                    ).fetchone()
                    current_corrupt = int(row[0]) if row and row[0] is not None else 0
                    if self._baseline_corrupt_count == 0:
                        return 1.0
                    return 1.0 - min(current_corrupt / self._baseline_corrupt_count, 1.0)

                case "fanout_join":
                    row = conn.execute(
                        "SELECT COUNT(*) - COUNT(DISTINCT product_id) "
                        "FROM bronze_products_raw"
                    ).fetchone()
                    duplicate_count = int(row[0]) if row and row[0] is not None else 0
                    return 1.0 if duplicate_count == 0 else 0.0

                case "schema_drift":
                    cols = conn.execute(
                        "PRAGMA table_info(bronze_orders_raw)"
                    ).fetchall()
                    col_names = {c[1] for c in cols}
                    return 1.0 if "total_amount" in col_names else 0.0

                case _:
                    return 0.0

        except Exception:
            logger.warning(
                "Fault progress query failed for fault_type=%s — defaulting to 0.0",
                fault_type
            )
            return 0.0

    def _compute_dense_progress(self) -> DenseProgress:
        """
        Compute per-step dense progress signals by querying the live warehouse.

        Returns a DenseProgress with:
          - kpi_proximity: fraction of ground-truth KPI revenue currently matched [0.0, 1.0]
          - fault_progress: fault-type-specific data-quality metric [0.0, 1.0]
          - kpi_delta: change in kpi_proximity since last step
          - fault_delta: change in fault_progress since last step
        """
        conn = self._warehouse.conn
        fault_type = self._etl_state.fault_type

        kpi_proximity = self._compute_kpi_proximity(conn)
        fault_progress = self._compute_fault_progress(conn, fault_type)

        kpi_delta = kpi_proximity - self._prev_kpi_proximity
        fault_delta = fault_progress - self._prev_fault_progress

        self._prev_kpi_proximity = kpi_proximity
        self._prev_fault_progress = fault_progress

        return DenseProgress(
            kpi_proximity=kpi_proximity,
            fault_progress=fault_progress,
            kpi_delta=kpi_delta,
            fault_delta=fault_delta,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: ETLAction, timeout_s: float | None = None, **kwargs) -> ETLObservation:  # type: ignore[override]
        # Ensure we pick up any server-level judge configuration.
        if self._provider_config is None:
            self._provider_config = get_provider_config()

        self._etl_state.step += 1
        step = self._etl_state.step

        # --- Execute command ---
        affected_kpi = (
            self._fault_injector.active_faults[0]["affected_kpi"]
            if self._fault_injector.active_faults
            else "gold.kpi_daily_revenue"
        )
        tool_result = self._tool_handlers.dispatch_command_with_defaults(
            action.command, default_kpi=affected_kpi
        )

        # --- Detect command type ---
        cmd_upper = action.command.strip().upper()
        is_mutation = cmd_upper.startswith(("UPDATE", "INSERT"))
        is_trace = cmd_upper.startswith("TRACE")
        is_verify = cmd_upper.startswith("VERIFY")

        # If the agent mutates a table, propagate changes into downstream KPIs.
        if is_mutation and tool_result.success:
            touched = self._infer_touched_tables(action.command)
            for t in touched:
                self._warehouse.recompute_downstream_from(t)

        # --- Track state for reward computation ---
        command_key = action.command
        repeat_count = self._tool_call_counts.get(command_key, 0)
        repeated = repeat_count > 0
        self._tool_call_counts[command_key] = repeat_count + 1

        if is_trace:
            self._called_trace_lineage = True

        called_fix_without_lineage = (
            is_mutation and not self._called_trace_lineage
        )
        malformed = not tool_result.success and not is_verify

        # Check resolution
        resolved_this_step = False
        if is_mutation and tool_result.success:
            verify = self._tool_handlers.verify_output(
                affected_kpi
            )
            if verify.success:
                resolved_this_step = True
                self._etl_state.resolved = True

        broke_something = (
            is_mutation and not tool_result.success and self._prev_kpi_ok
        )

        wrong_fix = is_mutation and not tool_result.success

        step_progressed = (
            tool_result.success
            and not repeated
            and not is_verify
        )

        # --- Judge score ---
        judge_debug: dict[str, Any] | None = None
        if self._debug_judge_raw or self._debug_judge_prompts:
            judge_score, judge_debug = self._judge.score_and_debug(
                alert=self._current_alert,
                compact_history=self._action_history[-5:],
                action=action,
                tool_result=tool_result,
                config=self._provider_config,
                include_prompts=self._debug_judge_prompts,
                include_raw=self._debug_judge_raw,
            )
        else:
            judge_score = self._judge.score(
                alert=self._current_alert,
                compact_history=self._action_history[-5:],
                action=action,
                tool_result=tool_result,
                config=self._provider_config,
            )

        # --- Compute dense progress (requires live DB, must be after tool execution) ---
        dense_progress = self._compute_dense_progress()

        # --- Compute reward ---
        breakdown = compute_step_reward(
            action=action,
            tool_result=tool_result,
            judge_score=judge_score,
            resolved_this_step=resolved_this_step,
            broke_something=broke_something,
            repeated_tool_call=repeated,
            wrong_fix_type=wrong_fix,
            called_apply_fix_without_lineage=called_fix_without_lineage,
            malformed_args=malformed,
            step_progressed=step_progressed,
            action_repeat_count=repeat_count,
            dense_progress=dense_progress,
        )

        self._cumulative_reward += breakdown.total

        # --- Check episode end ---
        episode_done = resolved_this_step or step >= MAX_STEPS
        terminal_reward = 0.0
        if episode_done:
            terminal = compute_terminal_reward(self._etl_state.resolved, step)
            breakdown.terminal = terminal
            self._cumulative_reward += terminal
            terminal_reward = terminal
            self._etl_state.cumulative_reward = self._cumulative_reward
            self._curriculum.record_outcome(
                self._etl_state.fault_type,
                self._etl_state.resolved,
                step,
            )
            logger.info(
                "Episode %s ended: resolved=%s steps=%d cum_reward=%.2f",
                self._etl_state.episode_id,
                self._etl_state.resolved,
                step,
                self._cumulative_reward,
            )
            breakdown.details["r_terminal"] = float(terminal_reward)

        # Track KPI state for next step
        if is_verify and tool_result.success:
            self._prev_kpi_ok = True

        # Build compact history entry
        cmd_short = action.command[:60] + "..." if len(action.command) > 60 else action.command
        summary = f"[{step}] {cmd_short} → {'OK' if tool_result.success else 'FAIL'}"
        self._action_history.append(summary)

        diff_idx = self._etl_state.difficulty

        judge_provider = getattr(self._provider_config, "provider", None) if self._provider_config else None
        judge_model = getattr(self._provider_config, "model", None) if self._provider_config else None

        return ETLObservation(
            episode_id=self._etl_state.episode_id,
            alert=self._current_alert,
            last_tool_output=tool_result.output,
            action_history=list(self._action_history),
            available_kpis=list(KPI_TABLES),
            available_tables=list(WAREHOUSE_TABLES),
            step=step,
            step_reward=breakdown.total,
            step_reward_breakdown=breakdown.details,
            episode_done=episode_done,
            difficulty=diff_idx,
            cumulative_reward=float(self._cumulative_reward),
            terminal_reward=float(terminal_reward),
            episode_return=float(self._cumulative_reward) if episode_done else 0.0,
            judge_raw=(judge_debug or {}).get("raw"),
            judge_prompt=(judge_debug or {}).get("user_prompt"),
            judge_score=judge_score,
            judge_brief=(judge_debug or {}).get("brief"),
            judge_cache_hit=(judge_debug or {}).get("cache_hit"),
            judge_provider=judge_provider,
            judge_model=judge_model,
            done=episode_done,
            reward=breakdown.total,
        )

    # ------------------------------------------------------------------
    # state property / accessor
    # ------------------------------------------------------------------
    @property
    def state(self) -> ETLState:
        return self._etl_state

    def get_state(self) -> ETLState:
        return self._etl_state

    # ------------------------------------------------------------------
    # test helpers (not for production use)
    # ------------------------------------------------------------------
    def reset_with_fault(self, fault_type: str, seed: int = 42) -> ETLObservation:
        """Test-only: reset with a specific fault type, bypassing curriculum."""
        fault_spec = self._catalogue.pick(fault_type, "warmup", self._rng)

        self._warehouse.setup(seed=seed)
        self._fault_injector.inject([fault_spec])

        # Reset per-episode bookkeeping first
        self._tool_handlers.reset()
        self._action_history = []
        self._called_trace_lineage = False
        self._tool_call_counts = {}
        self._cumulative_reward = 0.0
        self._prev_kpi_ok = False
        self._prev_kpi_proximity = 0.0
        self._prev_fault_progress = 0.0
        self._baseline_null_count = 0
        self._baseline_corrupt_count = 0
        self._baseline_total_count = 0

        # Capture dense signal baselines (must be after fault injection and after resetting baselines to 0)
        self._warehouse.recompute_downstream_from(fault_spec["target_table"])
        self._capture_dense_baselines(fault_type)

        kpi = fault_spec["affected_kpi"]
        alert_options = _KPI_ALERTS.get(kpi, [f"ALERT: KPI anomaly detected in {kpi}."])
        self._current_alert = alert_options[0]

        self._etl_state = ETLState(
            episode_id=f"test-{fault_type}",
            step=0,
            max_steps=MAX_STEPS,
            cumulative_reward=0.0,
            difficulty=0,
            fault_type=fault_type,
            resolved=False,
        )

        return ETLObservation(
            alert=self._current_alert,
            last_tool_output=None,
            action_history=[],
            available_kpis=list(KPI_TABLES),
            available_tables=list(WAREHOUSE_TABLES),
            step=0,
            step_reward=0.0,
            step_reward_breakdown={},
            episode_done=False,
            difficulty=0,
            done=False,
            reward=0.0,
            episode_id=self._etl_state.episode_id,
            cumulative_reward=0.0,
            terminal_reward=0.0,
            episode_return=0.0,
        )

    def get_canonical_fix_for_current_fault(self) -> dict:
        """Test-only: return the exact tool_args for apply_fix that correctly fixes the current fault."""
        faults = self._fault_injector.active_faults
        if not faults:
            raise RuntimeError("No active fault — call reset_with_fault first")

        spec = faults[0]
        fault_type = spec["fault_type"]
        params = spec.get("params", {})
        target = spec["target_table"]

        match fault_type:
            case "schema_drift":
                return {
                    "fix_type": "rename_column",
                    "target": target,
                    "params": {
                        "old": params.get("new_column", "order_total"),
                        "new": params.get("old_column", "total_amount"),
                    },
                }
            case "stale_partition":
                max_date = self._warehouse.conn.execute(
                    "SELECT MAX(order_date) FROM bronze_orders_raw"
                ).fetchone()[0]
                return {
                    "fix_type": "backfill_partition",
                    "target": target,
                    "params": {"date": max_date, "source_table": "bronze.orders_raw"},
                }
            case "null_explosion":
                return {
                    "fix_type": "coalesce_column",
                    "target": target,
                    "params": {
                        "column": params.get("column", "region"),
                        "default": "UNKNOWN",
                    },
                }
            case "fanout_join":
                return {
                    "fix_type": "deduplicate",
                    "target": target,
                    "params": {"columns": ["product_id"]},
                }
            case "type_mismatch":
                return {
                    "fix_type": "cast_column",
                    "target": target,
                    "params": {
                        "column": params.get("column", "total_amount"),
                        "to_type": "REAL",
                    },
                }
            case _:
                raise ValueError(f"No canonical fix defined for fault type: {fault_type}")

    # ------------------------------------------------------------------
    # tool dispatch
    # ------------------------------------------------------------------
