import logging
import random
import sys
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ETLAction, ETLObservation, ETLState, ToolName, ToolResult

from .adversarial_designer import AdversarialDesigner
from .constants import KPI_TABLES, MAX_STEPS, WAREHOUSE_TABLES
from .curriculum import CurriculumController
from .fault_catalogue import FaultCatalogue
from .fault_injector import FaultInjector
from .llm_judge import LLMJudge
from .reward import compute_step_reward, compute_terminal_reward
from .tool_handlers import ToolHandlers
from .warehouse import Warehouse

logger = logging.getLogger(__name__)

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

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._warehouse = Warehouse()
        self._fault_injector = FaultInjector(self._warehouse)
        self._tool_handlers = ToolHandlers(self._warehouse)
        self._judge = LLMJudge()
        self._catalogue = FaultCatalogue()
        self._designer = AdversarialDesigner()
        self._curriculum = CurriculumController(self._catalogue, self._designer)
        self._rng = random.Random()

        # Per-episode state
        self._etl_state = ETLState()
        self._action_history: list[str] = []
        self._prev_kpi_ok: bool = False
        self._called_trace_lineage: bool = False
        self._tool_call_counts: dict[str, int] = {}
        self._current_alert: str = ""
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> ETLObservation:
        ep_seed = seed if seed is not None else self._rng.randint(0, 2**31)
        ep_id = episode_id or str(uuid4())

        # Pick fault from curriculum
        fault_spec = self._curriculum.pick_fault()

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

        _tier_name = self._curriculum.state_summary()["tiers"].get(
            fault_spec["fault_type"], "warmup"
        )
        diff_idx = ["warmup", "beginner", "intermediate", "advanced", "expert"].index(_tier_name)

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
        alert_options = _KPI_ALERTS.get(kpi, [f"ALERT: KPI anomaly detected in {kpi}."])
        self._current_alert = self._rng.choice(alert_options)

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
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: ETLAction, timeout_s: float | None = None, **kwargs) -> ETLObservation:  # type: ignore[override]
        self._etl_state.step += 1
        step = self._etl_state.step

        # --- Execute tool ---
        tool_result = self._dispatch_tool(action)

        # --- Track state for reward computation ---
        tool_key = f"{action.tool_name.value}:{action.tool_args}"
        repeated = self._tool_call_counts.get(tool_key, 0) > 0
        self._tool_call_counts[tool_key] = self._tool_call_counts.get(tool_key, 0) + 1

        if action.tool_name == ToolName.TRACE_LINEAGE:
            self._called_trace_lineage = True

        called_fix_without_lineage = (
            action.tool_name == ToolName.APPLY_FIX and not self._called_trace_lineage
        )
        malformed = not tool_result.success and action.tool_name != ToolName.VERIFY_OUTPUT

        # Check resolution
        resolved_this_step = False
        if action.tool_name == ToolName.APPLY_FIX and tool_result.success:
            verify = self._tool_handlers.verify_output(
                self._fault_injector.active_faults[0]["affected_kpi"]
            )
            if verify.success:
                resolved_this_step = True
                self._etl_state.resolved = True

        broke_something = (
            action.tool_name == ToolName.APPLY_FIX and not tool_result.success and self._prev_kpi_ok
        )

        wrong_fix = action.tool_name == ToolName.APPLY_FIX and not tool_result.success

        step_progressed = (
            tool_result.success
            and not repeated
            and action.tool_name not in (ToolName.VERIFY_OUTPUT,)
        )

        # --- Judge score ---
        judge_score = self._judge.score(
            alert=self._current_alert,
            compact_history=self._action_history[-5:],
            action=action,
            tool_result=tool_result,
        )

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
        )

        self._cumulative_reward += breakdown.total

        # --- Check episode end ---
        episode_done = resolved_this_step or step >= MAX_STEPS
        if episode_done:
            terminal = compute_terminal_reward(self._etl_state.resolved, step)
            breakdown.terminal = terminal
            self._cumulative_reward += terminal
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

        # Track KPI state for next step
        if action.tool_name == ToolName.VERIFY_OUTPUT and tool_result.success:
            self._prev_kpi_ok = True

        # Build compact history entry
        summary = f"[{step}] {action.tool_name.value}({action.tool_args}) → {'OK' if tool_result.success else 'FAIL'}"
        self._action_history.append(summary)

        diff_idx = self._etl_state.difficulty

        return ETLObservation(
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

        self._tool_handlers.reset()
        self._action_history = []
        self._called_trace_lineage = False
        self._tool_call_counts = {}
        self._cumulative_reward = 0.0
        self._prev_kpi_ok = False

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
    def _dispatch_tool(self, action: ETLAction) -> ToolResult:
        h = self._tool_handlers
        args = action.tool_args
        match action.tool_name:
            case ToolName.RUN_QUERY:
                return h.run_query(args.get("sql", ""))
            case ToolName.INSPECT_SCHEMA:
                return h.inspect_schema(args.get("table", ""))
            case ToolName.CHECK_ROW_COUNTS:
                return h.check_row_counts(args.get("table", ""))
            case ToolName.TRACE_LINEAGE:
                return h.trace_lineage(args.get("table", ""))
            case ToolName.SAMPLE_ROWS:
                return h.sample_rows(args.get("table", ""), args.get("n", 5))
            case ToolName.APPLY_FIX:
                return h.apply_fix(
                    args.get("fix_type", ""),
                    args.get("target", ""),
                    args.get("params", {}),
                )
            case ToolName.VERIFY_OUTPUT:
                return h.verify_output(args.get("kpi_name", ""))
            case _:
                return ToolResult(success=False, output=f"Unknown tool: {action.tool_name}")
