import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ETLAction, ToolResult

from .constants import (
    R_EFFICIENCY_PER_EXTRA_STEP,
    R_EFFICIENCY_STEP_BONUS_START,
    R_PROGRESS_DELTA_CLIP,
    R_PROGRESS_FAULT_WEIGHT,
    R_PROGRESS_KPI_WEIGHT,
    R_TERMINAL_SUCCESS,
    R_TERMINAL_TIMEOUT,
    W_EFFICIENCY,
    W_OUTCOME,
    W_PROGRESS,
    W_REASONING,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    outcome: float = 0.0
    reasoning: float = 0.0
    efficiency: float = 0.0
    penalty: float = 0.0
    progress: float = 0.0
    total: float = 0.0
    terminal: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class DenseProgress:
    kpi_proximity: float = 0.0
    fault_progress: float = 0.0
    kpi_delta: float = 0.0
    fault_delta: float = 0.0


def compute_step_reward(
    action: ETLAction,
    tool_result: ToolResult,
    judge_score: float,
    resolved_this_step: bool,
    broke_something: bool,
    repeated_tool_call: bool,
    wrong_fix_type: bool,
    called_apply_fix_without_lineage: bool,
    malformed_args: bool,
    step_progressed: bool,
    action_repeat_count: int = 0,
    dense_progress: DenseProgress | None = None,
) -> RewardBreakdown:
    # r_outcome
    if resolved_this_step:
        r_outcome = 1.0
    elif broke_something:
        r_outcome = -1.0
    else:
        r_outcome = 0.0

    # r_reasoning from judge
    r_reasoning = float(judge_score)

    # r_efficiency
    r_efficiency = 0.0
    if step_progressed:
        r_efficiency += 0.1
    if repeated_tool_call:
        # Penalize repeated actions with increasing severity
        # 1st repeat: -0.1, 2nd repeat: -0.2, 3rd repeat: -0.3, etc.
        r_efficiency -= 0.1 * action_repeat_count
    if wrong_fix_type:
        r_efficiency -= 0.2

    # r_progress — dense per-step improvement signal
    r_progress = 0.0
    if dense_progress is not None:
        raw_progress = (
            R_PROGRESS_KPI_WEIGHT * dense_progress.kpi_delta
            + R_PROGRESS_FAULT_WEIGHT * dense_progress.fault_delta
        )
        r_progress = max(-R_PROGRESS_DELTA_CLIP, min(R_PROGRESS_DELTA_CLIP, raw_progress))

    # r_penalty
    r_penalty = 0.0
    if called_apply_fix_without_lineage:
        r_penalty -= 0.5
    if malformed_args:
        r_penalty -= 0.3

    total = (
        W_OUTCOME * r_outcome
        + W_REASONING * r_reasoning
        + W_EFFICIENCY * r_efficiency
        + W_PROGRESS * r_progress
        + r_penalty
    )

    bd = RewardBreakdown(
        outcome=r_outcome,
        reasoning=r_reasoning,
        efficiency=r_efficiency,
        penalty=r_penalty,
        progress=r_progress,
        total=total,
        details={
            "r_outcome": r_outcome,
            "r_reasoning": r_reasoning,
            "r_efficiency": r_efficiency,
            "r_penalty": r_penalty,
            "r_progress": r_progress,
            "kpi_proximity": dense_progress.kpi_proximity if dense_progress else 0.0,
            "fault_progress": dense_progress.fault_progress if dense_progress else 0.0,
            "kpi_delta": dense_progress.kpi_delta if dense_progress else 0.0,
            "fault_delta": dense_progress.fault_delta if dense_progress else 0.0,
        },
    )
    return bd


def compute_terminal_reward(resolved: bool, steps_used: int) -> float:
    if resolved:
        r = R_TERMINAL_SUCCESS
        extra = max(0, steps_used - R_EFFICIENCY_STEP_BONUS_START)
        r -= R_EFFICIENCY_PER_EXTRA_STEP * extra
        return r
    return R_TERMINAL_TIMEOUT
