from enum import StrEnum
from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ToolName(StrEnum):
    RUN_QUERY = "run_query"
    INSPECT_SCHEMA = "inspect_schema"
    CHECK_ROW_COUNTS = "check_row_counts"
    TRACE_LINEAGE = "trace_lineage"
    SAMPLE_ROWS = "sample_rows"
    APPLY_FIX = "apply_fix"
    VERIFY_OUTPUT = "verify_output"


class ETLAction(Action):
    command: str = Field(..., description="SQL command or tool invocation (e.g., 'SELECT ...', 'INSPECT TABLE silver.orders_enriched', 'FIX coalesce_column silver.orders_enriched region UNKNOWN')")


class ToolResult(Action):
    success: bool = True
    output: str = ""
    data: dict | None = None


class ETLObservation(Observation):
    episode_id: str = Field(default="", description="Episode identifier for reproducibility/debugging")
    alert: str = Field(default="", description="Initial KPI anomaly alert")
    last_tool_output: str | None = None
    action_history: list[str] = Field(default_factory=list)
    available_kpis: list[str] = Field(default_factory=list)
    available_tables: list[str] = Field(default_factory=list)
    step: int = 0
    step_reward: float = 0.0
    step_reward_breakdown: dict[str, float] = Field(default_factory=dict)
    episode_done: bool = False
    difficulty: int = 0
    cumulative_reward: float = Field(
        default=0.0,
        description="Cumulative return so far (includes terminal reward once episode is done)",
    )
    terminal_reward: float = Field(
        default=0.0,
        description="Terminal reward granted when episode ends (0.0 for non-terminal steps)",
    )
    episode_return: float = Field(
        default=0.0,
        description="Final episode return (only non-zero when episode_done=True)",
    )
    # Optional debugging fields (only populated when enabled on the server).
    judge_raw: str | None = None
    judge_prompt: str | None = None
    judge_score: float | None = None
    judge_brief: str | None = None
    judge_cache_hit: bool | None = None
    judge_provider: str | None = None
    judge_model: str | None = None


class ETLState(State):
    episode_id: str = ""
    step: int = 0
    max_steps: int = 20
    cumulative_reward: float = 0.0
    difficulty: int = 0
    fault_type: str = ""  # HIDDEN from agent observations
    resolved: bool = False


ProviderName = Literal["anthropic", "openai", "groq", "openrouter", "ollama"]


class ProviderConfig(BaseModel):
    provider: ProviderName
    model: str
    api_key: str = ""
    base_url: str = ""

    model_config = {"extra": "forbid"}
