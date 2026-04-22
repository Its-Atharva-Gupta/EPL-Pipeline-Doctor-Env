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
