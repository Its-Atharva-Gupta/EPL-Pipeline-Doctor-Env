"""
ETL Pipeline Doctor Environment Client.

Uses OpenEnv's EnvClient for stateful WebSocket sessions.
One environment instance persists across reset() and step() calls.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

sys.path.insert(0, str(Path(__file__).parent))

from models import ETLAction, ETLObservation, ETLState


class ETLPipelineDoctorEnvAsync(EnvClient[ETLAction, ETLObservation, ETLState]):
    """
    Async client for the ETL Pipeline Doctor environment.

    Uses WebSocket (/ws) to maintain stateful sessions.
    One environment instance per connection — state persists across reset/step.
    """

    def __init__(self, base_url: str = "http://localhost:8000", **kwargs) -> None:
        kwargs.setdefault("message_timeout_s", 120.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: ETLAction) -> Dict[str, Any]:
        """Serialize ETLAction to JSON for WebSocket transmission."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ETLObservation]:
        """Parse WebSocket response into ETLObservation."""
        obs_data = payload.get("observation", {})
        observation = ETLObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ETLState:
        """Parse state snapshot for debugging."""
        payload.pop("judge", None)
        return ETLState(**payload)


class ETLPipelineDoctorEnv:
    """
    Synchronous wrapper around ETLPipelineDoctorEnvAsync.

    Provides sync reset() and step() methods by running async code in event loop.

    Example:
        >>> env = ETLPipelineDoctorEnv(base_url="http://localhost:8000")
        >>> obs = env.reset()
        >>> obs = env.step(ETLAction(tool_name="trace_lineage", tool_args={"table": "gold.kpi_daily_revenue"}))
        >>> env.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000", **kwargs) -> None:
        self._async_env = ETLPipelineDoctorEnvAsync(base_url=base_url, **kwargs)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def reset(self, **kwargs) -> ETLObservation:
        """Reset the environment (sync)."""
        result = self._loop.run_until_complete(self._async_env.reset(**kwargs))
        return result.observation

    def step(self, action: ETLAction, **kwargs) -> ETLObservation:
        """Execute an action (sync)."""
        result = self._loop.run_until_complete(self._async_env.step(action, **kwargs))
        return result.observation

    def state(self, **kwargs) -> ETLState:
        """Get environment state (sync)."""
        result = self._loop.run_until_complete(self._async_env.state(**kwargs))
        return result

    def close(self) -> None:
        """Close the connection and clean up."""
        try:
            self._loop.run_until_complete(self._async_env.close())
        finally:
            self._loop.close()

    def __enter__(self) -> "ETLPipelineDoctorEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# Example usage helper
def example_commands() -> None:
    """Print example commands for the agent."""
    examples = """
    # Exploration commands:
    SELECT * FROM gold_kpi_daily_revenue ORDER BY date DESC LIMIT 5
    INSPECT TABLE silver.orders_enriched
    CHECK ROWS gold.kpi_daily_revenue
    TRACE LINEAGE silver.orders_enriched
    SAMPLE silver.daily_sales 10

    # Mutation commands:
    UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL
    INSERT INTO silver_daily_sales SELECT date, region, SUM(total_amount), COUNT(*) FROM silver_orders_enriched GROUP BY date, region

    # Verification:
    VERIFY
    """
    print(examples)
