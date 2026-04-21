import json
import logging
import pathlib
from typing import Any

from .constants import FAULT_TYPES, KPI_TABLES, TIERS, WAREHOUSE_TABLES
from .fault_catalogue import FaultSpec
from .llm_client import get_llm_response

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "designer_system.md").read_text()

_VALID_FAULT_TYPES = set(FAULT_TYPES)
_VALID_TABLES = set(WAREHOUSE_TABLES)
_VALID_KPIS = set(KPI_TABLES)
_VALID_TIERS = set(TIERS)


class AdversarialDesigner:
    """Generates novel fault specs targeting the agent's weak spots."""

    def design(
        self,
        worst_fault_type: str,
        failure_rate: float,
        underused_tools: list[str],
        common_wrong_fix: str,
        current_tier: str,
        config: Any = None,  # ProviderConfig | None
    ) -> FaultSpec | None:
        user_prompt = _build_designer_prompt(
            worst_fault_type, failure_rate, underused_tools, common_wrong_fix, current_tier
        )
        try:
            raw = get_llm_response(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=config,
            )
            return _parse_and_validate(raw)
        except Exception as exc:
            logger.warning("Adversarial designer failed: %s", exc)
            return None


def _build_designer_prompt(
    worst_fault_type: str,
    failure_rate: float,
    underused_tools: list[str],
    common_wrong_fix: str,
    current_tier: str,
) -> str:
    return (
        f"AGENT FAILURE PATTERN:\n"
        f"  Hardest fault type last 20 eps: {worst_fault_type} (failure rate: {failure_rate:.0%})\n"
        f"  Tool underuse: {underused_tools}\n"
        f"  Common wrong fix: {common_wrong_fix}\n\n"
        f"CURRENT TIER: {current_tier}\n\n"
        "Design a fault spec targeting this agent's weakest skill."
    )


def _parse_and_validate(raw: str) -> FaultSpec | None:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        data = json.loads(raw[start:end])

        ft = data.get("fault_type", "")
        table = data.get("target_table", "")
        kpi = data.get("affected_kpi", "")
        tier = data.get("tier", "beginner")

        if ft not in _VALID_FAULT_TYPES:
            logger.warning("Designer returned invalid fault_type: %s", ft)
            return None
        if table not in _VALID_TABLES:
            logger.warning("Designer returned invalid target_table: %s", table)
            return None
        if kpi not in _VALID_KPIS:
            logger.warning("Designer returned invalid affected_kpi: %s", kpi)
            return None
        if tier not in _VALID_TIERS:
            tier = "beginner"

        return FaultSpec(
            fault_type=ft,
            target_table=table,
            params=data.get("params", {}),
            affected_kpi=kpi,
            tier=tier,
        )
    except Exception as exc:
        logger.warning("Failed to parse designer response: %s", exc)
        return None
