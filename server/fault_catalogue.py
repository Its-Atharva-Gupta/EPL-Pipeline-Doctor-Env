import random
from typing import TypedDict

from .constants import FAULT_TYPES, TIERS


class FaultSpec(TypedDict):
    fault_type: str
    target_table: str
    params: dict
    affected_kpi: str
    tier: str


# Base scenario pools per fault type and tier.
# Each entry is a FaultSpec without the 'tier' field (added at pick time).
_BASE_SCENARIOS: dict[str, list[dict]] = {
    "schema_drift": [
        {
            "fault_type": "schema_drift",
            "target_table": "bronze.orders_raw",
            "params": {"old_column": "total_amount", "new_column": "order_total"},
            "affected_kpi": "gold.kpi_daily_revenue",
        }
    ],
    "stale_partition": [
        {
            "fault_type": "stale_partition",
            "target_table": "silver.orders_enriched",
            "params": {},
            "affected_kpi": "gold.kpi_daily_revenue",
        }
    ],
    "null_explosion": [
        {
            "fault_type": "null_explosion",
            "target_table": "silver.orders_enriched",
            "params": {"column": "region", "null_fraction": 0.8},
            "affected_kpi": "gold.kpi_daily_revenue",
        }
    ],
    "fanout_join": [
        {
            "fault_type": "fanout_join",
            "target_table": "bronze.products_raw",
            "params": {"duplicate_rows": 1},
            "affected_kpi": "gold.kpi_category_mix",
        }
    ],
    "type_mismatch": [
        {
            "fault_type": "type_mismatch",
            "target_table": "bronze.orders_raw",
            "params": {"column": "total_amount", "cast_to": "TEXT"},
            "affected_kpi": "gold.kpi_daily_revenue",
        }
    ],
}


def build_cascade(fault_types: list[str], tier: str) -> list[FaultSpec]:
    """Build a multi-fault cascade spec list for advanced/expert tiers."""
    specs: list[FaultSpec] = []
    used_kpis: set[str] = set()
    for ft in fault_types:
        candidates = [s for s in _BASE_SCENARIOS.get(ft, []) if s["affected_kpi"] not in used_kpis]
        if not candidates:
            candidates = _BASE_SCENARIOS.get(ft, [])
        if candidates:
            spec = dict(candidates[0])
            spec["tier"] = tier
            specs.append(spec)  # type: ignore[arg-type]
            used_kpis.add(spec["affected_kpi"])
    return specs


class FaultCatalogue:
    """Manages scenario pools and picks fault specs for curriculum controller."""

    def __init__(self) -> None:
        # pool[fault_type][tier] = list of FaultSpec
        self._pool: dict[str, dict[str, list[FaultSpec]]] = {
            ft: {tier: [] for tier in TIERS} for ft in FAULT_TYPES
        }
        self._populate_base_scenarios()
        self._last_kpi: str | None = None

    def _populate_base_scenarios(self) -> None:
        for ft, scenarios in _BASE_SCENARIOS.items():
            for tier in TIERS:
                for s in scenarios:
                    spec: FaultSpec = {**s, "tier": tier}  # type: ignore[misc]
                    self._pool[ft][tier].append(spec)

    def add_scenario(self, spec: FaultSpec) -> None:
        ft = spec["fault_type"]
        tier = spec["tier"]
        if ft in self._pool and tier in self._pool[ft]:
            self._pool[ft][tier].append(spec)

    def pick(self, fault_type: str, tier: str, rng: random.Random) -> FaultSpec:
        """Pick a scenario, avoiding repeating the same KPI twice in a row."""
        candidates = list(self._pool[fault_type][tier])
        if not candidates:
            candidates = list(self._pool[fault_type]["warmup"])

        if self._last_kpi and len(candidates) > 1:
            filtered = [c for c in candidates if c["affected_kpi"] != self._last_kpi]
            if filtered:
                candidates = filtered

        spec = rng.choice(candidates)
        self._last_kpi = spec["affected_kpi"]
        return spec
