from typing import Final

WAREHOUSE_TABLES: Final[list[str]] = [
    "bronze.orders_raw",
    "bronze.products_raw",
    "silver.orders_enriched",
    "silver.daily_sales",
    "gold.kpi_daily_revenue",
    "gold.kpi_category_mix",
]

KPI_TABLES: Final[list[str]] = [
    "gold.kpi_daily_revenue",
    "gold.kpi_category_mix",
]

LINEAGE: Final[dict[str, list[str]]] = {
    "bronze.orders_raw": [],
    "bronze.products_raw": [],
    "silver.orders_enriched": ["bronze.orders_raw", "bronze.products_raw"],
    "silver.daily_sales": ["silver.orders_enriched"],
    "gold.kpi_daily_revenue": ["silver.daily_sales"],
    "gold.kpi_category_mix": ["silver.orders_enriched", "bronze.products_raw"],
}

DOWNSTREAM: Final[dict[str, list[str]]] = {
    "bronze.orders_raw": ["silver.orders_enriched"],
    "bronze.products_raw": ["silver.orders_enriched", "gold.kpi_category_mix"],
    "silver.orders_enriched": ["silver.daily_sales", "gold.kpi_category_mix"],
    "silver.daily_sales": ["gold.kpi_daily_revenue"],
    "gold.kpi_daily_revenue": [],
    "gold.kpi_category_mix": [],
}

FAULT_TYPES: Final[list[str]] = [
    "schema_drift",
    "stale_partition",
    "null_explosion",
    "fanout_join",
    "type_mismatch",
]

TIERS: Final[list[str]] = ["warmup", "beginner", "intermediate", "advanced", "expert"]

PROMOTION_THRESHOLD: Final[float] = 0.7
DEMOTION_THRESHOLD: Final[float] = 0.3

MAX_STEPS: Final[int] = 20

# Reward weights
W_OUTCOME: Final[float] = 0.5
W_REASONING: Final[float] = 0.15
W_EFFICIENCY: Final[float] = 0.15
W_PROGRESS: Final[float] = 0.2

# Dense progress reward parameters
R_PROGRESS_KPI_WEIGHT: Final[float] = 0.6
R_PROGRESS_FAULT_WEIGHT: Final[float] = 0.4
R_PROGRESS_DELTA_CLIP: Final[float] = 1.0

# Terminal rewards
R_TERMINAL_SUCCESS: Final[float] = 3.0
R_TERMINAL_TIMEOUT: Final[float] = -2.0
R_EFFICIENCY_STEP_BONUS_START: Final[int] = 5
R_EFFICIENCY_PER_EXTRA_STEP: Final[float] = 0.1

OLLAMA_MODEL: Final[str] = "gemma4:e4b"
OLLAMA_TIMEOUT: Final[float] = 8.0

MAX_CONCURRENT_ENVS: Final[int] = 4
