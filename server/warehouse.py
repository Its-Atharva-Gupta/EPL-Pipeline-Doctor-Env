import logging
import random
import sqlite3
from datetime import date, timedelta

from .constants import KPI_TABLES, LINEAGE, WAREHOUSE_TABLES

logger = logging.getLogger(__name__)

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Sports"]
STATUSES = ["completed", "pending", "cancelled"]


def _table_to_sql_name(qualified: str) -> str:
    """Convert 'schema.table' → 'schema_table' for SQLite (no schema support)."""
    return qualified.replace(".", "_")


def seed_warehouse(conn: sqlite3.Connection, seed: int = 42, noise_level: float = 0.1) -> None:
    """Create and populate all warehouse tables deterministically."""
    rng = random.Random(seed)
    conn.row_factory = sqlite3.Row

    _create_tables(conn)
    start_date = date(2024, 1, 1)
    days = 30

    # --- bronze.orders_raw ---
    orders: list[tuple] = []
    order_id = 1
    for day_offset in range(days):
        order_date = start_date + timedelta(days=day_offset)
        n_orders = int(200 * (1 + rng.uniform(-noise_level, noise_level)))
        for _ in range(n_orders):
            customer_id = rng.randint(1, 500)
            total_amount = round(rng.uniform(10.0, 500.0), 2)
            status = rng.choice(STATUSES)
            orders.append((order_id, customer_id, str(order_date), total_amount, status))
            order_id += 1

    conn.executemany("INSERT INTO bronze_orders_raw VALUES (?,?,?,?,?)", orders)

    # --- bronze.products_raw ---
    products: list[tuple] = []
    for pid in range(1, 51):
        name = f"Product_{pid}"
        category = rng.choice(CATEGORIES)
        unit_price = round(rng.uniform(5.0, 200.0), 2)
        created_at = str(start_date - timedelta(days=rng.randint(1, 365)))
        products.append((pid, name, category, unit_price, created_at))
    conn.executemany("INSERT INTO bronze_products_raw VALUES (?,?,?,?,?)", products)

    # --- silver.orders_enriched ---
    region_map = {cid: rng.choice(REGIONS) for cid in range(1, 501)}
    enriched: list[tuple] = []
    for oid, cid, odate, amount, status in orders:
        product_count = rng.randint(1, 5)
        region = region_map[cid]
        enriched.append((oid, cid, odate, amount, product_count, region))
    conn.executemany("INSERT INTO silver_orders_enriched VALUES (?,?,?,?,?,?)", enriched)

    # --- silver.daily_sales ---
    daily: dict[tuple[str, str], tuple[float, int]] = {}
    for _, _, odate, amount, _, region in enriched:
        key = (odate, region)
        prev_rev, prev_cnt = daily.get(key, (0.0, 0))
        daily[key] = (prev_rev + amount, prev_cnt + 1)
    conn.executemany(
        "INSERT INTO silver_daily_sales VALUES (?,?,?,?)",
        [(d, r, rev, cnt) for (d, r), (rev, cnt) in daily.items()],
    )

    # --- gold.kpi_daily_revenue ---
    date_revenue: dict[str, float] = {}
    for (d, _), (rev, _) in daily.items():
        date_revenue[d] = date_revenue.get(d, 0.0) + rev
    sorted_dates = sorted(date_revenue.keys())
    kpi_rev: list[tuple] = []
    for i, d in enumerate(sorted_dates):
        yoy = 0.0  # no prior-year data in 30-day window
        kpi_rev.append((d, round(date_revenue[d], 2), yoy))
    conn.executemany("INSERT INTO gold_kpi_daily_revenue VALUES (?,?,?)", kpi_rev)

    # --- gold.kpi_category_mix ---
    category_map = {pid: cat for pid, _, cat, _, _ in products}
    cat_rev: dict[tuple[str, str], float] = {}
    for _, cid, odate, amount, product_count, _ in enriched:
        pid = ((cid + int(odate[-2:])) % 50) + 1
        cat = category_map.get(pid, CATEGORIES[0])
        key = (odate, cat)
        cat_rev[key] = cat_rev.get(key, 0.0) + amount

    date_totals: dict[str, float] = {}
    for (d, _), rev in cat_rev.items():
        date_totals[d] = date_totals.get(d, 0.0) + rev

    kpi_cat: list[tuple] = []
    for (d, cat), rev in cat_rev.items():
        share = round(rev / date_totals[d], 6) if date_totals[d] > 0 else 0.0
        kpi_cat.append((d, cat, share))
    conn.executemany("INSERT INTO gold_kpi_category_mix VALUES (?,?,?)", kpi_cat)

    conn.commit()
    logger.info("Seeded warehouse with seed=%d, %d orders over %d days", seed, len(orders), days)


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS bronze_orders_raw;
        DROP TABLE IF EXISTS bronze_products_raw;
        DROP TABLE IF EXISTS silver_orders_enriched;
        DROP TABLE IF EXISTS silver_daily_sales;
        DROP TABLE IF EXISTS gold_kpi_daily_revenue;
        DROP TABLE IF EXISTS gold_kpi_category_mix;

        CREATE TABLE bronze_orders_raw (
            order_id      INTEGER PRIMARY KEY,
            customer_id   INTEGER,
            order_date    TEXT,
            total_amount  REAL,
            status        TEXT
        );
        CREATE TABLE bronze_products_raw (
            product_id    INTEGER PRIMARY KEY,
            name          TEXT,
            category      TEXT,
            unit_price    REAL,
            created_at    TEXT
        );
        CREATE TABLE silver_orders_enriched (
            order_id      INTEGER PRIMARY KEY,
            customer_id   INTEGER,
            order_date    TEXT,
            total_amount  REAL,
            product_count INTEGER,
            region        TEXT
        );
        CREATE TABLE silver_daily_sales (
            date          TEXT,
            region        TEXT,
            gross_revenue REAL,
            order_count   INTEGER
        );
        CREATE TABLE gold_kpi_daily_revenue (
            date           TEXT PRIMARY KEY,
            revenue        REAL,
            yoy_growth_pct REAL
        );
        CREATE TABLE gold_kpi_category_mix (
            date           TEXT,
            category       TEXT,
            revenue_share  REAL
        );
    """)


class Warehouse:
    """Manages a per-episode in-memory SQLite warehouse."""

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._ground_truth: dict[str, list[dict]] = {}

    def setup(self, seed: int = 42) -> None:
        """Tear down any prior connection and seed a fresh in-memory DB."""
        if self._conn is not None:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        seed_warehouse(self._conn, seed=seed)
        self._capture_ground_truth()

    def _capture_ground_truth(self) -> None:
        assert self._conn is not None
        self._ground_truth = {}
        for kpi in KPI_TABLES:
            sql_name = _table_to_sql_name(kpi)
            rows = self._conn.execute(f"SELECT * FROM {sql_name}").fetchall()
            self._ground_truth[kpi] = [dict(r) for r in rows]
        logger.info("Ground truth captured for %d KPI tables", len(self._ground_truth))

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Warehouse not initialised — call setup() first"
        return self._conn

    def ground_truth(self, kpi: str) -> list[dict]:
        return self._ground_truth.get(kpi, [])

    def table_sql_name(self, qualified: str) -> str:
        return _table_to_sql_name(qualified)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def available_tables(self) -> list[str]:
        return list(WAREHOUSE_TABLES)

    def lineage_upstream(self, table: str) -> list[str]:
        return LINEAGE.get(table, [])

    def lineage_downstream(self, table: str) -> list[str]:
        from .constants import DOWNSTREAM

        return DOWNSTREAM.get(table, [])

    # ------------------------------------------------------------------
    # Simple pipeline recomputation helpers (used for training environment)
    # ------------------------------------------------------------------
    def recompute_daily_sales(self) -> bool:
        """Recompute silver.daily_sales from silver.orders_enriched.

        Returns True on success; on failure, clears downstream tables so KPIs break.
        """
        conn = self.conn
        try:
            conn.execute("DELETE FROM silver_daily_sales")
            conn.execute(
                """
                INSERT INTO silver_daily_sales (date, region, gross_revenue, order_count)
                SELECT
                    order_date AS date,
                    region,
                    SUM(total_amount) AS gross_revenue,
                    COUNT(*) AS order_count
                FROM silver_orders_enriched
                WHERE region IS NOT NULL AND total_amount IS NOT NULL
                GROUP BY order_date, region
                ORDER BY order_date
                """
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.warning("Failed to recompute silver.daily_sales: %s", exc)
            # Clear downstream so verification fails loudly.
            try:
                conn.execute("DELETE FROM silver_daily_sales")
                conn.execute("DELETE FROM gold_kpi_daily_revenue")
                conn.commit()
            except sqlite3.Error:
                pass
            return False

    def recompute_kpi_daily_revenue(self) -> bool:
        """Recompute gold.kpi_daily_revenue from silver.daily_sales."""
        conn = self.conn
        try:
            conn.execute("DELETE FROM gold_kpi_daily_revenue")
            conn.execute(
                """
                INSERT INTO gold_kpi_daily_revenue (date, revenue, yoy_growth_pct)
                SELECT
                    date,
                    ROUND(SUM(gross_revenue), 2) AS revenue,
                    0.0 AS yoy_growth_pct
                FROM silver_daily_sales
                GROUP BY date
                ORDER BY date
                """
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.warning("Failed to recompute gold.kpi_daily_revenue: %s", exc)
            try:
                conn.execute("DELETE FROM gold_kpi_daily_revenue")
                conn.commit()
            except sqlite3.Error:
                pass
            return False

    def recompute_kpi_category_mix(self) -> bool:
        """Recompute gold.kpi_category_mix from silver.orders_enriched + bronze.products_raw.

        Intentionally uses an un-duplicated denominator so fanout joins can break the KPI.
        """
        conn = self.conn
        try:
            conn.execute("DELETE FROM gold_kpi_category_mix")
            conn.execute(
                """
                WITH orders AS (
                    SELECT
                        order_date AS date,
                        customer_id,
                        total_amount,
                        ((customer_id + CAST(substr(order_date, -2) AS INT)) % 50) + 1 AS product_id
                    FROM silver_orders_enriched
                    WHERE total_amount IS NOT NULL
                ),
                denom AS (
                    SELECT date, SUM(total_amount) AS total_rev
                    FROM orders
                    GROUP BY date
                ),
                joined AS (
                    SELECT o.date, p.category, o.total_amount
                    FROM orders o
                    JOIN bronze_products_raw p
                      ON p.product_id = o.product_id
                )
                INSERT INTO gold_kpi_category_mix (date, category, revenue_share)
                SELECT
                    j.date,
                    j.category,
                    ROUND(SUM(j.total_amount) / d.total_rev, 6) AS revenue_share
                FROM joined j
                JOIN denom d
                  ON d.date = j.date
                GROUP BY j.date, j.category
                ORDER BY j.date, j.category
                """
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.warning("Failed to recompute gold.kpi_category_mix: %s", exc)
            try:
                conn.execute("DELETE FROM gold_kpi_category_mix")
                conn.commit()
            except sqlite3.Error:
                pass
            return False

    def recompute_downstream_from(self, qualified_table: str) -> None:
        """Recompute downstream tables that depend on qualified_table."""
        # For our small DAG, hardcode downstream recompute ordering.
        if qualified_table in ("silver.orders_enriched", "silver_orders_enriched"):
            self.recompute_daily_sales()
            self.recompute_kpi_daily_revenue()
            self.recompute_kpi_category_mix()
        elif qualified_table in ("silver.daily_sales", "silver_daily_sales"):
            self.recompute_kpi_daily_revenue()
        elif qualified_table in ("bronze.products_raw", "bronze_products_raw"):
            self.recompute_kpi_category_mix()
