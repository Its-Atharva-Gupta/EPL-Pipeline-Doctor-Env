import logging
import sqlite3
from typing import Any

from .fault_catalogue import FaultSpec
from .warehouse import Warehouse

logger = logging.getLogger(__name__)


class FaultInjector:
    """Applies fault specs to the warehouse and tracks injection state."""

    def __init__(self, warehouse: Warehouse) -> None:
        self._wh = warehouse
        self._active_faults: list[FaultSpec] = []

    def inject(self, specs: list[FaultSpec]) -> None:
        """Inject one or more faults into the warehouse."""
        self._active_faults = list(specs)
        for spec in specs:
            self._apply(spec)

    def _apply(self, spec: FaultSpec) -> None:
        conn = self._wh.conn
        ft = spec["fault_type"]
        table = self._wh.table_sql_name(spec["target_table"])
        params = spec.get("params", {})

        match ft:
            case "schema_drift":
                self._schema_drift(conn, table, params)
            case "stale_partition":
                self._stale_partition(conn, table)
            case "null_explosion":
                self._null_explosion(conn, table, params)
            case "fanout_join":
                self._fanout_join(conn, table, params)
            case "type_mismatch":
                self._type_mismatch(conn, table, params)
            case _:
                logger.warning("Unknown fault type: %s", ft)

        conn.commit()
        logger.info("Injected fault '%s' into table '%s'", ft, table)

    def _schema_drift(self, conn: sqlite3.Connection, table: str, params: dict[str, Any]) -> None:
        old_col = params.get("old_column", "total_amount")
        new_col = params.get("new_column", "order_total")
        # SQLite ALTER TABLE RENAME COLUMN requires SQLite >= 3.25
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")

    def _stale_partition(self, conn: sqlite3.Connection, table: str) -> None:
        # Delete the most recent day's rows
        result = conn.execute(f"SELECT MAX(order_date) FROM {table}").fetchone()
        if result and result[0]:
            max_date = result[0]
            conn.execute(f"DELETE FROM {table} WHERE order_date = ?", (max_date,))
            logger.info("Deleted stale partition date=%s from %s", max_date, table)

    def _null_explosion(self, conn: sqlite3.Connection, table: str, params: dict[str, Any]) -> None:
        column = params.get("column", "region")
        fraction = float(params.get("null_fraction", 0.8))
        # Null out the given fraction of rows using rowid modulo
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        threshold = int(total * fraction)
        conn.execute(
            f"UPDATE {table} SET {column} = NULL WHERE rowid IN "
            f"(SELECT rowid FROM {table} ORDER BY rowid LIMIT ?)",
            (threshold,),
        )

    def _fanout_join(self, conn: sqlite3.Connection, table: str, params: dict[str, Any]) -> None:
        # Duplicate a subset of rows to cause join fan-out.
        # SQLite PRIMARY KEY prevents direct duplicates, so recreate table without PK.
        n_dupes = int(params.get("duplicate_rows", 1))
        all_rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        placeholders = ", ".join(["?"] * len(cols))

        # Recreate without constraints so duplicates can be inserted
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"CREATE TABLE {table} ({col_defs})")
        conn.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})", [tuple(r) for r in all_rows]
        )

        dupe_rows = all_rows[:n_dupes]
        conn.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})", [tuple(r) for r in dupe_rows]
        )

    def _type_mismatch(self, conn: sqlite3.Connection, table: str, params: dict[str, Any]) -> None:
        column = params.get("column", "total_amount")
        cast_to = params.get("cast_to", "TEXT")
        # SQLite is dynamically typed; we simulate type mismatch by storing a text prefix
        if cast_to.upper() == "TEXT":
            # Prefix forces SQLite SUM()/numeric coercion to treat values as 0.0.
            conn.execute(
                f"UPDATE {table} SET {column} = 'corrupted_' || CAST({column} AS TEXT) || '_corrupted'"
            )

    @property
    def active_faults(self) -> list[FaultSpec]:
        return list(self._active_faults)
