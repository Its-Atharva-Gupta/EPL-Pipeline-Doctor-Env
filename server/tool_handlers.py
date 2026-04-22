import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ToolResult

from .warehouse import Warehouse

logger = logging.getLogger(__name__)

_READ_ONLY_PATTERN = re.compile(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE)\b", re.IGNORECASE)
_DANGEROUS_PATTERNS = re.compile(r"\b(DROP|ALTER|CREATE)\b", re.IGNORECASE)


class CommandParseError(Exception):
    """Raised when command parsing fails."""
    pass


class ToolHandlers:
    """Implementations of all 7 agent tools."""

    def __init__(self, warehouse: Warehouse) -> None:
        self._wh = warehouse
        self._fix_log: dict[str, set[str]] = {}  # target -> set of applied fix keys

    def reset(self) -> None:
        self._fix_log = {}

    def dispatch_command(self, command: str) -> ToolResult:
        """Parse and dispatch a raw command to the appropriate handler.

        Supported command formats:
        - SELECT ... / SELECT * FROM ... (run_query)
        - INSPECT TABLE <table> (inspect_schema)
        - CHECK_ROWS <table> (check_row_counts)
        - TRACE <table> (trace_lineage)
        - SAMPLE <table> [n] (sample_rows)
        - UPDATE ... / INSERT ... (mutations, routed to apply_fix)
        - VERIFY (verify_output)
        """
        cmd = command.strip()
        if not cmd:
            return ToolResult(success=False, output="Empty command")

        # Route based on command prefix/keyword
        cmd_upper = cmd.upper()

        # SELECT (read-only query)
        if cmd_upper.startswith("SELECT"):
            return self.run_query(cmd)

        # INSPECT TABLE <table>
        if cmd_upper.startswith("INSPECT"):
            match = re.match(r"INSPECT\s+TABLE\s+(\S+)", cmd, re.IGNORECASE)
            if not match:
                return ToolResult(success=False, output="INSPECT usage: INSPECT TABLE <table>")
            table = match.group(1)
            return self.inspect_schema(table)

        # CHECK_ROWS <table>
        if cmd_upper.startswith("CHECK"):
            match = re.match(r"CHECK\s+(?:ROWS\s+)?(\S+)", cmd, re.IGNORECASE)
            if not match:
                return ToolResult(success=False, output="CHECK usage: CHECK <table> or CHECK ROWS <table>")
            table = match.group(1)
            return self.check_row_counts(table)

        # TRACE <table>
        if cmd_upper.startswith("TRACE"):
            match = re.match(r"TRACE\s+(?:LINEAGE\s+)?(\S+)", cmd, re.IGNORECASE)
            if not match:
                return ToolResult(success=False, output="TRACE usage: TRACE <table> or TRACE LINEAGE <table>")
            table = match.group(1)
            return self.trace_lineage(table)

        # SAMPLE <table> [n]
        if cmd_upper.startswith("SAMPLE"):
            match = re.match(r"SAMPLE\s+(\S+)(?:\s+(\d+))?", cmd, re.IGNORECASE)
            if not match:
                return ToolResult(success=False, output="SAMPLE usage: SAMPLE <table> [n]")
            table = match.group(1)
            n = int(match.group(2)) if match.group(2) else 5
            return self.sample_rows(table, n)

        # VERIFY (verify_output)
        if cmd_upper.startswith("VERIFY"):
            return self.verify_output(self._wh.lineage_graph.get("kpi", "gold.kpi_daily_revenue") if hasattr(self._wh, 'lineage_graph') else "gold.kpi_daily_revenue")

        # UPDATE / INSERT (mutations)
        if cmd_upper.startswith("UPDATE") or cmd_upper.startswith("INSERT"):
            return self._handle_mutation(cmd)

        # DROP / ALTER / CREATE (dangerous)
        if _DANGEROUS_PATTERNS.search(cmd):
            return ToolResult(success=False, output="Dangerous operations (DROP, ALTER, CREATE) are not allowed")

        return ToolResult(success=False, output=f"Unknown command format. Valid: SELECT, INSPECT, CHECK, TRACE, SAMPLE, VERIFY, UPDATE, INSERT")

    def _handle_mutation(self, sql: str) -> ToolResult:
        """Execute a mutation (UPDATE/INSERT) safely."""
        if _DANGEROUS_PATTERNS.search(sql):
            return ToolResult(success=False, output="Dangerous operations (DROP, ALTER, CREATE) are not allowed")

        try:
            self._wh.conn.executescript(sql)
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Mutation executed successfully")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.1 run_query
    # ------------------------------------------------------------------
    def run_query(self, sql: str) -> ToolResult:
        if _READ_ONLY_PATTERN.search(sql):
            return ToolResult(
                success=False,
                output="Read-only queries only. Use apply_fix to mutate.",
            )
        try:
            cursor = self._wh.conn.execute(sql)
            rows = cursor.fetchmany(21)
            headers = [d[0] for d in cursor.description] if cursor.description else []
            total_fetched = len(rows)
            display_rows = rows[:20]
            lines = ["\t".join(headers)]
            lines += ["\t".join(str(v) for v in row) for row in display_rows]
            if total_fetched > 20:
                extra = self._wh.conn.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0] - 20
                lines.append(f"... and {extra} more rows")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"rows": [dict(zip(headers, r)) for r in display_rows]},
            )
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.2 inspect_schema
    # ------------------------------------------------------------------
    def inspect_schema(self, table: str) -> ToolResult:
        sql_name = self._wh.table_sql_name(table)
        try:
            info = self._wh.conn.execute(f"PRAGMA table_info({sql_name})").fetchall()
            if not info:
                available = ", ".join(self._wh.available_tables())
                return ToolResult(
                    success=False,
                    output=f"Table '{table}' not found. Available: {available}",
                )
            count = self._wh.conn.execute(f"SELECT COUNT(*) FROM {sql_name}").fetchone()[0]
            col_lines: list[str] = []
            for col in info:
                name, ctype = col[1], col[2]
                null_count = self._wh.conn.execute(
                    f"SELECT COUNT(*) FROM {sql_name} WHERE {name} IS NULL"
                ).fetchone()[0]
                col_lines.append(f"  {name} {ctype}  nulls={null_count}")
            output = f"Table: {table}\nRow count: {count}\nColumns:\n" + "\n".join(col_lines)
            return ToolResult(
                success=True,
                output=output,
                data={
                    "table": table,
                    "row_count": count,
                    "columns": [{"name": c[1], "type": c[2]} for c in info],
                },
            )
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.3 check_row_counts
    # ------------------------------------------------------------------
    def check_row_counts(self, table: str) -> ToolResult:
        sql_name = self._wh.table_sql_name(table)
        try:
            total = self._wh.conn.execute(f"SELECT COUNT(*) FROM {sql_name}").fetchone()[0]
            # Check for date column
            info = self._wh.conn.execute(f"PRAGMA table_info({sql_name})").fetchall()
            date_col = next((c[1] for c in info if c[1] in ("date", "order_date")), None)
            lines = [f"Table: {table}", f"Total rows: {total}"]
            if date_col:
                breakdown = self._wh.conn.execute(
                    f"SELECT {date_col}, COUNT(*) as cnt FROM {sql_name} "
                    f"GROUP BY {date_col} ORDER BY {date_col}"
                ).fetchall()
                lines.append(f"Rows by {date_col}:")
                for row in breakdown[-10:]:  # last 10 dates
                    lines.append(f"  {row[0]}: {row[1]}")
                if len(breakdown) > 10:
                    lines.append(f"  ... ({len(breakdown) - 10} earlier dates omitted)")
            return ToolResult(success=True, output="\n".join(lines), data={"total": total})
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.4 trace_lineage
    # ------------------------------------------------------------------
    def trace_lineage(self, table: str) -> ToolResult:
        upstream = self._wh.lineage_upstream(table)
        downstream = self._wh.lineage_downstream(table)
        if not upstream and not downstream and table not in self._wh.available_tables():
            available = ", ".join(self._wh.available_tables())
            return ToolResult(
                success=False,
                output=f"Table '{table}' not found. Available: {available}",
            )
        output = (
            f"Table: {table}\n"
            f"Upstream (feeds into this table): {upstream or ['none']}\n"
            f"Downstream (this table feeds into): {downstream or ['none']}"
        )
        return ToolResult(
            success=True,
            output=output,
            data={"upstream": upstream, "downstream": downstream},
        )

    # ------------------------------------------------------------------
    # 9.5 sample_rows
    # ------------------------------------------------------------------
    def sample_rows(self, table: str, n: int = 5) -> ToolResult:
        n = min(n, 20)
        sql_name = self._wh.table_sql_name(table)
        try:
            rows = self._wh.conn.execute(
                f"SELECT * FROM {sql_name} ORDER BY RANDOM() LIMIT ?", (n,)
            ).fetchall()
            if not rows:
                return ToolResult(success=True, output=f"Table '{table}' is empty.")
            headers = rows[0].keys()
            lines = ["\t".join(headers)]
            lines += ["\t".join(str(v) for v in row) for row in rows]
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"rows": [dict(r) for r in rows]},
            )
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.6 apply_fix
    # ------------------------------------------------------------------
    def apply_fix(self, fix_type: str, target: str, params: dict[str, Any]) -> ToolResult:
        fix_key = f"{fix_type}:{target}:{sorted(params.items())}"
        if fix_key in self._fix_log.get(target, set()):
            return ToolResult(success=False, output="Fix already applied")

        match fix_type:
            case "rename_column":
                result = self._fix_rename_column(target, params)
            case "backfill_partition":
                result = self._fix_backfill_partition(target, params)
            case "coalesce_column":
                result = self._fix_coalesce_column(target, params)
            case "deduplicate":
                result = self._fix_deduplicate(target, params)
            case "cast_column":
                result = self._fix_cast_column(target, params)
            case "custom_sql":
                result = self._fix_custom_sql(target, params)
            case _:
                return ToolResult(
                    success=False,
                    output=f"Unknown fix_type '{fix_type}'. Valid: rename_column, backfill_partition, coalesce_column, deduplicate, cast_column, custom_sql",
                )

        if result.success:
            self._fix_log.setdefault(target, set()).add(fix_key)
            logger.info("Applied fix '%s' to '%s'", fix_type, target)
        return result

    def _fix_rename_column(self, target: str, params: dict) -> ToolResult:
        old = params.get("old", "")
        new = params.get("new", "")
        if not old or not new:
            return ToolResult(success=False, output="rename_column requires 'old' and 'new' params")
        sql_name = self._wh.table_sql_name(target)
        try:
            self._wh.conn.execute(f"ALTER TABLE {sql_name} RENAME COLUMN {old} TO {new}")
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Renamed column '{old}' → '{new}' in {target}")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    def _fix_backfill_partition(self, target: str, params: dict) -> ToolResult:
        date_val = params.get("date", "")
        source = params.get("source_table", "bronze.orders_raw")
        if not date_val:
            return ToolResult(success=False, output="backfill_partition requires 'date' param")
        sql_target = self._wh.table_sql_name(target)
        sql_source = self._wh.table_sql_name(source)
        try:
            # Re-derive silver.orders_enriched rows for the given date from bronze
            self._wh.conn.execute(f"DELETE FROM {sql_target} WHERE order_date = ?", (date_val,))
            self._wh.conn.execute(
                f"""
                INSERT INTO {sql_target} (order_id, customer_id, order_date, total_amount, product_count, region)
                SELECT o.order_id, o.customer_id, o.order_date, o.total_amount, 1, 'Unknown'
                FROM {sql_source} o
                WHERE o.order_date = ?
            """,
                (date_val,),
            )
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Backfilled partition {date_val} in {target}")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    def _fix_coalesce_column(self, target: str, params: dict) -> ToolResult:
        column = params.get("column", "")
        default = params.get("default", "UNKNOWN")
        if not column:
            return ToolResult(success=False, output="coalesce_column requires 'column' param")
        sql_name = self._wh.table_sql_name(target)
        try:
            self._wh.conn.execute(
                f"UPDATE {sql_name} SET {column} = COALESCE({column}, ?) WHERE {column} IS NULL",
                (default,),
            )
            self._wh.conn.commit()
            return ToolResult(
                success=True, output=f"Coalesced NULL {column} → '{default}' in {target}"
            )
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    def _fix_deduplicate(self, target: str, params: dict) -> ToolResult:
        columns: list[str] = params.get("columns", [])
        if not columns:
            return ToolResult(success=False, output="deduplicate requires 'columns' param")
        sql_name = self._wh.table_sql_name(target)
        col_list = ", ".join(columns)
        try:
            self._wh.conn.execute(f"""
                DELETE FROM {sql_name}
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM {sql_name} GROUP BY {col_list}
                )
            """)
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Deduplicated {target} on columns {columns}")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    def _fix_cast_column(self, target: str, params: dict) -> ToolResult:
        column = params.get("column", "")
        to_type = params.get("to_type", "REAL")
        if not column:
            return ToolResult(success=False, output="cast_column requires 'column' param")
        sql_name = self._wh.table_sql_name(target)
        try:
            # Strip corruption suffix and recast to numeric
            self._wh.conn.execute(
                f"UPDATE {sql_name} SET {column} = CAST(REPLACE({column}, '_corrupted', '') AS {to_type})"
            )
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Recast {column} to {to_type} in {target}")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    def _fix_custom_sql(self, target: str, params: dict) -> ToolResult:
        sql = params.get("sql", "")
        if not sql:
            return ToolResult(success=False, output="custom_sql requires 'sql' param")
        try:
            self._wh.conn.executescript(sql)
            self._wh.conn.commit()
            return ToolResult(success=True, output=f"Executed custom SQL on {target}")
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

    # ------------------------------------------------------------------
    # 9.7 verify_output
    # ------------------------------------------------------------------
    def verify_output(self, kpi_name: str, tolerance: float = 0.005) -> ToolResult:
        sql_name = self._wh.table_sql_name(kpi_name)
        ground_truth = self._wh.ground_truth(kpi_name)
        if not ground_truth:
            return ToolResult(success=False, output=f"No ground truth for '{kpi_name}'")

        try:
            current_rows = self._wh.conn.execute(f"SELECT * FROM {sql_name}").fetchall()
            current = [dict(r) for r in current_rows]
        except sqlite3.Error as exc:
            return ToolResult(success=False, output=str(exc))

        # Compare numeric totals rather than row-by-row to allow flexible fixes
        gt_numeric = _sum_numeric(ground_truth)
        cur_numeric = _sum_numeric(current)

        passed = True
        detail_lines: list[str] = []
        for col, expected in gt_numeric.items():
            actual = cur_numeric.get(col, 0.0)
            if expected == 0.0:
                diff = abs(actual)
            else:
                diff = abs(actual - expected) / abs(expected)
            ok = diff <= tolerance
            if not ok:
                passed = False
            detail_lines.append(
                f"  {col}: expected≈{expected:.4f}, actual={actual:.4f}, diff={diff:.4%} {'✓' if ok else '✗'}"
            )

        status = "PASS" if passed else "FAIL"
        output = f"KPI verification {status}: {kpi_name}\n" + "\n".join(detail_lines)
        return ToolResult(
            success=passed,
            output=output,
            data={"passed": passed, "kpi": kpi_name},
        )

    @property
    def fix_log(self) -> dict[str, set[str]]:
        return dict(self._fix_log)

    def has_called_trace_lineage(self) -> bool:
        return any(
            True for fixes in self._fix_log.values() if any("trace_lineage" in f for f in fixes)
        )


def _sum_numeric(rows: list[dict]) -> dict[str, float]:
    """Sum all numeric columns across rows."""
    totals: dict[str, float] = {}
    for row in rows:
        for k, v in row.items():
            try:
                totals[k] = totals.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                pass
    return totals
