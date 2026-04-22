# Raw SQL Commands — ETL Pipeline Doctor

The agent now uses **raw SQL commands** instead of structured tool names. This is similar to how Kube SRE Gym uses raw `kubectl` commands.

## Action Format

Instead of:
```python
ETLAction(
    tool_name="inspect_schema",
    tool_args={"table": "gold.kpi_daily_revenue"},
    reasoning="Check the schema"
)
```

The agent now sends:
```python
ETLAction(command="INSPECT TABLE gold.kpi_daily_revenue")
```

---

## Command Types

### 1. **SELECT** — Run read-only SQL queries

```python
# Sample data
ETLAction(command="SELECT * FROM gold_kpi_daily_revenue LIMIT 5")

# Complex query
ETLAction(command="SELECT order_date, COUNT(*) FROM silver_orders_enriched GROUP BY order_date")

# Find NULLs
ETLAction(command="SELECT COUNT(*) FROM silver_orders_enriched WHERE region IS NULL")
```

**Returns:** Query results in tab-separated format (max 20 rows)

---

### 2. **INSPECT TABLE** — Get table schema

```python
ETLAction(command="INSPECT TABLE silver.orders_enriched")
```

**Returns:**
```
Table: silver.orders_enriched
Row count: 6000
Columns:
  order_id INTEGER  nulls=0
  customer_id INTEGER  nulls=0
  region TEXT  nulls=1200
```

---

### 3. **CHECK ROWS** — Get row counts by date

```python
ETLAction(command="CHECK ROWS gold.kpi_daily_revenue")
# or
ETLAction(command="CHECK gold.kpi_daily_revenue")
```

**Returns:**
```
Table: gold.kpi_daily_revenue
Total rows: 30
Rows by date:
  2024-01-01: 1
  2024-01-02: 1
  ...
```

---

### 4. **TRACE LINEAGE** — Find data dependencies

```python
ETLAction(command="TRACE LINEAGE silver.orders_enriched")
# or
ETLAction(command="TRACE silver.orders_enriched")
```

**Returns:**
```
Table: silver.orders_enriched
Upstream (feeds into this table): ['bronze.orders_raw']
Downstream (this table feeds into): ['silver.daily_sales', 'gold.kpi_daily_revenue']
```

---

### 5. **SAMPLE** — Sample random rows

```python
# Sample 5 rows (default)
ETLAction(command="SAMPLE silver.daily_sales")

# Sample 10 rows
ETLAction(command="SAMPLE silver.daily_sales 10")
```

**Returns:** Random rows in tab-separated format

---

### 6. **UPDATE** — Mutate data (fix issues)

```python
# Fill NULLs
ETLAction(command="UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL")

# Backfill missing partition
ETLAction(command="UPDATE silver_daily_sales SET gross_revenue = 0 WHERE gross_revenue IS NULL")

# Deduplicate
ETLAction(command="""
DELETE FROM silver_orders_enriched 
WHERE rowid NOT IN (SELECT MIN(rowid) FROM silver_orders_enriched GROUP BY order_id)
""")
```

**Returns:** Success/failure message

---

### 7. **INSERT** — Add data

```python
ETLAction(command="""
INSERT INTO silver_daily_sales (date, region, gross_revenue, order_count)
SELECT order_date, region, SUM(total_amount), COUNT(*) 
FROM silver_orders_enriched 
GROUP BY order_date, region
""")
```

**Returns:** Success/failure message

---

### 8. **VERIFY** — Check if KPI matches ground truth

```python
ETLAction(command="VERIFY")
```

**Returns:**
```
KPI verification PASS: gold.kpi_daily_revenue
  revenue: expected≈1200000.0000, actual=1200000.0000, diff=0.00% ✓
  yoy_growth_pct: expected≈0.0000, actual=0.0000, diff=0.00% ✓
```

---

## Command Parsing Rules

### **Case insensitive**
```python
ETLAction(command="select * from gold_kpi_daily_revenue")  # Works
ETLAction(command="SELECT * FROM gold_kpi_daily_revenue")  # Also works
```

### **Table names**
```python
"gold.kpi_daily_revenue"          # Full qualified name
"silver.orders_enriched"          # With schema
"gold_kpi_daily_revenue"          # Underscores (both supported)
```

### **Shortcuts**
```python
"INSPECT TABLE orders"            # Full syntax
"TRACE orders"                    # Shorthand (both work)
"CHECK ROWS orders"               # Full
"CHECK orders"                    # Shorthand
```

### **Multi-line SQL**
```python
ETLAction(command="""
UPDATE silver_orders_enriched 
SET region = COALESCE(region, 'UNKNOWN') 
WHERE region IS NULL
""")  # Indentation is OK, newlines are OK
```

---

## Safety Constraints

**Blocked operations:**
```python
ETLAction(command="DROP TABLE silver.orders_enriched")  # ❌ Blocked
ETLAction(command="ALTER TABLE gold.kpi_daily_revenue ADD COLUMN x INT")  # ❌ Blocked
ETLAction(command="CREATE TABLE my_table ...")  # ❌ Blocked
```

**Allowed mutations:**
```python
ETLAction(command="UPDATE ...")  # ✓ Allowed
ETLAction(command="INSERT ...")  # ✓ Allowed
```

**Allowed reads:**
```python
ETLAction(command="SELECT ...")  # ✓ Allowed
ETLAction(command="INSPECT ...")  # ✓ Allowed
```

---

## Example Episode

```python
from client import ETLPipelineDoctorEnv
from models import ETLAction

env = ETLPipelineDoctorEnv(base_url="http://localhost:8000")
obs = env.reset()

# Explore
obs = env.step(ETLAction(command="TRACE LINEAGE gold.kpi_daily_revenue"))
obs = env.step(ETLAction(command="INSPECT TABLE silver.orders_enriched"))
obs = env.step(ETLAction(command="SELECT COUNT(*) FROM silver_orders_enriched WHERE region IS NULL"))

# Fix
obs = env.step(ETLAction(command="UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL"))

# Verify
obs = env.step(ETLAction(command="VERIFY"))
print(f"Episode done: {obs.episode_done}")

env.close()
```

---

## Comparison: Before vs After

### Before (Structured Tools)
```python
ETLAction(
    tool_name="apply_fix",
    tool_args={
        "fix_type": "coalesce_column",
        "target": "silver.orders_enriched",
        "params": {"column": "region", "default": "UNKNOWN"}
    },
    reasoning="Fill NULLs in region column"
)
```

### After (Raw SQL)
```python
ETLAction(
    command="UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL"
)
```

**Advantages:**
- ✅ More flexible (custom SQL, not just predefined fixes)
- ✅ Like Kube (raw commands, not structured)
- ✅ Agent learns actual SQL
- ✅ Simpler parsing (no tool_args complexity)

---

## Reward Signal

After each command, the agent gets:
- `step_reward`: Overall reward for this step
- `step_reward_breakdown`: Detailed breakdown (r_outcome, r_reasoning, r_efficiency, r_penalty)
- `last_tool_output`: Result of the command
- `action_history`: Summary of all steps taken

**Example:**
```python
obs = env.step(ETLAction(command="INSPECT TABLE gold.kpi_daily_revenue"))
print(obs.step_reward)            # 0.27
print(obs.step_reward_breakdown)  # {"r_outcome": 0.0, "r_reasoning": 0.9, "r_efficiency": 0.1, "r_penalty": 0.0}
print(obs.last_tool_output)       # Table schema + null counts
```
