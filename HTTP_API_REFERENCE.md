# HTTP API Reference — ETL Pipeline Doctor

Complete request/response formats for the environment server.

---

## Overview

The server runs at `http://localhost:8000` and uses OpenEnv standard endpoints.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/reset` | POST | Start new episode, return initial observation |
| `/step` | POST | Execute action, return observation + reward |
| `/state` | GET | Get current environment state (hidden fault, etc.) |
| `/schema` | GET | Get JSON schema for Action/Observation/State |
| `/configure` | POST | Set LLM provider (judge backend) |
| `/health` or `/healthz` | GET | Health check |

---

## POST /reset

**Start a new episode.**

### Request

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{
    "seed": 42,
    "episode_id": "ep-001"
  }'
```

**Fields:**
- `seed` (optional, int): RNG seed for reproducibility
- `episode_id` (optional, str): Custom episode ID

### Response (200 OK)

```json
{
  "observation": {
    "alert": "ALERT: Daily revenue KPI dropped unexpectedly. Expected ~$40,000/day, seeing anomalous values.",
    "last_tool_output": null,
    "action_history": [],
    "available_kpis": ["gold.kpi_daily_revenue", "gold.kpi_category_mix"],
    "available_tables": [
      "bronze.orders_raw",
      "bronze.products_raw",
      "silver.orders_enriched",
      "silver.daily_sales",
      "gold.kpi_daily_revenue",
      "gold.kpi_category_mix"
    ],
    "step": 0,
    "step_reward": 0.0,
    "step_reward_breakdown": {},
    "episode_done": false,
    "difficulty": 0,
    "done": false,
    "reward": 0.0
  }
}
```

**Fields:**
- `alert`: KPI anomaly alert (human-readable, read by agent)
- `available_tables`: Table names accessible via tools
- `available_kpis`: KPI names for verify_output tool
- `step`: Current step (0 at reset)
- `difficulty`: Curriculum difficulty tier (0-4)
- `episode_done`: False at reset, True when fault resolved or max steps reached

---

## POST /step

**Execute an action and get reward + next observation.**

### Request

⚠️ **Important:** Action is wrapped in an `"action"` field.

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "tool_name": "trace_lineage",
      "tool_args": {
        "table": "gold.kpi_daily_revenue"
      },
      "reasoning": "Starting diagnosis: trace the KPI upstream to identify data dependencies"
    }
  }'
```

**Fields:**
- `tool_name` (str, enum): One of:
  - `run_query` — execute read-only SQL
  - `inspect_schema` — get column names, types, null counts
  - `check_row_counts` — get row counts with date breakdown
  - `trace_lineage` — get upstream/downstream tables
  - `sample_rows` — random sample from table
  - `apply_fix` — apply a repair (mutating)
  - `verify_output` — check KPI against ground truth

- `tool_args` (dict): Arguments for the tool
- `reasoning` (str): Agent's thought process (read by judge)

### Tool Arguments Reference

#### trace_lineage
```json
{"tool_name": "trace_lineage", "tool_args": {"table": "gold.kpi_daily_revenue"}}
```
Returns upstream and downstream tables.

#### inspect_schema
```json
{"tool_name": "inspect_schema", "tool_args": {"table": "silver.orders_enriched"}}
```
Returns column names, types, null counts, row count.

#### check_row_counts
```json
{"tool_name": "check_row_counts", "tool_args": {"table": "silver.orders_enriched"}}
```
Returns row count and date partition breakdown.

#### sample_rows
```json
{
  "tool_name": "sample_rows",
  "tool_args": {
    "table": "silver.orders_enriched",
    "n": 5
  }
}
```
Returns random sample (n ≤ 20).

#### run_query
```json
{
  "tool_name": "run_query",
  "tool_args": {
    "sql": "SELECT COUNT(*) FROM silver.orders_enriched WHERE region IS NULL"
  }
}
```
Execute read-only query. Returns results or error.

#### apply_fix
```json
{
  "tool_name": "apply_fix",
  "tool_args": {
    "fix_type": "coalesce_column",
    "target": "silver.orders_enriched",
    "params": {
      "column": "region",
      "default": "UNKNOWN"
    }
  }
}
```
**Fix types:**
- `rename_column` — params: `{old: str, new: str}`
- `backfill_partition` — params: `{date: str, source_table: str}`
- `coalesce_column` — params: `{column: str, default: any}`
- `deduplicate` — params: `{columns: [str]}`
- `cast_column` — params: `{column: str, to_type: str}`
- `custom_sql` — params: `{sql: str}` (escape hatch, penalized)

#### verify_output
```json
{
  "tool_name": "verify_output",
  "tool_args": {
    "kpi_name": "gold.kpi_daily_revenue"
  }
}
```
Compare KPI to ground truth. Returns pass/fail.

### Response (200 OK)

```json
{
  "observation": {
    "alert": "ALERT: Daily revenue KPI dropped unexpectedly...",
    "last_tool_output": "Table: gold.kpi_daily_revenue\nUpstream (feeds into this table): ['silver.daily_sales']\nDownstream (this table feeds into): []",
    "action_history": [
      "[1] trace_lineage({'table': 'gold.kpi_daily_revenue'}) → OK"
    ],
    "available_kpis": ["gold.kpi_daily_revenue", "gold.kpi_category_mix"],
    "available_tables": [
      "bronze.orders_raw",
      "bronze.products_raw",
      "silver.orders_enriched",
      "silver.daily_sales",
      "gold.kpi_daily_revenue",
      "gold.kpi_category_mix"
    ],
    "step": 1,
    "step_reward": 0.26,
    "step_reward_breakdown": {
      "r_outcome": 0.0,
      "r_reasoning": 0.8,
      "r_efficiency": 0.1,
      "r_penalty": 0.0
    },
    "episode_done": false,
    "difficulty": 0,
    "done": false,
    "reward": 0.26
  }
}
```

**Fields:**
- `last_tool_output`: Result of the tool call
- `action_history`: List of past actions (compact format)
- `step_reward`: Total reward for this step
- `step_reward_breakdown`: Components:
  - `r_outcome`: ±1.0 for fixing/breaking, 0.0 for investigation
  - `r_reasoning`: -1 to +1, judge score
  - `r_efficiency`: -0.3 to +0.1, progress toward diagnosis
  - `r_penalty`: -0.5 for apply_fix without lineage, etc.
  - `terminal`: +3 for resolving, -2 for timeout (only at episode end)
- `episode_done`: True when fault resolved or max steps (20) reached

---

## GET /state

**Get hidden environment state (not visible to agent).**

### Request

```bash
curl -X GET http://localhost:8000/state
```

### Response (200 OK)

```json
{
  "episode_id": "4ee17392-9d1b-4ce9-8416-cb2ded742be6",
  "step": 1,
  "max_steps": 20,
  "cumulative_reward": 0.26,
  "difficulty": 0,
  "fault_type": "type_mismatch",
  "resolved": false
}
```

**Fields:**
- `fault_type`: The actual fault (hidden from agent during episode)
- `resolved`: Whether fault has been fixed
- `cumulative_reward`: Total reward so far this episode
- Other fields echo the current state

---

## GET /schema

**Get JSON schemas for all types.**

### Request

```bash
curl -X GET http://localhost:8000/schema
```

### Response (200 OK)

```json
{
  "action": {
    "type": "object",
    "properties": {
      "tool_name": {
        "enum": [
          "run_query",
          "inspect_schema",
          "check_row_counts",
          "trace_lineage",
          "sample_rows",
          "apply_fix",
          "verify_output"
        ]
      },
      "tool_args": {"type": "object"},
      "reasoning": {"type": "string"}
    },
    "required": ["tool_name", "reasoning"]
  },
  "observation": { ... },
  "state": { ... }
}
```

---

## POST /configure

**Set the LLM provider for the judge.**

### Request

```bash
curl -X POST http://localhost:8000/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "api_key": "sk-xxx..."
  }'
```

**Fields:**
- `provider`: One of: `anthropic`, `openai`, `groq`, `openrouter`, `ollama`
- `model`: Model name (e.g., `claude-sonnet-4-6`, `gpt-4`, `gemma4:26b`)
- `api_key`: API key (empty for local Ollama)
- `base_url`: Base URL for provider (optional, for custom endpoints)

### Response (200 OK)

```json
{
  "status": "ok",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6"
}
```

---

## Error Responses

### 400 Bad Request

Judge not configured:
```json
{
  "detail": "Judge not configured. POST /configure first."
}
```

### 422 Unprocessable Entity

Malformed action:
```json
{
  "detail": "1 validation error for ETLAction..."
}
```

### 500 Internal Server Error

Judge call failed (timeout, API error, etc.):
```json
{
  "detail": "Judge timeout after 8 seconds"
}
```

---

## Example: Full Episode Trace

### 1. Reset
```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"seed": 42}'
# Returns: observation with alert
```

### 2. Step 1: Diagnose
```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{
    "action": {
      "tool_name": "trace_lineage",
      "tool_args": {"table": "gold.kpi_daily_revenue"},
      "reasoning": "Trace KPI to identify dependencies"
    }
  }'
# Returns: tool result, reward breakdown, next observation
```

### 3. Step 2: Inspect
```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{
    "action": {
      "tool_name": "inspect_schema",
      "tool_args": {"table": "silver.orders_enriched"},
      "reasoning": "Check for schema drift or type issues"
    }
  }'
# Returns: schema info, reward breakdown
```

### 4. Step 3: Apply Fix
```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{
    "action": {
      "tool_name": "apply_fix",
      "tool_args": {
        "fix_type": "coalesce_column",
        "target": "silver.orders_enriched",
        "params": {"column": "region", "default": "UNKNOWN"}
      },
      "reasoning": "Fix NULL explosion by coalescing region with default"
    }
  }'
# Returns: terminal reward if fix worked, episode_done=true
```

### 5. Check State
```bash
curl -X GET http://localhost:8000/state
# Returns: hidden fault_type, resolved=true
```

---

## Python Example (requests library)

```python
import requests

base = "http://localhost:8000"

# Reset
obs = requests.post(f"{base}/reset", json={"seed": 42}).json()["observation"]
print(f"Alert: {obs['alert']}")

# Step
action = {
  "action": {
    "tool_name": "trace_lineage",
    "tool_args": {"table": "gold.kpi_daily_revenue"},
    "reasoning": "Diagnose issue"
  }
}
resp = requests.post(f"{base}/step", json=action).json()
obs = resp["observation"]
print(f"Reward: {obs['step_reward']}")
print(f"Breakdown: {obs['step_reward_breakdown']}")

# State
state = requests.get(f"{base}/state").json()
print(f"Fault: {state['fault_type']}, Resolved: {state['resolved']}")
```

