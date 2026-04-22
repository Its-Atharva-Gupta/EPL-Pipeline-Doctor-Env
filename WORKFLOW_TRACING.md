
# ETL Pipeline Doctor — Workflow Tracing

Two new files for detailed debugging and understanding the complete workflow:

## Files

### 1. `workflow_trace.py`
**Purpose:** Single-episode trace showing complete workflow with HTTP request/response details.

**What it shows:**
- ✅ Reset request payload
- ✅ Observation returned (alert, available tables/KPIs)
- ✅ Curriculum state (fault type selected, difficulty)
- ✅ For each step:
  - Action sent (tool name, args, reasoning)
  - Tool result/output
  - **Reward breakdown** (r_outcome, r_reasoning, r_efficiency, r_penalty)
  - Cumulative reward so far
  - Episode status

**Usage:**
```bash
python3 workflow_trace.py
```

**Output example:**
```
================================================================================
EPISODE 1: RESET WORKFLOW
================================================================================

[REQUEST] POST /reset
Payload:
{
  "seed": 42
}

[RESPONSE] Observation returned:
  Alert: ALERT: Daily revenue KPI dropped unexpectedly...
  Available tables: ['bronze.orders_raw', 'bronze.products_raw', ...]
  Available KPIs: ['gold.kpi_daily_revenue', 'gold.kpi_category_mix']
  Step: 0
  Difficulty: 0
  Episode done: False

[SERVER STATE] Fetching internal state...
  Episode ID: 4ee17392-9d1b-4ce9-8416-cb2ded742be6
  Fault type (HIDDEN from agent): type_mismatch
  Max steps: 20
  Cumulative reward: 0.0

................................................................................

--------------------------------------------------------------------------------
STEP 1: TOOL EXECUTION
--------------------------------------------------------------------------------

[ACTION] Sending to /step
  Tool: trace_lineage
  Args: {"table": "gold.kpi_daily_revenue"}
  Reasoning: Starting diagnosis: trace the KPI upstream...

[TOOL RESULT]
  Output: Table: gold.kpi_daily_revenue
  Upstream (feeds into this table): ['silver.daily_sales']
  Downstream (this table feeds into): ['none']

[REWARD BREAKDOWN]
  r_outcome:    0.000  (not fixing, just investigating)
  r_reasoning:  0.800  (good diagnostic step)
  r_efficiency: 0.100  (step made progress)
  r_penalty:    0.000  (no penalties)
  ──────────────
  TOTAL:        0.260

[EPISODE STATUS]
  Step: 1
  Episode done: False
  Action history: 1 actions so far
  Cumulative reward: 0.000
```

---

### 2. `train_traced.py`
**Purpose:** Multi-episode training trace showing agent learning progression.

**What it shows:**
- ✅ Curriculum state for each episode (fault type, difficulty tier, mastery EMA)
- ✅ Per-step action selection and reward
- ✅ Reward breakdown for each step
- ✅ Judge scores (if configured)
- ✅ Episode resolution (success/failure)
- ✅ Summary statistics per fault type
- ✅ Training progression (cumulative reward, resolution rate)

**Usage:**
```bash
# Run 3 episodes with detailed tracing
python3 train_traced.py --episodes 3 --dry-run

# Run 10 episodes without starting server (assumes already running)
python3 train_traced.py --episodes 10 --max-steps 20
```

**Output example:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      ETL PIPELINE DOCTOR — TRAINING TRACE                    ║
║              Running 3 episodes with detailed workflow logging               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

====================================================================================================
[1/3] RESET
====================================================================================================

📍 Episode ID: abc123-def456
🔧 Fault type: schema_drift
📊 Difficulty: 0
🚨 Alert: ALERT: Daily revenue KPI dropped unexpectedly...

----------------------------------------------------------------------------------------------------
  STEP 1
----------------------------------------------------------------------------------------------------

  📤 Action:
      Tool: trace_lineage
      Args: {"table": "gold.kpi_daily_revenue"}
      Reasoning: Trace KPI upstream to understand data lineage

  📥 Tool result:
      Table: gold.kpi_daily_revenue
      Upstream (feeds into this table): ['silver.daily_sales']

  💰 Reward: +0.2600
      r_outcome                     = +0.0000
      r_reasoning                   = +0.8000
      r_efficiency                  = +0.1000
      r_penalty                     = +0.0000
      ──────────────────────────────
      Cumulative: +0.2600

[Other steps...]

====================================================================================================
[1/3] EPISODE SUMMARY
====================================================================================================
  Fault type: schema_drift
  Steps taken: 6
  Resolved: ✓ YES
  Cumulative reward: +3.8450
  Rewards per step: ['+0.26', '+0.18', '+0.12', '-0.05', '+3.00', '+0.45']

====================================================================================================
TRAINING SUMMARY
====================================================================================================

  Total episodes: 3
  Resolved: 2/3 (67%)

  Per-fault-type performance:
      schema_drift      : 1/1 resolved, avg reward = +3.85
      stale_partition   : 0/1 resolved, avg reward = +0.42
      null_explosion    : 1/1 resolved, avg reward = +2.15

  Average cumulative reward: +2.14
  Total reward across all episodes: +6.42
```

---

## Configuration

### With Judge (Full Reward Computation)

For complete reward breakdowns including judge reasoning scores:

**Option A: Use Anthropic API**
```bash
export ANTHROPIC_API_KEY=sk-xxx...
python3 workflow_trace.py
```

**Option B: Use Ollama locally**
```bash
# In another terminal
ollama run gemma4:26b

# Then
export JUDGE_BACKEND=ollama
python3 workflow_trace.py
```

**Option C: Configure via HTTP**
```bash
# First configure the judge
curl -X POST http://localhost:8000/configure \
  -H "Content-Type: application/json" \
  -d '{"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "sk-xxx"}'

python3 workflow_trace.py
```

---

## What Each Part Shows

### Reset Workflow
```
[REQUEST]     ← What payload is sent to /reset
[RESPONSE]    ← Observation: alert, available tables, KPIs
[STATE]       ← Hidden state: fault type, difficulty, curriculum
```

### Step Workflow
```
[ACTION]      ← Tool name, args, agent reasoning
[TOOL RESULT] ← Output from the tool (query result, schema, etc.)
[REWARD]      ← Breakdown of r_outcome + r_reasoning + r_efficiency + penalties
              ← Per-component scores (critical for debugging GRPO)
[STATUS]      ← Episode progress, cumulative reward, whether done
```

### Reward Components Explained

| Component | Range | Meaning |
|-----------|-------|---------|
| `r_outcome` | [-1, +1] | Did this step fix the fault? (+1) Break something? (-1) Investigate? (0) |
| `r_reasoning` | [-1, +1] | LLM judge score: is this step logical? (high) or random? (low) |
| `r_efficiency` | [-0.3, +0.1] | Did this step materially progress diagnosis (+0.1) or repeat info (-0.1)? |
| `r_penalty` | [various] | -0.5 for apply_fix without lineage, -0.3 for malformed args, etc. |
| **TOTAL** | [varies] | w_outcome·r_o + w_reasoning·r_r + w_efficiency·r_e + penalty |

---

## Curriculum State During Training

Each episode, the curriculum controller:

1. **Picks a fault type** — biased toward un-mastered faults
   - `weights = 1 - ema`  (focus on weak spots)
   
2. **Selects a difficulty tier** — based on mastery of that fault type
   - Warmup → Beginner → Intermediate → Advanced → Expert
   
3. **Returns a fault spec** — injected into the warehouse

At **episode end**, `record_outcome()` updates the curriculum:
```
new_ema = (1 - alpha) * old_ema + alpha * success
alpha = 0.2  (exponential moving average, window ≈ 10 episodes)

if new_ema ≥ 0.7 (PROMOTION_THRESHOLD): promote to next tier
if new_ema ≤ 0.3 (DEMOTION_THRESHOLD): demote to previous tier
```

**In the traces**, you'll see:
- Episode 1-5: All faults start at "warmup" tier, low difficulty
- Episode 6-20: Successful fault types promote to "beginner", difficulty increases
- Episode 20+: Failed fault types demote or get targeted by adversarial designer

---

## Debugging Checklist

**Reset always works?** ✓ Check logs
- Curriculum should pick fault type
- Warehouse should seed data
- Fault should be injected

**Step works but reward is weird?** Check breakdown:
```python
breakdown = obs.get("step_reward_breakdown", {})
# If r_outcome = 0, r_reasoning = 0, r_efficiency = 0 → check if tool_result.success is False
# If r_penalty = -0.5 → agent called apply_fix without trace_lineage
# If r_penalty = -0.3 → tool_args were malformed
```

**Judge scores seem stuck at 0.0?** 
- Judge not configured (check /configure endpoint)
- Judge timeout (8s limit) — check logs for "judge timeout"
- Judge crash — check ANTHROPIC_API_KEY or Ollama status

**Curriculum not advancing?**
- Check `ema` values — should change at episode end if `record_outcome()` called
- Check `record_outcome()` is called at episode end (line 214 in `etl_pipeline_doctor_environment.py`)
- Check EMA alpha (0.2) — slow learning by design

---

## Summary: What You'll See

| File | Traces | Best For |
|------|--------|----------|
| `workflow_trace.py` | 2 episodes, ~4-6 steps each | Understanding workflow, debugging single episode |
| `train_traced.py` | N episodes (configurable), variable steps | Training progression, curriculum adaptation, reward analysis |

Both files show the **complete picture**: what's sent, what's received, what reward is computed, why.

