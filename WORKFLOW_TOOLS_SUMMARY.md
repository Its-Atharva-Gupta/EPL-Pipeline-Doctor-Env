# Workflow Tracing Tools — Summary

Three new files to visualize and understand the complete ETL Pipeline Doctor workflow.

---

## Files Created

### 1. `workflow_trace.py` — Single Episode Trace
Detailed HTTP request/response logging for one episode.

**Run:**
```bash
python3 workflow_trace.py
```

**Shows:**
- Request payloads sent to `/reset` and `/step`
- Full observation objects returned
- Hidden state (fault type, difficulty)
- **Reward breakdown per step** ← KEY
  - r_outcome (did it fix the fault?)
  - r_reasoning (is the step logical?)
  - r_efficiency (is it making progress?)
  - r_penalty (violations)
  - **TOTAL step reward**

**Best for:**
- Understanding the complete request/response cycle
- Debugging a single episode
- Seeing how rewards are computed step-by-step

**Output: ~200 lines of detailed workflow**

---

### 2. `train_traced.py` — Multi-Episode Training Trace
Training loop instrumentation showing curriculum learning.

**Run:**
```bash
# Quick test: 3 episodes
python3 train_traced.py --episodes 3 --dry-run

# Full: 10 episodes with server already running
python3 train_traced.py --episodes 10 --max-steps 20
```

**Shows:**
- Curriculum state for each episode (fault type, difficulty, EMA mastery)
- Per-step actions and rewards
- Episode summaries (resolved/failed, cumulative reward)
- **Training summary** ← KEY
  - Resolution rate across episodes
  - Per-fault-type performance
  - Average reward trends

**Best for:**
- Watching agent learn over multiple episodes
- Understanding curriculum adaptation
- Analyzing training progression

**Output: ~500-1000 lines depending on --episodes**

---

### 3. `WORKFLOW_TRACING.md` — Documentation
Complete guide explaining what each tracer shows and how to interpret it.

**Covers:**
- How to use workflow_trace.py and train_traced.py
- Configuration (with judge, without judge, etc.)
- What each reward component means
- Debugging checklist
- Curriculum state progression

---

### 4. `HTTP_API_REFERENCE.md` — API Documentation
Complete HTTP endpoint reference with request/response examples.

**Covers:**
- `/reset` endpoint (start episode)
- `/step` endpoint (execute action, get reward)
- `/state` endpoint (hidden fault, resolved status)
- `/schema` endpoint (JSON schemas)
- `/configure` endpoint (set judge provider)
- Tool arguments for all 7 tools
- Curl examples for all endpoints
- Python requests examples

---

## What You Can See Now

### Before (train.py)
```
Episode 1: Action → Step → ?
Episode 2: Action → Step → ?
...
Result: Model accuracy
```

### After (workflow_trace.py + train_traced.py)
```
Episode 1:
  Reset: fault=schema_drift, difficulty=0
  
  Step 1:
    Action: trace_lineage("gold.kpi_daily_revenue")
    Reward breakdown:
      r_outcome = 0.0 (investigating, not fixing yet)
      r_reasoning = 0.8 (good diagnostic step)
      r_efficiency = 0.1 (step made progress)
      r_penalty = 0.0 (no violations)
      TOTAL = 0.26
    Cumulative: 0.26
  
  Step 2:
    Action: inspect_schema("silver.orders_enriched")
    Reward breakdown:
      r_outcome = 0.0
      r_reasoning = 0.7
      r_efficiency = 0.1
      r_penalty = 0.0
      TOTAL = 0.18
    Cumulative: 0.44
  
  ... (more steps)
  
  Episode summary:
    Steps: 6
    Resolved: YES (schema fixed)
    Cumulative reward: +3.85
    Rewards per step: [+0.26, +0.18, +0.12, -0.05, +3.00, +0.45]

Episode 2:
  (different fault type from curriculum)
  ...
```

---

## Quick Start

### Just See What's Happening
```bash
# Terminal 1: Start the server
python3 -c "from server.app import app; import uvicorn; uvicorn.run(app, port=8000)"

# Terminal 2: Run workflow trace
python3 workflow_trace.py
```

Output shows:
- What actions are sent
- What rewards are computed
- How curriculum picks faults

### Watch Agent Learn Over Multiple Episodes
```bash
python3 train_traced.py --episodes 5 --dry-run
```

Output shows:
- Curriculum difficulty progression
- Reward trends over episodes
- Resolution rate improvement

### Deep Dive into Specific Episode
```bash
# Modify workflow_trace.py to use specific seed
# Or check train_traced.py EPISODE SUMMARY sections
```

---

## Reward Breakdown: What It Means

From any step response:
```json
"step_reward_breakdown": {
  "r_outcome": 0.0,      # -1=broke it, 0=investigating, +1=fixed it
  "r_reasoning": 0.8,    # -1=irrational, 0=ok, +1=textbook diagnosis
  "r_efficiency": 0.1,   # -0.1=repeated, +0.1=progress, -0.2=wrong fix type
  "r_penalty": -0.5      # -0.5=no lineage, -0.3=malformed args, etc.
}
```

**Total = 0.5 × r_outcome + 0.3 × r_reasoning + 0.2 × r_efficiency + r_penalty**

Examples:

**Good investigative step:**
```
r_outcome: 0.0 (just exploring)
r_reasoning: 0.9 (smart question)
r_efficiency: 0.1 (discovered new info)
r_penalty: 0.0 (no violations)
TOTAL: 0.5×0 + 0.3×0.9 + 0.2×0.1 + 0 = 0.27
```

**Correct fix:**
```
r_outcome: 1.0 (RESOLVED!)
r_reasoning: 0.8 (good reasoning)
r_efficiency: 0.1 (efficient)
r_penalty: 0.0 (none)
TOTAL: 0.5×1.0 + 0.3×0.8 + 0.2×0.1 + 0 = 0.64
```

**Apply fix without diagnosis:**
```
r_outcome: -1.0 (wrong fix type)
r_reasoning: -0.5 (didn't diagnose first)
r_efficiency: -0.2 (wrong fix type penalty)
r_penalty: -0.5 (no trace_lineage called)
TOTAL: 0.5×(-1) + 0.3×(-0.5) + 0.2×(-0.2) + (-0.5) = -1.11
```

---

## Curriculum Progression Example

Over 20 episodes:

**Episodes 1-3:** All "warmup" tier, low difficulty
```
Ep 1: fault=schema_drift, tier=warmup, ema=0.5
Ep 2: fault=stale_partition, tier=warmup, ema=0.5
Ep 3: fault=null_explosion, tier=warmup, ema=0.5
```

**Episodes 4-10:** Agent learns schema_drift, gets promoted
```
Ep 4: schema_drift resolved ✓ → ema: 0.5 → 0.6
Ep 5: stale_partition failed ✗ → ema: 0.5 → 0.4
Ep 6: schema_drift resolved ✓ → ema: 0.6 → 0.68
Ep 7: schema_drift resolved ✓ → ema: 0.68 → 0.74 (PROMOTE!)
Ep 8: schema_drift tier=beginner (harder) → success
```

**Episodes 10-20:** Curriculum focuses on agent's weak spots
```
Ep 10-15: null_explosion success rate low → promoted by designer
Ep 15-20: Adversarial designer generates cascades of null_explosion + schema_drift
```

---

## Files Side-by-Side

| Aspect | workflow_trace.py | train_traced.py | HTTP_API_REFERENCE.md |
|--------|---|---|---|
| Episodes | 2 fixed | N configurable | — |
| Depth | Very deep (all details) | Medium (per-episode) | Reference only |
| Duration | ~30 sec | ~5 min for 10 eps | — |
| Shows reward? | ✓ Full breakdown | ✓ Full breakdown | ✓ Format only |
| Shows curriculum? | ✓ Hidden state | ✓ Progression | — |
| Best for | Understanding workflow | Watching learning | Building custom tools |

---

## Next Steps

### 1. Run and Understand
```bash
python3 workflow_trace.py  # See one episode in detail
```

Then read `WORKFLOW_TRACING.md` to understand every field.

### 2. Watch Learning Progress
```bash
python3 train_traced.py --episodes 10 --dry-run
```

Then read the episode summaries to see curriculum adaptation.

### 3. Build Custom Tools
Use `HTTP_API_REFERENCE.md` to write your own client or analysis script.

### 4. Debug Issues
- If reward seems wrong: check `step_reward_breakdown` in workflow_trace
- If curriculum stuck: check mastery EMA progression in train_traced
- If API call fails: check `HTTP_API_REFERENCE.md` error section

---

## No Changes to train.py

⚠️ **Important:** These tracing tools are **separate from train.py**.
- train.py remains unchanged
- These tools are for visibility and debugging only
- You can run train.py as normal: `python3 train.py --num-generations 4 --max-steps 200`

Use the tracers alongside train.py to understand what's happening under the hood.

