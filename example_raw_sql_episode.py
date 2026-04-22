#!/usr/bin/env python3
"""
Example: ETL Pipeline Doctor with Raw SQL Commands

This shows how the agent now communicates using raw SQL-like commands
instead of structured tool_name + tool_args format (similar to Kube's kubectl).

Commands supported:
  - SELECT ... — run SQL queries (read-only)
  - INSPECT TABLE <table> — get table schema + null counts
  - CHECK <table> / CHECK ROWS <table> — row counts by date
  - TRACE <table> / TRACE LINEAGE <table> — upstream/downstream dependencies
  - SAMPLE <table> [n] — sample n random rows
  - UPDATE ... / INSERT ... — mutate data
  - VERIFY — check if KPI now matches ground truth
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import ETLPipelineDoctorEnv
from models import ETLAction


def main():
    # Connect to environment
    env = ETLPipelineDoctorEnv(base_url="http://localhost:8000")

    # Reset episode
    obs = env.reset()
    print(f"Episode started: {obs.alert}\n")
    print(f"Available tables: {', '.join(obs.available_tables[:3])}...")
    print(f"Available KPIs: {', '.join(obs.available_kpis)}\n")

    # Step 1: Trace lineage to understand data dependencies
    print("=" * 60)
    print("STEP 1: Trace lineage of broken KPI")
    print("=" * 60)
    action = ETLAction(command="TRACE LINEAGE gold.kpi_daily_revenue")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 2: Inspect upstream table
    print("=" * 60)
    print("STEP 2: Inspect the silver.daily_sales table")
    print("=" * 60)
    action = ETLAction(command="INSPECT TABLE silver.daily_sales")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 3: Check row counts to find stale partitions
    print("=" * 60)
    print("STEP 3: Check row counts by date")
    print("=" * 60)
    action = ETLAction(command="CHECK ROWS silver.daily_sales")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 4: Sample rows to see data quality
    print("=" * 60)
    print("STEP 4: Sample rows from bronze.orders_raw")
    print("=" * 60)
    action = ETLAction(command="SAMPLE bronze.orders_raw 5")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 5: Run a custom SELECT query
    print("=" * 60)
    print("STEP 5: Query to find NULL values in region")
    print("=" * 60)
    action = ETLAction(command="SELECT region, COUNT(*) as null_count FROM silver_orders_enriched WHERE region IS NULL GROUP BY region")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 6: Fix the data with a raw UPDATE statement
    print("=" * 60)
    print("STEP 6: Apply fix - Fill NULLs with UNKNOWN")
    print("=" * 60)
    action = ETLAction(command="UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN') WHERE region IS NULL")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}\n")

    # Step 7: Verify the fix worked
    print("=" * 60)
    print("STEP 7: Verify KPI now matches ground truth")
    print("=" * 60)
    action = ETLAction(command="VERIFY")
    obs = env.step(action)
    print(f"Command: {action.command}")
    print(f"Result:\n{obs.last_tool_output}\n")
    print(f"Reward: {obs.step_reward:.2f}")
    print(f"Episode Done: {obs.episode_done}\n")

    # Summary
    print("=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Actions taken: {obs.step}")
    print(f"Action history:\n")
    for action_summary in obs.action_history:
        print(f"  {action_summary}")
    print(f"\nDifficulty: {obs.difficulty}")

    env.close()


if __name__ == "__main__":
    main()
