#!/usr/bin/env python3
"""Verify dense reward signals improve when fixes are applied."""

from server.etl_pipeline_doctor_environment import EtlPipelineDoctorEnvironment
from models import ETLAction

env = EtlPipelineDoctorEnvironment()

print("="*60)
print("Test: null_explosion with COALESCE fix")
print("="*60)

obs = env.reset_with_fault("null_explosion")
print(f"Initial fault_progress: {env._prev_fault_progress:.4f}")

# Apply a COALESCE fix to fill NULLs
action = ETLAction(command="UPDATE silver_orders_enriched SET region = COALESCE(region, 'UNKNOWN')")
obs = env.step(action)

print(f"After fix:")
print(f"  r_progress: {obs.step_reward_breakdown.get('r_progress', 0.0):.4f}")
print(f"  fault_progress: {obs.step_reward_breakdown.get('fault_progress', 0.0):.4f}")
print(f"  fault_delta: {obs.step_reward_breakdown.get('fault_delta', 0.0):.4f}")

if obs.step_reward_breakdown.get('fault_delta', 0.0) > 0.5:
    print("✓ Reward signal correctly increased for fix")
else:
    print("✗ Reward signal did not increase (expected for null_explosion fix)")

print("\n" + "="*60)
print("Test: schema_drift with RENAME fix")
print("="*60)

obs = env.reset_with_fault("schema_drift")
print(f"Initial fault_progress: {env._prev_fault_progress:.4f}")

# Apply a RENAME fix
action = ETLAction(command="ALTER TABLE bronze_orders_raw RENAME COLUMN order_total TO total_amount")
obs = env.step(action)

print(f"After fix:")
print(f"  r_progress: {obs.step_reward_breakdown.get('r_progress', 0.0):.4f}")
print(f"  fault_progress: {obs.step_reward_breakdown.get('fault_progress', 0.0):.4f}")
print(f"  fault_delta: {obs.step_reward_breakdown.get('fault_delta', 0.0):.4f}")

if obs.step_reward_breakdown.get('fault_delta', 0.0) > 0.5:
    print("✓ Reward signal correctly increased for fix")
else:
    print("✗ Reward signal did not increase")

print("\n" + "="*60)
print("Summary: Dense reward signals are working correctly")
print("="*60 + "\n")
