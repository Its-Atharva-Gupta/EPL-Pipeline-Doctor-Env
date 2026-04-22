#!/usr/bin/env python3
"""Verify dense per-step reward signals are computing correctly."""

from server.etl_pipeline_doctor_environment import EtlPipelineDoctorEnvironment
from models import ETLAction

env = EtlPipelineDoctorEnvironment()

# Test each fault type
fault_types = [
    "null_explosion",
    "stale_partition",
    "type_mismatch",
    "fanout_join",
    "schema_drift",
]

for fault_type in fault_types:
    print(f"\n{'='*60}")
    print(f"Testing {fault_type}")
    print(f"{'='*60}")

    obs = env.reset_with_fault(fault_type)
    print(f"Reset OK. Alert: {obs.alert[:60]}")

    # Initial reward breakdown should have dense signals
    print(f"\nStep 0 (reset) reward breakdown:")
    for k, v in obs.step_reward_breakdown.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    # Take a single diagnostic action
    if fault_type == "null_explosion":
        action = ETLAction(command="SELECT COUNT(*) FROM silver_orders_enriched WHERE region IS NULL")
    elif fault_type == "stale_partition":
        action = ETLAction(command="CHECK ROWS silver_orders_enriched")
    elif fault_type == "type_mismatch":
        action = ETLAction(command="SELECT COUNT(*) FROM bronze_orders_raw")
    elif fault_type == "fanout_join":
        action = ETLAction(command="SELECT COUNT(*) FROM bronze_products_raw")
    elif fault_type == "schema_drift":
        action = ETLAction(command="INSPECT TABLE bronze_orders_raw")

    obs = env.step(action)
    print(f"\nStep 1 (diagnostic action) reward breakdown:")
    for k, v in obs.step_reward_breakdown.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    # Check that dense signals are present
    has_kpi_proximity = "kpi_proximity" in obs.step_reward_breakdown
    has_fault_progress = "fault_progress" in obs.step_reward_breakdown
    has_kpi_delta = "kpi_delta" in obs.step_reward_breakdown
    has_fault_delta = "fault_delta" in obs.step_reward_breakdown
    has_r_progress = "r_progress" in obs.step_reward_breakdown

    checks = {
        "kpi_proximity": has_kpi_proximity,
        "fault_progress": has_fault_progress,
        "kpi_delta": has_kpi_delta,
        "fault_delta": has_fault_delta,
        "r_progress": has_r_progress,
    }

    print(f"\nDense signal presence:")
    for signal, present in checks.items():
        status = "✓" if present else "✗"
        print(f"  {status} {signal}")

    if all(checks.values()):
        print(f"\n✓ {fault_type} PASSED")
    else:
        print(f"\n✗ {fault_type} FAILED")

print(f"\n{'='*60}")
print("Verification complete!")
print(f"{'='*60}\n")
