"""
Cargo Payload Track – Evidently AI Monitoring

What it checks:
  1. Weight Ceiling Alert  – fires if any live cargo_weight_kg
     exceeds the maximum seen during training (model is extrapolating).
  2. Data Drift Report     – uses Evidently's DataDriftPreset to
     detect distribution shift across all monitored columns.

Output: HTML drift report saved to reports/payload_drift_<timestamp>.html

Usage:
    python monitoring/payload_monitor.py
    python monitoring/payload_monitor.py --ref data/processed/payload_clean.parquet \
                                          --cur data/processed/payload_live.parquet
"""

import argparse
import json
import logging
import os
from datetime import datetime

import pathlib
import pandas as pd
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift
from evidently import Report

# Always write reports next to this script regardless of cwd
REPORT_DIR = str(pathlib.Path(__file__).resolve().parent / "reports")
MONITORED_COLS = [
    "cargo_weight_kg",
    "vehicle_capacity_pct",
    "delivery_deadline_hrs",
    "trip_distance_km",
]

logging.basicConfig(level=logging.INFO, format="[MONITOR] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_parquet(path: str, cols: list) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")
    return df[cols].dropna()


# ---------------------------------------------------------------------------
# Alert 1: Cargo weight ceiling
# ---------------------------------------------------------------------------
def check_weight_ceiling(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Alert if any live weight exceeds the training-set maximum."""
    train_max = reference["cargo_weight_kg"].max()
    live_max  = current["cargo_weight_kg"].max()
    exceeded  = bool(live_max > train_max)

    result = {
        "alert":        exceeded,
        "train_max_kg": round(float(train_max), 2),
        "live_max_kg":  round(float(live_max),  2),
    }
    if exceeded:
        result["message"] = (
            f"ALERT: Live max weight ({live_max:.2f} kg) exceeds training "
            f"maximum ({train_max:.2f} kg). Model predictions may be unreliable."
        )
        log.warning(result["message"])
    else:
        result["message"] = (
            f"OK: Cargo weights within training range (max {train_max:.2f} kg)."
        )
        log.info(result["message"])

    return result


# ---------------------------------------------------------------------------
# Alert 2: Evidently drift report
# ---------------------------------------------------------------------------
def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = Report(metrics=[
        DataDriftPreset(),
        ValueDrift(column="cargo_weight_kg"),
    ])
    snapshot = report.run(reference_data=reference, current_data=current)

    os.makedirs(REPORT_DIR, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORT_DIR, f"payload_drift_{ts}.html")
    snapshot.save_html(html_path)
    log.info(f"Report saved → {html_path}")

    raw          = snapshot.dict()
    metrics      = raw.get("metrics", [])
    drift_found  = False
    weight_drift = False
    for m in metrics:
        result = m.get("result", {})
        if "dataset_drift" in result:
            drift_found = result["dataset_drift"]
        if "drift_detected" in result:
            weight_drift = result["drift_detected"]

    return {
        "dataset_drift":      drift_found,
        "cargo_weight_drift": weight_drift,
        "report_path":        html_path,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(ref_path: str, cur_path: str) -> dict:
    reference = load_parquet(ref_path, MONITORED_COLS)
    current   = load_parquet(cur_path, MONITORED_COLS)

    ceiling = check_weight_ceiling(reference, current)
    drift   = run_drift_report(reference, current)

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "weight_ceiling":     ceiling,
        "drift_report":       drift,
        "overall_alert":      ceiling["alert"] or drift["dataset_drift"],
    }

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    if summary["overall_alert"]:
        log.warning("MONITORING ALERT – Cargo payload distribution has shifted!")
    else:
        log.info("Monitoring check passed. No anomalies detected.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Payload track drift monitor")
    parser.add_argument(
        "--ref", default="data/processed/payload_clean.parquet",
        help="Reference (training) dataset path",
    )
    parser.add_argument(
        "--cur", default="data/processed/payload_live.parquet",
        help="Current (live inference) dataset path",
    )
    args = parser.parse_args()
    run(args.ref, args.cur)
