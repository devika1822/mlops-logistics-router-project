from pathlib import Path

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "conditions"
    / "train"
)

CURRENT_DATA_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "conditions"
    / "prediction_logs.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "conditions"
    / "reports"
    / "data_drift_report.html"
)


FEATURES = [
    "weather_impact_index",
    "average_speed_kmph",
    "time_of_day",
    "traffic_density_index",
]


reference_df = pd.read_parquet(
    REFERENCE_DATA_PATH
)

current_df = pd.read_csv(
    CURRENT_DATA_PATH
)

reference_df = reference_df[FEATURES].copy()
current_df = current_df[FEATURES].copy()

print("Reference dataset loaded.")
print(f"Reference rows: {len(reference_df)}")

print("\nCurrent prediction data loaded.")
print(f"Current rows: {len(current_df)}")

report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

snapshot = report.run(
    reference_data=reference_df,
    current_data=current_df,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

snapshot.save_html(
    str(REPORT_PATH)
)

print("\nData drift report generated successfully.")
print(f"Report saved to: {REPORT_PATH}")