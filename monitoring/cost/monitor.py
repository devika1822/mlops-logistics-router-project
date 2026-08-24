from pathlib import Path

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_cost_da25g503.csv"
)

CURRENT_DATA_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "cost"
    / "prediction_logs.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "cost"
    / "reports"
    / "data_drift_report.html"
)


reference_df = pd.read_csv(
    REFERENCE_DATA_PATH
)

current_df = pd.read_csv(
    CURRENT_DATA_PATH
)


FEATURES = list(current_df.columns)

FEATURES = [
    column
    for column in FEATURES
    if column
    not in {
        "predicted_optimized_route_cost",
        "prediction_timestamp",
    }
]


reference_df = reference_df[FEATURES].copy()
current_df = current_df[FEATURES].copy()


print("Reference dataset loaded.")
print(f"Reference rows: {len(reference_df)}")

print("\nCurrent prediction data loaded.")
print(f"Current rows: {len(current_df)}")

print("\nMonitoring features:")
for feature in FEATURES:
    print(f"  - {feature}")


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