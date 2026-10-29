"""Day 7: manually check MRI data and saved models, then write a report."""

from datetime import datetime, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# Docker Compose mounts the project folders under /app in Airflow.
PROJECT_ROOT = Path("/app")
DATA_FOLDERS = [
    "data/raw/Train",
    "data/raw/Val",
    "data/processed",
    "data/yolo/images/train",
    "data/yolo/images/val",
    "data/yolo/labels/train",
    "data/yolo/labels/val",
]
CSV_FILES = [
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
]
MODEL_FILES = [
    "models/finetuned_mobilenetv2.keras",
    "models/finetuned_resnet50v2.keras",
    "models/yolo_runs/yolov8n_mri/weights/best.pt",
]


def check_data():
    """Fail with a helpful message if an expected folder or CSV is missing."""
    for folder in DATA_FOLDERS:
        if not (PROJECT_ROOT / folder).is_dir():
            raise FileNotFoundError(f"Missing MRI folder: {folder}")
        print(f"Found folder: {folder}")

    for filename in CSV_FILES:
        if not (PROJECT_ROOT / filename).is_file():
            raise FileNotFoundError(f"Missing CSV file: {filename}")
        print(f"Found CSV: {filename}")


def check_models():
    """Check saved model files without loading or training any model."""
    for filename in MODEL_FILES:
        if not (PROJECT_ROOT / filename).is_file():
            raise FileNotFoundError(f"Missing saved model: {filename}")
        print(f"Found model: {filename}")


def generate_report():
    """Write this report only after both check tasks have succeeded."""
    report = PROJECT_ROOT / "results/airflow_report.txt"
    lines = [
        "Brain MRI Tumor Detection - Airflow Report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "DAG: mri_pipeline",
        "check_data: SUCCESS",
        "check_models: SUCCESS",
        "All required folders and files exist.",
        "This is a file existence check, not a model accuracy evaluation.",
        "",
        "Checked paths:",
    ]
    lines.extend(DATA_FOLDERS + CSV_FILES + MODEL_FILES)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {report}")


with DAG(
    dag_id="mri_pipeline",
    description="Check MRI files and saved models, then create a simple report.",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,  # Run only when manually triggered.
    catchup=False,
    max_active_runs=1,
    tags=["mri", "day7"],
) as dag:
    data_task = PythonOperator(task_id="check_data", python_callable=check_data)
    models_task = PythonOperator(task_id="check_models", python_callable=check_models)
    report_task = PythonOperator(task_id="generate_report", python_callable=generate_report)

    data_task >> models_task >> report_task
