# Brain MRI Tumor Classification & Detection Pipeline

> **Educational AI/ML project — not a medical diagnostic tool.**
>
> Four-class MRI image classification, annotated-region detection and workflow automation.

## 1. Project overview

This project explores brain MRI images and builds two types of computer-vision models:

- **Classification:** predict one category per image — **Glioma**, **Meningioma**, **No Tumor**, or **Pituitary** — using **MobileNetV2** and **ResNet50V2** (TensorFlow/Keras).
- **Detection:** use **YOLOv8n** (Ultralytics/PyTorch) to locate *annotated regions* with bounding boxes. A proposed **three-class detector** uses Glioma, Meningioma and Pituitary; the annotation semantics must be verified before trusting this mapping.
- **Orchestration:** use **Apache Airflow** to check data and models and generate a simple report. **Jupyter** supports the experiments, and **Docker Compose** runs the development services.

The work is organised into a **10-day learning roadmap**. This README combines the supplied Git log with the notebook/workflow plan. **Only Days 1–2 and the beginning of Day 3 are directly evidenced by the uploaded Git history; later implementations and performance results need verification in the actual repository.**

## 2. Dataset

The dataset explored during the project contains **5,249 MRI images**:

| Original subset | Images |
| --- | ---: |
| `Train/` | 4,737 |
| `Val/` | 512 |
| **Total** | **5,249** |

Classification classes: `Glioma`, `Meningioma`, `No Tumor`, `Pituitary`.

Expected original layout (some datasets may be stored outside Git because of size):

```text
data/raw/
├── Train/
│   ├── Glioma/{images/,labels/}
│   ├── Meningioma/{images/,labels/}
│   ├── No Tumor/{images/,labels/}
│   └── Pituitary/{images/,labels/}
└── Val/
    └── (same four class folders)
```

**Linux is case-sensitive:** the folder is `Train` (capital T), not `train`.

Day 2 uses CSV splits under `data/processed/`:

| File | Intended usage | Example count from exploration |
| --- | --- | ---: |
| `train.csv` | Model training | 4,263 |
| `val.csv` | Model comparison / early stopping | 512 |
| `test.csv` | Held-out final classification assessment | 474 |

The CSV counts depend on regenerating the original split. **Do not choose hyperparameters using the test set.** When possible, split by **patient**, not only by image, to avoid data leakage.

### Important annotation checks

Day 1 revealed issues that must be reviewed before interpreting detection performance:

- In `Val/No Tumor`, two images have no matching TXT annotation and two TXT files have no matching image.
- `No Tumor` training images contain **nonempty** annotations with class ID `2`; these are **not automatically valid tumor boxes**.
- Tumor folder labels contain several YOLO class IDs; the ID meaning is not guaranteed by the folder name.
- Some images have **multiple or overlapping boxes**. These are not necessarily duplicates.
- A small coordinate-boundary discrepancy was caused by rounding; do not discard annotations without inspection.

Keep `data/raw/` unchanged. Review and document the labels with the dataset source/professor before preparing a detection-specific copy in `data/yolo/`.

## 3. Technologies

| Tool | Purpose |
| --- | --- |
| Python, Pandas, NumPy, Pillow | Data checks and preprocessing |
| TensorFlow / Keras | MobileNetV2 and ResNet50V2 classification |
| Ultralytics / PyTorch | YOLOv8n detection |
| scikit-learn, Matplotlib, Seaborn | Metrics and visualisation |
| JupyterLab | Experiments and notebooks |
| NVIDIA GPU / CUDA | Accelerated model operations (when supported) |
| Docker Compose / Ubuntu WSL2 | Reproducible local environment |
| Apache Airflow | Workflow checks and reporting |
| Streamlit | Planned simple prediction interface |

## 4. Project structure

The following combines **used** and **planned** paths. Check the actual repository before treating every file as available.

```text
mri-tumor-detection-pipeline/
├── dags/
│   └── mri_pipeline.py                  # Day 7, verify implementation
├── data/
│   ├── raw/                             # Original images and TXT labels
│   ├── processed/                       # train.csv, val.csv, test.csv
│   └── yolo/                            # Detection copy + data.yaml (if generated)
├── models/                              # Saved .keras and .pt models (if trained)
├── notebooks/
│   ├── D1_Exploration.ipynb
│   ├── D2_Preprocessing.ipynb
│   ├── D3_Transfer_Learning.ipynb
│   ├── D4_Fine_Tuning.ipynb             # Verify name/existence
│   ├── D5_YOLO_Detection.ipynb          # Verify name/existence
│   └── D6_Model_Evaluation.ipynb        # Verify name/existence
├── results/                             # Evaluation files and Airflow report
├── docker-compose.yml                   # Verify configured services
├── Dockerfile
├── requirements.txt
└── README.md
```

## 5. Quick start (Windows + Ubuntu WSL2 + Docker Desktop)

**Prerequisites:** Docker Desktop with WSL integration for Ubuntu, and a local copy of this repository with the MRI dataset in its expected location. GPU support is optional for dataset inspection; training requires a compatible environment.

Open Ubuntu:

```bash
wsl -d Ubuntu  # run this in PowerShell
```

Inside Ubuntu:

```bash
cd ~/mri-tumor-detection-pipeline
docker compose config --services
docker compose up -d jupyter
```

Open Jupyter at **http://localhost:8888** if the Jupyter service is present and configured for that port.

Test GPU access in Docker (requires NVIDIA Windows driver + WSL2 GPU support):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Test TensorFlow from a Jupyter cell:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

For YOLO, test PyTorch separately:

```python
import torch
print(torch.cuda.is_available())
```

A working NVIDIA container does **not** automatically prove that both TensorFlow and PyTorch can use CUDA inside Jupyter.

### Airflow (if the `airflow` Compose service has been configured)

```bash
docker compose up -d airflow
docker compose logs --tail=50 airflow
```

Open **http://localhost:8080**, sign in with the local Airflow administrator account, find `mri_pipeline`, and trigger it manually. The example DAG checks input files and saved models, then writes `results/airflow_report.txt` — **it does not retrain models or measure accuracy**.

**Do not commit credentials**, Airflow's local database, raw medical images or large model weights without confirming licensing and data permissions.

## 6. Experimental workflow

| Day | Focus | Evidence / next check |
| --- | --- | --- |
| 1 | Explore dataset and annotations | **Git history:** exploration checks, class counts, boxes and label review |
| 2 | Preprocessing and CSV splits | **Git history:** dataset tables, splits, loader, augmentation |
| 3 | MobileNetV2 / ResNet50V2 | **Git history:** D3 notebook setup begins; training/checkpoints need verification |
| 4 | Fine-tune both classifiers | Discussed in project work; verify `.keras` files and actual scores |
| 5 | Prepare labels and train YOLOv8n | Work discussed; annotation decisions and complete run need verification |
| 6 | Classification reports and YOLO mAP | Work discussed; YOLO validation encountered memory/kernel issues |
| 7 | Airflow DAG | Setup discussed; verify DAG import, successful run and TXT report |
| 8 | Streamlit UI | Planned |
| 9 | Docker integration and smoke tests | Planned finalisation |
| 10 | Final test, documentation and demo | Planned |

### Model outputs (when training has completed)

```text
models/best_mobilenetv2.keras
models/best_resnet50v2.keras
models/finetuned_mobilenetv2.keras
models/finetuned_resnet50v2.keras
models/yolo_runs/yolov8n_mri/weights/best.pt
```

**These are expected paths, not a claim that all weights exist.** Use the local file system and notebook outputs to confirm them.

## 7. Evaluation

- **Classification:** accuracy, per-class precision, recall, F1-score and confusion matrices, on the same validation split.
- **Detection:** per-class precision/recall, `mAP50` and `mAP50-95`, with visual inspection of predicted boxes.
- **Final test:** evaluate the selected classifier on held-out `test.csv` only after model selection.

Do **not** compare classification accuracy directly to YOLO mAP; they measure different tasks.

No final accuracy/mAP numbers are included here because the supplied Git log does not contain verified evaluation results.

### Troubleshooting already encountered

| Symptom | What to check |
| --- | --- |
| `No such file or directory` when loading images | `Train` vs `train` and notebook working directory; verify all CSV paths |
| `libxcb.so.1` missing | OpenCV/system libraries inside Jupyter Docker image |
| `No space left on device` for `/dev/shm` | Try Ultralytics `workers=0`, smaller batch; consider Compose `shm_size` |
| Jupyter kernel dies during YOLO validation | Run YOLO in a fresh notebook without TensorFlow, then inspect container logs/memory |
| `docker` not found in Ubuntu | Docker Desktop → Resources → WSL Integration → enable Ubuntu |

## 8. Limitations and ethics

This project is for **learning and demonstration only**. MRI model predictions are **not a medical diagnosis**, and model confidence is not a guarantee of correctness. Results may be affected by annotation errors, class imbalance, patient-level data leakage and differences in MRI acquisition. Verify data usage rights and avoid publishing identifiable medical data.

## 9. Project tracking

See `JIRA_PLANIFICATION.md` for a detailed ten-day Jira backlog with epics, task descriptions, dependencies and acceptance criteria. A `JIRA_IMPORT.csv` is also available with 10 epics, 52 tasks and parent references for optional Jira CSV import (mapping/permissions depend on the Jira setup).

## 10. Git history provenance

Prepared from an uploaded **Git `HEAD` reflog (43 entries, 2026-10-20 to 2026-10-23 UTC)**. It contains *commit descriptions*, not source file diffs or proof that notebooks ran successfully. Day 1 and Day 2 steps are supported by that log. Day 3 is only partially covered. Later milestones are based on the project discussion and are explicitly marked for verification or planned work.

**Maintainer:** Project repository owner. **License and dataset source:** To be documented before redistribution.
