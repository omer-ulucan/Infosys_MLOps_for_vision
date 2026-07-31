# MLOps for Vision — Part of Infosys Topaz

## Overview

**MLOps for Vision** is a utility that is part of the Infosys Topaz suite. It is designed to accelerate end-to-end Computer Vision workflows — Training, Validation, Model Download, Feature Extraction, Augmentation, and Ground Truth Testing — across multiple annotation and training platforms including **Dataloop**, **Label Studio**, **LabelImg**, **Azure Custom Vision**, and **AI Cloud**.

The framework abstracts platform-specific details behind a single configuration-driven entry point, so the same command can train a YOLOv8 model, kick off an Azure Custom Vision run, or validate a model against a Dataloop-hosted dataset simply by swapping a config value.

## Purpose

MLOps for Vision bridges raw annotated data and production-ready computer vision models. It provides a unified pipeline framework so that data scientists, ML engineers, and Vision use-case developers can:

- Standardize model training and validation across teams and use cases.
- Reuse the same data preparation, augmentation, and reporting stack regardless of the target training platform.
- Track experiments, compare runs, and generate publishable reports (PDF, confusion matrices, similarity plots) with minimal glue code.

## Benefits

- **Platform-agnostic training**: A single interface for Azure Custom Vision, AI Cloud, and custom YOLO (v4 / v7 / v8) pipelines.
- **Pluggable data sources**: Ingest raw data from the local filesystem, AWS S3, or Dataloop without changing pipeline code.
- **Config-driven execution**: All pipeline behavior — model, data source, hyperparameters, augmentations — is controlled through JSON/YAML files in [framework_config/](framework_config/).
- **Rich validation suite**: Standard validation, IVA validation, and validation-by-embeddings pipelines with automated report generation.
- **Feature extraction**: Compute image embeddings, plot them, and generate similarity scores for dataset analysis.
- **Ground Truth reporting**: Detailed and summary GT reports including confusion matrices.
- **Patch-based inference**: Sliding-window inference for high-resolution images.
- **Experiment tracking**: First-class MLflow integration under [mlruns/](mlruns/).
- **Reproducible environment**: Docker-first setup with pinned dependencies via [requirements.txt](requirements.txt) and [constraints.txt](constraints.txt).

---

## 1. Pre-requisites

- Raw training data collected from data sources such as the internet, public datasets, or custom captures.
- Access to annotation tools (Dataloop, Label Studio, LabelImg) for data preparation.
- Working knowledge of Computer Vision concepts — models, model formats, annotation strategies.
- Working knowledge of Azure Custom Vision or AI Cloud (depending on the target platform).
- Raw data annotated in **YOLO format** and organized in a nested folder structure whose folder names encode metadata as `key=value` pairs (see [§7 Data Layout](#7-data-layout)).

## 2. Dependencies

- Docker (recommended runtime)
- Python **>= 3.9**
- Access to annotation tools (Dataloop / Label Studio / LabelImg)
- Azure Custom Vision or AI Cloud subscription (only if using those platforms)
- System connected to the Infosys VPN
- Free disk space: **10–15 GB**
- Optional: NVIDIA GPU with CUDA 11.4 for GPU training (see [Dockerfile](Dockerfile))

---

## 3. Installation Guide

### 3.1 Cloning the repository

```bash
git clone <your-internal-git-url>/visionops-codescan.git
cd visionops-codescan
```

### 3.2 Option A — Docker (recommended)

The [Dockerfile](Dockerfile) is based on `nvidia/cuda:11.4.3-devel-ubuntu20.04` and installs all system + Python dependencies, including a compiled YOLOv4 backend.

```bash
docker build -t mlops-vision .
docker run --gpus all -it --rm -v ${PWD}:/app mlops-vision bash
```

### 3.3 Option B — Local Python virtual environment

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

> For GPU support locally, install `requirements_gpu.txt` in addition and ensure a CUDA 11.4 toolchain is available.

---

## 4. Supported Pipelines

Below are the pipelines currently supported. The value in **`pipeline_type`** is what must be set in [framework_config/vision_sdk_primary_config.json](framework_config/vision_sdk_primary_config.json) to select a pipeline.

| S.No. | Use case | `pipeline_type` | Compatible `pipeline_platform` |
|-------|----------|-----------------|--------------------------------|
| 1 | Model training on Azure Custom Vision | `TrainingPipeline` | `AzureCustomVisionPipeline` |
| 2 | Model training on AI Cloud | `TrainingPipeline` | `AICloudPipeline` |
| 3 | Custom YOLOv4 / YOLOv7 / YOLOv8 training | `TrainingPipeline` | `CustomYoloPipeline` (+ `custom_pipeline_yolo`) |
| 4 | Model validation with IVA reports | `ValidationPipeline` | `IVAValidationPipeline` |
| 5 | Validation using image embeddings | `ValidationByEmbeddingsPipeline` | `IVAValidationPipeline` |
| 6 | Feature / embedding extraction | `FeatureExtractionPipeline` | `IVAValidationPipeline` |
| 7 | Dataset augmentation | `AugmentationPipeline` | `IVAValidationPipeline` |
| 8 | Ground Truth report generation | Driven via `gt_report` | `GTReport` |

Supported raw data sources (**`raw_data_source`**): `Local`, `AWSS3`, `Dataloop`.

---

## 5. Configuration Explanation

The pipeline is driven by a small set of JSON files under [framework_config/](framework_config/). The primary entry-point file is [framework_config/vision_sdk_primary_config.json](framework_config/vision_sdk_primary_config.json):

```json
{
  "raw_data_source": "Local",
  "pipeline_type": "ValidationPipeline",
  "pipeline_platform": "IVAValidationPipeline",
  "custom_pipeline_yolo": "YoloV8",
  "gt_report": "GTReport",
  "platform_config": "framework_config/iva_validation_config.json"
}
```

### 5.1 Primary config fields

| Config File | Field | Description |
|-------------|-------|-------------|
| `vision_sdk_primary_config.json` | `raw_data_source` | Source of raw data. One of `Local`, `AWSS3`, `Dataloop`. |
|  | `pipeline_type` | Which pipeline to run — see [§4 Supported Pipelines](#4-supported-pipelines). |
|  | `pipeline_platform` | Backend platform: `AzureCustomVisionPipeline`, `AICloudPipeline`, `CustomYoloPipeline`, `IVAValidationPipeline`. |
|  | `custom_pipeline_yolo` | YOLO version used when `pipeline_platform` is `CustomYoloPipeline`. One of `YoloV4`, `YoloV7`, `YoloV8`. |
|  | `gt_report` | Ground Truth report type. Default `GTReport`. |
|  | `platform_config` | Path to the JSON config for the active pipeline. Routed automatically to the loader class matching `pipeline_type`. See [framework_config/config_helper.txt](framework_config/config_helper.txt) for the value per pipeline. |

### 5.2 Per-pipeline / per-platform configs

| Config File | Purpose |
|-------------|---------|
| [general_config.json](framework_config/general_config.json) | Global logging config (`logging_type`, `logging_file`, `logging_level`, `logging_db_type`, `logging_db_engine`). |
| [local_config.json](framework_config/local_config.json) | Options when `raw_data_source = Local`. |
| [aws_s3.json](framework_config/aws_s3.json) | Credentials and bucket settings for `raw_data_source = AWSS3`. |
| [dataloop_config.json](framework_config/dataloop_config.json) | Project/dataset settings when using Dataloop. |
| [azure_custom_vision_config.json](framework_config/azure_custom_vision_config.json) | Endpoint, keys and project settings for Azure Custom Vision. |
| [ai_cloud_config.json](framework_config/ai_cloud_config.json) | AI Cloud training configuration. |
| [yolov4-custom_training_pipeline.json](framework_config/yolov4-custom_training_pipeline.json), [yolov7-custom_training_pipeline.json](framework_config/yolov7-custom_training_pipeline.json), [yolov8-custom_training_pipeline.json](framework_config/yolov8-custom_training_pipeline.json), [yolov8-patch_training_pipeline.json](framework_config/yolov8-patch_training_pipeline.json) | Per-version custom YOLO training configs. `yolov8-patch_training_pipeline.json` targets the patch/valve dataset. |
| [hyperparameters.yaml](framework_config/hyperparameters.yaml), [hyperparameters_patch.yaml](framework_config/hyperparameters_patch.yaml) | Training hyperparameters for YOLO runs. |
| [augmentations_config.json](framework_config/augmentations_config.json) | Augmentation pipeline configuration. |
| [feature_extraction.json](framework_config/feature_extraction.json) | Feature extraction pipeline configuration. |
| [validation_by_embeddings.json](framework_config/validation_by_embeddings.json) | Embeddings-based validation configuration. |
| [iva_validation_config.json](framework_config/iva_validation_config.json), [iva_validation_patch_config.json](framework_config/iva_validation_patch_config.json), [acv_validation_config.json](framework_config/acv_validation_config.json), [aws_validation_config.json](framework_config/aws_validation_config.json) | Validation platform configs. `iva_validation_patch_config.json` activates patch-based inference (its `patch_based_inference` flag toggles the code path at runtime). |
| [patch_based_config.json](framework_config/patch_based_config.json) | Patch-based inference configuration. |
| [gt_categories.json](framework_config/gt_categories.json) | Category list used for Ground Truth reports. |
| [yolov4-custom_training_pipeline.names](framework_config/yolov4-custom_training_pipeline.names), [AI Cloud Pipeline files/obj.names](AI%20Cloud%20Pipeline%20files/obj.names) | Class names files. |

See [framework_config/config_helper.txt](framework_config/config_helper.txt) for additional sample configs and inline notes.

---

## 6. Testing the Installation

To verify a clean installation, run the default pipeline (Validation → IVA) declared in [framework_config/vision_sdk_primary_config.json](framework_config/vision_sdk_primary_config.json):

```bash
python main.py
```

Expected console flow:

```
loading config ..
general_config is loaded
raw_data_config is loaded
gt_report_config is loaded
pipeline_config is loaded
config loaded..
component created..
pipeline is running..
```

If the run fails immediately, check:

- All configs in [framework_config/](framework_config/) exist and are valid JSON.
- Log file specified in [general_config.json](framework_config/general_config.json) (`logging_file`) is writable.
- Python version is **>= 3.9** and dependencies from [requirements.txt](requirements.txt) are installed.

---

## 7. Data Layout

Raw annotated data must follow a nested folder structure where each folder name encodes metadata as `key=value` and the leaves contain YOLO-format image/label pairs:

```
dataset_root/
├── Product=Cheerios/
│   ├── lighting=Bright/
│   │   ├── Distance=2FT/
│   │   │   ├── images/*.jpg
│   │   │   └── labels/*.txt
│   │   └── Distance=4FT/
│   └── lighting=Cool/
└── Product=Skittles/
```

The metadata keys/values understood by the framework are declared in [framework_config/config_helper.txt](framework_config/config_helper.txt), for example:

```json
{
  "rawtraining": ["data"],
  "Product": ["Cheerios", "Cococola", "Coffeemate", "Lindor", "Skittles"],
  "lighting": ["Bright", "Cool", "High", "Low", "Warm"],
  "Distance": ["2FT", "4FT", "6FT"]
}
```

Class names are supplied via a `.names` file (see [framework_config/yolov4-custom_training_pipeline.names](framework_config/yolov4-custom_training_pipeline.names) or [AI Cloud Pipeline files/obj.names](AI%20Cloud%20Pipeline%20files/obj.names)).

---

## 8. Usage

### 8.1 Basic Usage — run the default pipeline

The simplest way to run any pipeline is to edit [framework_config/vision_sdk_primary_config.json](framework_config/vision_sdk_primary_config.json) and execute:

```bash
python main.py
```

### 8.2 Command-line overrides

[main.py](main.py) accepts positional arguments to override the primary config without editing the file:

```bash
python main.py <raw_data_source> <pipeline_platform> <gt_report> <pipeline_type>
```

Example — train a custom YOLOv8 model from a local dataset:

```bash
python main.py Local CustomYoloPipeline GTReport TrainingPipeline
```

Example — validate a model against a Dataloop dataset with IVA reports:

```bash
python main.py Dataloop IVAValidationPipeline GTReport ValidationPipeline
```

### 8.3 Programmatic usage

```python
from main import MLOPSPipeline

pipeline = MLOPSPipeline(
    raw_data_source="Local",
    pipeline_platform="CustomYoloPipelineYoloV8",
    gt_report="GTReport",
    pipeline_type="TrainingPipeline",
)

pipeline.load_config()
pipeline.create_components()
pipeline.create_and_run_pipeline()
```

### 8.4 Common workflows

| Task | `pipeline_type` | `pipeline_platform` | Notes |
|------|-----------------|---------------------|-------|
| Train custom YOLOv8 | `TrainingPipeline` | `CustomYoloPipeline` | Set `custom_pipeline_yolo` to `YoloV8` |
| Train on Azure Custom Vision | `TrainingPipeline` | `AzureCustomVisionPipeline` | Configure [azure_custom_vision_config.json](framework_config/azure_custom_vision_config.json) |
| Train on AI Cloud | `TrainingPipeline` | `AICloudPipeline` | Configure [ai_cloud_config.json](framework_config/ai_cloud_config.json) |
| Validate model | `ValidationPipeline` | `IVAValidationPipeline` | Produces IVA reports |
| Validate via embeddings | `ValidationByEmbeddingsPipeline` | `IVAValidationPipeline` | Uses [validation_by_embeddings.json](framework_config/validation_by_embeddings.json) |
| Extract features / embeddings | `FeatureExtractionPipeline` | `IVAValidationPipeline` | See [feature_extraction_pipeline.ipynb](feature_extraction_pipeline.ipynb) |
| Generate augmented dataset | `AugmentationPipeline` | `IVAValidationPipeline` | Uses [augmentations_config.json](framework_config/augmentations_config.json) |
| Run Ground Truth inference | Driven via `gt_inference.py` | — | Run `python gt_inference.py` |

---

## 9. Outputs

- **Trained models** — written to the path configured in the selected platform config.
- **Detailed & summary reports** — produced by [module/report_generator.py](module/report_generator.py), [module/generate_detail_report.py](module/generate_detail_report.py), [module/generate_overall_summary.py](module/generate_overall_summary.py), and [module/generate_summary_representation.py](module/generate_summary_representation.py).
- **Confusion matrices** — via [module/confusion_matrix_generator.py](module/confusion_matrix_generator.py).
- **PDF reports & annotated images** — under [pdf_image_output_folder/](pdf_image_output_folder/).
- **Patch-based inference outputs** — under [patch_outputs/](patch_outputs/) and [patch_inference/](patch_inference/).
- **Embedding & similarity plots** — via [module/plot_embeddings.py](module/plot_embeddings.py) and [module/plot_similarity_score.py](module/plot_similarity_score.py).
- **MLflow runs** — under [mlruns/](mlruns/); browse with `mlflow ui`.

---

## 10. Project Structure

```
├── main.py                          # Entry point — builds and runs the selected pipeline
├── pipeline_manager.py              # Pipeline factory / dispatcher
├── config_manager.py                # Loads and merges configuration files
├── gt_inference.py                  # Ground Truth inference driver
├── sagemaker_inference.py           # SageMaker endpoint inference + class-count Excel report
├── feature_extraction_pipeline.ipynb  # Notebook walkthrough for feature extraction
├── framework_config/                # JSON / YAML configs for every pipeline
├── module/                          # Core pipeline implementations
│   ├── model_training_pipeline.py
│   ├── validation_pipeline.py
│   ├── embeddings_pipeline.py
│   ├── augmentations_pipeline.py
│   ├── patch_inference.py
│   ├── report_generator.py
│   ├── confusion_matrix_generator.py
│   └── ...
├── src/                             # Reusable components (models, features, utils, UI, visualization)
├── utility/                         # Shared helpers and constants
├── mlruns/                          # MLflow experiment tracking output
├── patch_outputs/                   # Patch-based inference results
├── pdf_image_output_folder/         # Generated PDF reports and annotated images
├── requirements.txt / constraints.txt
├── Dockerfile
├── CONTRIBUTING.md
├── LICENCE.md
└── NOTICE.TXT
```

---

## 11. Example — programmatic training run

```python
"""Train a custom YOLOv8 model on a local dataset and generate reports."""
from main import MLOPSPipeline

pipeline = MLOPSPipeline(
    raw_data_source="Local",
    pipeline_platform="CustomYoloPipelineYoloV8",
    gt_report="GTReport",
    pipeline_type="TrainingPipeline",
)

# 1. Load and merge all configs referenced by the primary config
pipeline.load_config()

# 2. Create shared components (logger, exception handler, ...)
pipeline.create_components()

# 3. Build the concrete pipeline for the selected platform and run it end-to-end
pipeline.create_and_run_pipeline()

print("Training complete. See mlruns/ for tracked runs and pdf_image_output_folder/ for reports.")
```

To then generate a consolidated ground-truth report over the trained model:

```bash
python gt_inference.py
python sagemaker_inference.py
```

---

## 12. Logging

Logging is centralized through `module.logger.BPLogger` and driven by [framework_config/general_config.json](framework_config/general_config.json):

```json
{
  "logging_type": "File",
  "logging_file": "ML-OPS_SDK.log",
  "logging_level": 20,
  "logging_db_type": "RDBMS",
  "logging_db_engine": "sqlite:///log_db.sqlite3"
}
```

| Field | Description |
|-------|-------------|
| `logging_type` | `File`, `Console`, or `DB`. |
| `logging_file` | Path of the log file when `logging_type = File`. |
| `logging_level` | Standard Python logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR). |
| `logging_db_type` | DB flavor to log to when `logging_type = DB` (e.g. `RDBMS`). |
| `logging_db_engine` | SQLAlchemy-style DB URL for DB logging. |

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` on startup | Missing dependency | Re-run `pip install -r requirements.txt -c constraints.txt` |
| CUDA / GPU not detected | Wrong CUDA version or missing `requirements_gpu.txt` | Match the CUDA 11.4 toolchain used in the [Dockerfile](Dockerfile) |
| Wrong pipeline runs | Stale/incorrect primary config | Verify [framework_config/vision_sdk_primary_config.json](framework_config/vision_sdk_primary_config.json) against the table in [§4 Supported Pipelines](#4-supported-pipelines) |
| Empty or partial reports | Data folder structure doesn't follow `key=value` convention | See [§7 Data Layout](#7-data-layout) |
| Dataloop / S3 auth errors | Missing credentials in the corresponding config | Populate [dataloop_config.json](framework_config/dataloop_config.json) or [aws_s3.json](framework_config/aws_s3.json) |
| Azure Custom Vision 401/403 | Invalid endpoint or key | Re-check [azure_custom_vision_config.json](framework_config/azure_custom_vision_config.json) |
| Log file not created | `logging_file` path not writable | Update [general_config.json](framework_config/general_config.json) or run with elevated permissions |

---

## 14. Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and ensure any new pipeline follows the config-driven pattern described in [§5 Configuration Explanation](#5-configuration-explanation).

---

## 15. License

Distributed under the terms of [LICENCE.md](LICENCE.md). See [NOTICE.TXT](NOTICE.TXT) for third-party attributions.
