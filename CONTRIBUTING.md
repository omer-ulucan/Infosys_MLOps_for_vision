# Contributing to MLOps for Vision Part of Infosys Topaz

Thank you for considering contributing to **MLOps for Vision Part of Infosys Topaz**! We appreciate your help in making this project better.

## Ways You Can Contribute
There are many ways to help, not just writing code:

### 1) Report bugs & request features
- File issues with reproducible steps, logs, environment details.
- Suggest improvements to APIs, UX, docs, examples.

### 2) Code contributions
- New modules (data loaders, transforms, metrics, evaluators, visualization utilities).
- Model support (wrappers, export/import utilities, ONNX/TensorRT tooling).
- Pipelines (training, inference, batch/stream processing). 
- Integrations (cloud storage, message queues, experiment trackers, model registries).
- CLI tools and scripts.

### 3) Documentation & writing
- Improve README sections, add tutorials/how‑to guides, concept explainers.
- Create example notebooks and end‑to‑end walkthroughs.
- Add API reference docstrings and diagrams.
- Write troubleshooting and migration guides.

### 4) Testing & QA
- Unit, integration, and regression tests.
- Add test fixtures, synthetic datasets, and mocks.
- Increase coverage and tighten CI checks.

### 5) Performance, ops & reliability
- Profiling, caching, vectorization, distributed training/inference.
- Observability hooks (logging, metrics, tracing).
- CI/CD pipelines, release automation, Docker images.

### 6) Community & support
- Triage issues, answer questions, label tickets.
- Curate “good first issue” tasks and mentor newcomers.

## Development Setup
### Prerequisites
- Git ≥ 2.30
- Python ≥ 3.10 (virtualenv or conda recommended)
- Optional: Docker, CUDA/cuDNN for GPU, Make

### Quick start
1. **Fork** the repository.
2. **Clone** your fork locally.
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m "feat: describe your change"
   ```
5. Push your branch:
   ```bash
   git push origin feature-name
   ```
6. Open a **Pull Request** (PR) to the `main` branch.

## Seting Up
- Ensure you have all required dependencies installed.
- Example for Python:
   ```bash
   pip install -r requirements.txt
   ```

## Project Layout
A quick orientation to where code lives — please place new contributions in the folder that matches their role:

- **`module/`** — Core pipeline components (training, validation, embeddings, report generation, patch inference, etc.). Each file typically exposes a class or top-level function that is wired into a pipeline stage.
- **`utility/`** — Standalone helper scripts and small utilities (label checks, image processing, JSON ops, logging helpers). Prefer this folder for reusable tools that are not part of a specific pipeline stage.
- **`src/`** — Vendored object-detection scaffold (bounding boxes, evaluators, metric enumerators) consumed by `gt_inference.py`. Treat as upstream code; avoid unrelated edits.
- **`framework_config/`** — JSON / YAML configuration files that drive pipeline behavior. `general_config.json` selects the platform (`raw_data_source`, `pipeline_platform`, `pipeline_type`); the corresponding platform-specific JSON (e.g. `local_config.json`, `ai_cloud_config.json`) supplies the details. New pipelines should ship their own config file here.
- **`main.py`** / **`pipeline_manager.py`** / **`config_manager.py`** — Entry point, dynamic pipeline builder, and config loader respectively. `ConfigManager` reads the JSONs above and hands typed config objects to the pipeline factory in `pipeline_manager.py`, which uses `importlib` to build the requested pipeline.

When adding a new pipeline type, the usual touch-points are: a new class in `module/`, a matching config file in `framework_config/`, and a route entry in `pipeline_manager.py` / `config_manager.py`.

## Coding Guidelines
- Follow [PEP8](https://peps.python.org/pep-0008/) for Python code style.
- Use descriptive commit messages (e.g., `fix: resolve issue with API response`).
- Keep code modular and well-documented.

## Testing
- Run all tests before submitting a PR:
   ```bash
   pytest
   ```
- Add tests for any new functionality.

## Issue Reporting
- Use GitHub Issues to report bugs or request features.
- Provide detailed steps to reproduce the issue.

## Pull Request Guidelines
- Ensure your PR:
  - Passes all tests.
  - Includes relevant documentation updates.
  - Is linked to an issue if applicable.

## License
By contributing, you agree that your contributions will be licensed under the same license as this project.

Thank you for helping improve MLOps for Vision Part of Infosys Topaz!
