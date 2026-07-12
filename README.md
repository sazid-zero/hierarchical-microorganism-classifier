# Microorganism Classification

Reconstructed repository for the microorganism classification project (code restored from paper and recovered fragments).

This repository contains model builders, data utilities and training scripts for single-stage ResNet baselines used in the paper. The code is provided for review and archival on GitHub — it has been refactored to be readable and self-contained, but it has NOT been run or validated end-to-end in this workspace.

Status
- Code reconstruction: mostly complete (model builders, dataset utils, training scripts)
- Tested: basic import/smoke checks only (no full training performed)
- Purpose: repository snapshot suitable for GitHub; intended to be polished and executed later

Quick links
- Training scripts: [training/train_resnet50_single_stage.py](training/train_resnet50_single_stage.py) and [training/train_resnet152_single_stage.py](training/train_resnet152_single_stage.py)
- Model builders: [models/resnet50_single_stage_tpu.py](models/resnet50_single_stage_tpu.py), [models/resnet152_single_stage_tpu.py](models/resnet152_single_stage_tpu.py)
- Dataset utilities: [data/prepare_dataset.py](data/prepare_dataset.py), [data/utils_io.py](data/utils_io.py)
- Callbacks: [training/callbacks.py](training/callbacks.py)
- Evaluation script: [scripts/evaluate_baseline_vs_hierarchical.py](scripts/evaluate_baseline_vs_hierarchical.py)

Minimal usage
 - Inspect or run the training CLI (example):

```bash
python training/train_resnet50_single_stage.py --data-dir all_classes_merged --output-dir baseline_checkpoints
python training/train_resnet152_single_stage.py --data-dir all_classes_merged --output-dir baseline_checkpoints
```

Notes for maintainers
- The repository was reconstructed from the associated paper and partial code fragments; training pipelines follow the paper's warmup + fine-tune protocol (warmup with base frozen, then unfrozen AdamW fine-tune).
- TPU-specific code paths were replaced with portable strategies (MirroredStrategy / TPUClusterResolver detection) so scripts are easier to read and present on GitHub.
- If you plan to run code locally, review and adapt `requirements.txt` to your platform (TensorFlow wheels and numpy have strict platform compatibility on Windows).

Next steps (future work)
- Implement the hierarchical runner with confidence-based routing (Level-1 → Level-2 routing at τ=0.7)
- Add CI smoke tests and example notebooks with small dummy data for quick verification
- Provide verified `requirements.txt` and Docker/Conda recipe for reproducible runs

License
- Add a license file if you intend to open-source this project publicly.

Project structure

```
microorganism-classification/
├── data/
│   ├── prepare_dataset.py
	│   ├── merge_all_classes.py
│   └── utils_io.py
├── models/
│   ├── resnet50_single_stage_tpu.py
│   ├── resnet152_single_stage_tpu.py
│   └── layers_head.py
├── training/
│   ├── train_resnet50_single_stage.py
│   ├── train_resnet152_single_stage.py
│   └── callbacks.py
├── configs/
│   ├── paths_example_kaggle.yaml
│   ├── paths_example_local.yaml
│   └── hyperparams.yaml
├── notebooks/
│   └── kaggle_experiments.ipynb
├── scripts/
│   ├── export_to_onnx.py
│   └── evaluate_baseline_vs_hierarchical.py
├── README.md
└── requirements.txt
```
