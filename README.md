# Hierarchical Microorganism Classification

Reconstructed implementation of a hierarchical deep learning approach for microorganism classification. This project combines two stages of classification—a coarse-grained taxonomy classifier followed by a fine-grained bacteria species classifier—using confidence-based routing between the two stages.

## Overview

Microorganism classification is essential in microbiology, medicine, and environmental science. Traditional single-stage classifiers face challenges when dealing with imbalanced hierarchical datasets and high inter-class similarity. This work proposes a **two-stage hierarchical classification system** that:

1. **Level 1 (Coarse classification):** Classifies images into 6 broad microorganism categories (e.g., Bacteria, Fungi, Protozoa, Algae, Virus, Other)
2. **Level 2 (Fine classification):** Provides detailed bacteria species classification into 33 specialized classes
3. **Confidence-based routing:** Routes predictions to Level 2 only when Level 1 identifies Bacteria with high confidence (τ ≥ 0.7)

This hierarchical approach reduces the classification burden on individual models and improves overall accuracy by leveraging domain structure.

## Key Features

- **Modular architecture:** Separate ResNet50 and ResNet152V2 model builders for flexibility
- **Two-stage training:** Warmup phase (frozen base) followed by fine-tuning (unfrozen base)
- **Confidence routing:** Smart routing mechanism that avoids unnecessary inference on the fine-grained classifier
- **Data augmentation:** Synthetic augmentation pipeline with random flips, brightness/contrast adjustments, crops, and rotations
- **Portable training:** Works on GPU (MirroredStrategy), CPU, and Kaggle TPU (via TPUClusterResolver)
- **Comprehensive callbacks:** Early stopping, learning rate scheduling, and checkpoint management

## Model Architecture

### Level 1 Classifier (ResNet50 Single-Stage)
- Base: ResNet50 (pre-trained on ImageNet)
- Head: GAP → BN → Dropout(0.5) → Dense(512, ReLU, L2=1e-3) → BN → Dropout(0.4) → Dense(256, ReLU, L2=1e-3) → BN → Dropout(0.3) → Dense(6, softmax)
- Input: 224×224 RGB images
- Output: 6-class probability distribution

### Level 2 Classifier (ResNet152V2 Single-Stage)
- Base: ResNet152V2 (pre-trained on ImageNet)
- Head: GAP → BN → Dropout(0.5) → Dense(256, ReLU, L2=1e-3) → BN → Dropout(0.4) → Dense(33, softmax)
- Input: 224×224 RGB images
- Output: 33-class probability distribution (bacteria species)

## Training Protocol

### Phase 1: Warmup (20 epochs)
- Base model: frozen
- Optimizer: Adam (lr=1e-3)
- Trains only the classification head

### Phase 2: Fine-tuning (100 epochs)
- Base model: unfrozen
- Optimizer: AdamW (lr=1e-5, weight_decay=1e-4, clipnorm=1.0)
- Full model training with regularization

### Regularization & Callbacks
- Class weights (computed from training distribution)
- Early stopping (patience=25, monitor=val_accuracy)
- Learning rate scheduling (ReduceLROnPlateau, factor=0.5, patience=7)
- Batch normalization and dropout layers

## Dataset

- **Source:** Kaggle microorganism dataset (all_classes_merged)
- **Classes:** 38 total (6 coarse + 32 fine bacteria)
- **Split:** 70% train / 15% validation / 15% test
- **Preprocessing:** Standardized to 224×224, ResNet preprocessing applied

## Inference Pipeline

The hierarchical inference runner (`scripts/run_hierarchical.py`) implements Algorithm 1 from the paper:

```
for each image:
    p_level1, conf_level1 = level1_model.predict(image)
    if p_level1 == "Bacteria" and conf_level1 >= tau:
        p_level2, conf_level2 = level2_model.predict(image)
        final_prediction = p_level2
    else:
        final_prediction = p_level1
    save(image, p_level1, conf_level1, p_level2, final_prediction)
```

- **Threshold (τ):** Default 0.7 (configurable)
- **Output:** CSV with per-image predictions, routing decisions, and confidence scores

## Repository Structure

```
hierarchical-microorganism-classifier/
├── data/
│   ├── prepare_dataset.py          # Dataset preparation & synthetic augmentation
│   ├── merge_all_classes.py        # Merge general + bacteria datasets
│   └── utils_io.py                 # I/O utilities
├── models/
│   ├── resnet50_single_stage_tpu.py       # ResNet50 builder
│   ├── resnet152_single_stage_tpu.py      # ResNet152V2 builder
│   └── layers_head.py                     # Reusable classification head
├── training/
│   ├── train_resnet50_single_stage.py     # ResNet50 training CLI
│   ├── train_resnet152_single_stage.py    # ResNet152V2 training CLI
│   └── callbacks.py                       # Centralized callbacks
├── scripts/
│   ├── run_hierarchical.py                # Hierarchical inference runner
│   ├── evaluate_baseline_vs_hierarchical.py # Evaluation harness
│   └── commit_files_individually.ps1      # Git commit helper
├── configs/
│   ├── paths_example_kaggle.yaml          # Kaggle-specific paths
│   ├── paths_example_local.yaml           # Local development paths
│   └── hyperparams.yaml                   # Hyperparameter config
├── notebooks/
│   └── kaggle_experiments.ipynb           # Experiment notebook
├── tests/
│   └── test_pipeline_smoke.py             # Basic import/file checks
├── .github/workflows/ci.yml               # GitHub Actions CI
├── README.md
├── LICENSE                                # MIT License
└── requirements.txt
```

## Quick Start

### Training
```bash
# Prepare dataset
python data/prepare_dataset.py --data-dir raw_data --output-dir all_classes_merged

# Train Level 1 (ResNet50)
python training/train_resnet50_single_stage.py \
    --data-dir all_classes_merged \
    --output-dir baseline_checkpoints \
    --warmup-epochs 20 \
    --epochs 100

# Train Level 2 (ResNet152V2) - on bacteria subset
python training/train_resnet152_single_stage.py \
    --data-dir bacteria_subset \
    --output-dir bacteria_checkpoints \
    --warmup-epochs 20 \
    --epochs 100
```

### Hierarchical Inference
```bash
python scripts/run_hierarchical.py \
    --level1-model baseline_checkpoints/best_model \
    --level2-model bacteria_checkpoints/best_model \
    --data-dir test_images \
    --output-dir outputs \
    --tau 0.7
```

### Evaluation
```bash
python scripts/evaluate_baseline_vs_hierarchical.py \
    --model-path baseline_checkpoints/best_model \
    --data-dir test_set \
    --output-dir eval_results
```

## Next Steps (Future Work)
- Implement multi-GPU distributed training strategy
- Add example notebooks with small dummy datasets for quick verification
- Provide Docker/Conda recipes for reproducible environment setup
- Integrate with MLOps pipeline (Weights & Biases / TensorBoard)
- Benchmark against other hierarchical classification methods

License
- MIT License

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
