import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_files_exist():
    expected = [
        ROOT / 'data' / 'prepare_dataset.py',
        ROOT / 'data' / 'merge_all_classes.py',
        ROOT / 'training' / 'train_resnet50_single_stage.py',
        ROOT / 'training' / 'train_resnet152_single_stage.py',
        ROOT / 'models' / 'resnet50_single_stage_tpu.py',
        ROOT / 'models' / 'resnet152_single_stage_tpu.py',
        ROOT / 'models' / 'layers_head.py',
    ]
    for path in expected:
        assert path.exists(), f"Missing expected file: {path}"


def test_training_module_imports():
    modules = [
        'data.prepare_dataset',
        'data.merge_all_classes',
        'training.train_resnet50_single_stage',
        'training.train_resnet152_single_stage',
        'models.layers_head',
    ]
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        assert spec is not None, f"Module could not be resolved: {module_name}"
