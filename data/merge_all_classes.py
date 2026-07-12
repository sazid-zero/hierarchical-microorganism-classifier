# Merge all classes script
"""
DATASET PREPARATION FOR BASELINE COMPARISONS
============================================
Merge general and bacterial datasets into single directory
with all 38 classes for single-stage baseline training
"""

import os
import shutil
from pathlib import Path
import json

print("="*70)
print("DATASET PREPARATION FOR BASELINE TRAINING")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

KAGGLE_RUN = os.path.exists('/kaggle')

if KAGGLE_RUN:
    # Kaggle paths
    GENERAL_DATA = '/kaggle/working/hierarchical_dataset'
    if not os.path.exists(GENERAL_DATA):
        GENERAL_DATA = '/kaggle/input/microorganism-image-classification'
    
    BACTERIAL_DATA = '/kaggle/input/datasets/asmsharifmahmudsazid/bacteria-classification'
    if not os.path.exists(BACTERIAL_DATA):
        BACTERIAL_DATA = '/kaggle/input/datasets/asmsharifmahmudsazid/bacteria-classification'
    
    OUTPUT_DIR = '/kaggle/working/all_classes_merged'
else:
    # Local paths
    GENERAL_DATA = 'hierarchical_dataset'
    BACTERIAL_DATA = 'cleaned_dataset'
    OUTPUT_DIR = 'all_classes_merged'

print(f"\nSource Directories:")
print(f"  General microorganisms: {GENERAL_DATA}")
print(f"  Bacterial species: {BACTERIAL_DATA}")
print(f"\nOutput Directory:")
print(f"  Merged dataset: {OUTPUT_DIR}")

# ============================================================================
# CHECK SOURCE DIRECTORIES
# ============================================================================

print(f"\n{'='*70}")
print("CHECKING SOURCE DIRECTORIES")
print(f"{'='*70}")

if not os.path.exists(GENERAL_DATA):
    print(f"❌ ERROR: General dataset not found at {GENERAL_DATA}")
    exit(1)
else:
    print(f"✅ General dataset found")

if not os.path.exists(BACTERIAL_DATA):
    print(f"❌ ERROR: Bacterial dataset not found at {BACTERIAL_DATA}")
    exit(1)
else:
    print(f"✅ Bacterial dataset found")

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

print(f"\n{'='*70}")
print("CREATING OUTPUT DIRECTORY")
print(f"{'='*70}")

if os.path.exists(OUTPUT_DIR):
    print(f"⚠️  Output directory already exists: {OUTPUT_DIR}")
    print(f"   Removing existing directory...")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)
print(f"✅ Created: {OUTPUT_DIR}")

# ============================================================================
# COPY NON-BACTERIAL CLASSES
# ============================================================================

print(f"\n{'='*70}")
print("COPYING NON-BACTERIAL CLASSES")
print(f"{'='*70}")

general_classes = [d for d in os.listdir(GENERAL_DATA) 
                  if os.path.isdir(os.path.join(GENERAL_DATA, d))]

print(f"\nFound {len(general_classes)} classes in general dataset:")

non_bacterial_count = 0
bacteria_count = 0

for class_name in general_classes:
    src_dir = os.path.join(GENERAL_DATA, class_name)
    
    # Skip the Bacteria class (we'll use specific species instead)
    if class_name.lower() == 'bacteria':
        bacteria_count = len([f for f in os.listdir(src_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  ⏭️  Skipping: {class_name} ({bacteria_count} images)")
        print(f"      Will use 33 specific bacterial species instead")
        continue
    
    # Copy non-bacterial class
    dst_dir = os.path.join(OUTPUT_DIR, class_name)
    shutil.copytree(src_dir, dst_dir)
    
    img_count = len([f for f in os.listdir(dst_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✅ Copied: {class_name} ({img_count} images)")
    non_bacterial_count += 1

print(f"\n✅ Copied {non_bacterial_count} non-bacterial classes")

# ============================================================================
# COPY BACTERIAL SPECIES CLASSES
# ============================================================================

print(f"\n{'='*70}")
print("COPYING BACTERIAL SPECIES CLASSES")
print(f"{'='*70}")

bacterial_classes = [d for d in os.listdir(BACTERIAL_DATA) 
                    if os.path.isdir(os.path.join(BACTERIAL_DATA, d))]

print(f"\nFound {len(bacterial_classes)} bacterial species:")

bacterial_species_count = 0
total_bacterial_images = 0

for species_name in bacterial_classes:
    src_dir = os.path.join(BACTERIAL_DATA, species_name)
    dst_dir = os.path.join(OUTPUT_DIR, species_name)
    
    shutil.copytree(src_dir, dst_dir)
    
    img_count = len([f for f in os.listdir(dst_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✅ Copied: {species_name} ({img_count} images)")
    bacterial_species_count += 1
    total_bacterial_images += img_count

print(f"\n✅ Copied {bacterial_species_count} bacterial species")
print(f"   Total bacterial images: {total_bacterial_images}")

# ============================================================================
# VERIFY MERGED DATASET
# ============================================================================

print(f"\n{'='*70}")
print("VERIFYING MERGED DATASET")
print(f"{'='*70}")

merged_classes = [d for d in os.listdir(OUTPUT_DIR) 
                 if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

total_classes = len(merged_classes)
expected_classes = 38  # 5 non-bacterial + 33 bacterial

print(f"\nMerged Dataset Summary:")
print(f"  Total classes: {total_classes}")
print(f"  Expected: {expected_classes}")

if total_classes == expected_classes:
    print(f"  ✅ Correct number of classes!")
else:
    print(f"  ⚠️  WARNING: Expected {expected_classes}, found {total_classes}")

# Count images per class
class_stats = {}
total_images = 0

for class_name in sorted(merged_classes):
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    img_count = len([f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    class_stats[class_name] = img_count
    total_images += img_count

print(f"\n  Total images: {total_images}")
print(f"  Average per class: {total_images / total_classes:.1f}")

# Show class distribution
print(f"\n{'='*70}")
print("CLASS DISTRIBUTION")
print(f"{'='*70}")

# Non-bacterial classes
print(f"\nNon-bacterial classes ({non_bacterial_count}):")
for class_name in sorted(merged_classes):
    if class_name in [c for c in general_classes if c.lower() != 'bacteria']:
        print(f"  {class_name:30s}: {class_stats[class_name]:4d} images")

# Bacterial species
print(f"\nBacterial species ({bacterial_species_count}):")
for class_name in sorted(merged_classes):
    if class_name in bacterial_classes:
        print(f"  {class_name:30s}: {class_stats[class_name]:4d} images")

# ============================================================================
# SAVE DATASET METADATA
# ============================================================================

print(f"\n{'='*70}")
print("SAVING METADATA")
print(f"{'='*70}")

metadata = {
    'dataset_type': 'Merged (All Classes)',
    'purpose': 'Baseline single-stage training',
    'total_classes': total_classes,
    'total_images': total_images,
    'non_bacterial_classes': non_bacterial_count,
    'bacterial_species': bacterial_species_count,
    'class_distribution': class_stats,
    'source_datasets': {
        'general': GENERAL_DATA,
        'bacterial': BACTERIAL_DATA
    }
}

metadata_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved to: {metadata_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("DATASET PREPARATION COMPLETE!")
print(f"{'='*70}")

print(f"""
✅ SUMMARY:
   - Non-bacterial classes: {non_bacterial_count}
   - Bacterial species: {bacterial_species_count}
   - Total classes: {total_classes}
   - Total images: {total_images}
   
   Output directory: {OUTPUT_DIR}

NEXT STEPS:
   1. Run baseline_resnet50_single_stage.py
   2. Run baseline_resnet152v2_single_stage.py
   3. Compare results with hierarchical system (98.40%)

Expected baseline results: 88-94% accuracy
Expected hierarchical advantage: +4-10%
""")

print("="*70)