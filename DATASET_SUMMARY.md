# FS2K Dataset Preparation and Splitting Summary

## Overview

The FS2K (Face Sketch to Image) dataset has been successfully prepared and split for Pix2Pix GAN training. This document summarizes the entire process and provides details about the final dataset structure.

## Dataset Preparation Process

### 1. Original Dataset Structure
- **Source**: FS2K dataset with photos and sketches
- **Photos**: Located in `FS2K/FS2K/photo/photo1/`, `photo2/`, `photo3/`
- **Sketches**: Located in `FS2K/FS2K/sketch/sketch1/`, `sketch2/`, `sketch3/`
- **Annotations**: `anno_train.json` and `anno_test.json` with facial attributes

### 2. Data Preparation (`data_preparation.py`)
- **Merged sketch and photo pairs** into side-by-side images
- **Image size**: 512x256 pixels (sketch on left, photo on right)
- **Format**: JPG files suitable for Pix2Pix training
- **Metadata preservation**: All original attributes maintained
- **Output**: `prepared_data/` directory with train/val split (80%/20%)

### 3. Dataset Splitting (`dataset_splitter.py`)
- **Input**: Prepared dataset from `prepared_data/`
- **Split ratios**: 70% train, 15% validation, 15% test
- **Output**: `split_data/` directory with proper train/val/test structure

## Final Dataset Statistics

### Total Dataset
- **Total images**: 2,104 sketch-photo pairs
- **Image format**: 512x256 pixels (side-by-side merged)
- **File format**: JPG

### Split Distribution
| Split | Count | Percentage | Usage |
|-------|-------|------------|-------|
| **Train** | 1,472 | 70.0% | Model training |
| **Validation** | 315 | 15.0% | Early stopping, hyperparameter tuning |
| **Test** | 317 | 15.0% | Final evaluation |

## Directory Structure

```
split_data/
├── train/                    # 1,472 training images
│   ├── train_000000.jpg
│   ├── train_000001.jpg
│   └── ...
├── val/                      # 315 validation images
│   ├── val_000000.jpg
│   ├── val_000001.jpg
│   └── ...
├── test/                     # 317 test images
│   ├── test_000000.jpg
│   ├── test_000001.jpg
│   └── ...
├── train_metadata.json       # Training data metadata
├── val_metadata.json         # Validation data metadata
├── test_metadata.json        # Test data metadata
└── dataset_metadata.json     # Overall dataset information
```

## Image Format

Each image contains:
- **Left half (256x256)**: Sketch image (input for Pix2Pix)
- **Right half (256x256)**: Photo image (target for Pix2Pix)

```
┌─────────────┬─────────────┐
│             │             │
│   SKETCH    │    PHOTO    │
│   (Input)   │   (Target)  │
│             │             │
└─────────────┴─────────────┘
```

## Metadata Structure

Each metadata file contains:
- **Filename**: New filename in the split
- **Original source**: Original directory (train/val)
- **Original filename**: Original filename
- **Split**: Current split assignment
- **Index**: Sequential index in the split
- **Original metadata**: All facial attributes (skin color, hair type, gender, etc.)

## Facial Attributes Preserved

The dataset maintains all original facial attributes:
- **Skin color**: RGB values
- **Lip color**: RGB values
- **Eye color**: RGB values
- **Hair**: Type (0/1)
- **Hair color**: Category (0-4)
- **Gender**: Binary (0/1)
- **Earring**: Presence (0/1)
- **Smile**: Presence (0/1)
- **Frontal face**: Orientation (0/1)
- **Style**: Category (0/1)

## Usage for Pix2Pix Training

### Training Process
1. **Input**: Left half of merged image (sketch)
2. **Target**: Right half of merged image (photo)
3. **Generator**: Transforms sketch → photo
4. **Discriminator**: Evaluates realism of generated photo|sketch pairs
5. **Loss**: Adversarial loss + L1 reconstruction loss

### Data Loading
```python
# Example data loading for training
train_images = load_images_from_directory('split_data/train/')
val_images = load_images_from_directory('split_data/val/')
test_images = load_images_from_directory('split_data/test/')

# Split each image into input (sketch) and target (photo)
for image in train_images:
    sketch = image[:, :256, :]  # Left half
    photo = image[:, 256:, :]   # Right half
```

## Quality Assurance

### Data Integrity
- ✅ All images successfully merged and resized
- ✅ Metadata preserved and linked
- ✅ No corrupted or missing files
- ✅ Proper train/val/test split with no overlap
- ✅ Random shuffling for unbiased splits

### File Validation
- ✅ All images are 512x256 pixels
- ✅ All images are valid JPG files
- ✅ Metadata files are valid JSON
- ✅ File counts match expected totals

## Next Steps

1. **Train Pix2Pix Model**: Use `split_data/train/` for training
2. **Validate During Training**: Use `split_data/val/` for early stopping
3. **Evaluate Performance**: Use `split_data/test/` for final evaluation
4. **Fine-tune Architecture**: Experiment with different generator/discriminator designs
5. **Add Conditioning**: Incorporate facial attributes for controlled generation

## Scripts Created

1. **`data_preparation.py`**: Merges sketch-photo pairs into training images
2. **`dataset_splitter.py`**: Splits prepared data into train/val/test sets
3. **`requirements.txt`**: Python dependencies
4. **`README.md`**: Comprehensive documentation
5. **`DATASET_SUMMARY.md`**: This summary document

## Reproducibility

- **Random seed**: 42 (for consistent splits)
- **Split ratios**: Configurable via command line arguments
- **Metadata tracking**: Full provenance of all files
- **Version control**: All scripts and configurations tracked

The dataset is now ready for Pix2Pix GAN training with proper train/validation/test splits! 