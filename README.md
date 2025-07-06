# Sketch-to-Image Generation with Pix2Pix GAN

This project implements a Pix2Pix GAN (Generative Adversarial Network) for converting facial sketches to realistic photos using the FS2K dataset.

## Project Structure

```
Sketch-to-image/
├── FS2K/                    # Original dataset
├── prepared_data/           # Processed dataset (train/val/test splits)
├── model.py                 # Model architecture (Generator & Discriminator)
├── train.py                 # Training script
├── inference.py             # Inference script
├── data_preparation.py      # Data preparation script
├── dataset_splitter.py      # Dataset splitting script
├── dataset_augmentation.py  # Data augmentation script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Features

- **U-Net Generator**: Encoder-decoder architecture with skip connections
- **PatchGAN Discriminator**: Patch-based discriminator for high-quality results
- **Pix2Pix Loss**: Combined adversarial and L1 loss for realistic generation
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Training Monitoring**: Progress tracking, sample generation, and checkpointing
- **Inference Pipeline**: Easy-to-use inference for new sketches

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Sketch-to-image
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python model.py
   ```

## Data Preparation

The dataset preparation has already been completed. The process included:

1. **Data Preparation**: Merging sketch and photo pairs into side-by-side images
2. **Dataset Splitting**: 70% train, 15% validation, 15% test
3. **Data Augmentation**: 5x augmentation of training set (7,360 total training images)

The prepared data is located in `prepared_data/` with the following structure:
- `train/`: 7,360 augmented training images
- `val/`: 1,472 validation images
- `test/`: 1,472 test images

## Training

### Quick Start

To start training with default parameters:

```bash
python train.py
```

### Configuration

The training configuration is defined in `train.py`. Key parameters:

```python
config = {
    'data_dir': 'prepared_data',           # Dataset directory
    'output_dir': 'training_output',       # Output directory
    'image_size': 256,                     # Image size
    'batch_size': 8,                       # Batch size
    'num_epochs': 100,                     # Number of epochs
    'learning_rate': 0.0002,               # Learning rate
    'lambda_L1': 100.0,                    # L1 loss weight
    'sample_interval': 100,                # Sample generation interval
    'save_interval': 10,                   # Checkpoint save interval
}
```

### Training Output

The training script creates the following structure:

```
training_output/
├── checkpoints/           # Model checkpoints
│   ├── best_model.pth     # Best model based on validation loss
│   ├── final_model.pth    # Final model after training
│   └── checkpoint_epoch_X.pth  # Periodic checkpoints
├── samples/               # Generated samples during training
│   ├── epoch_0000.png
│   ├── epoch_0100.png
│   └── ...
└── logs/                  # Training logs
    └── training_history.json
```

### Resume Training

To resume training from a checkpoint:

```python
# In train.py, uncomment and modify:
trainer.load_checkpoint('training_output/checkpoints/checkpoint_epoch_50.pth')
```

## Inference

### Single Image

Generate an image from a single sketch:

```bash
python inference.py \
    --model_path training_output/checkpoints/best_model.pth \
    --input path/to/sketch.jpg \
    --output path/to/output.jpg
```

### Batch Processing

Process all sketches in a directory:

```bash
python inference.py \
    --model_path training_output/checkpoints/best_model.pth \
    --input path/to/sketches/ \
    --output path/to/generated_images/
```

### Using CPU

If you don't have a GPU:

```bash
python inference.py \
    --model_path training_output/checkpoints/best_model.pth \
    --input path/to/sketch.jpg \
    --output path/to/output.jpg \
    --device cpu
```

## Model Architecture

### Generator (U-Net)

- **Input**: 256×256×3 sketch image
- **Output**: 256×256×3 generated photo
- **Architecture**: 8-layer encoder-decoder with skip connections
- **Activation**: LeakyReLU (encoder), ReLU (decoder), Tanh (output)

### Discriminator (PatchGAN)

- **Input**: 256×256×6 (concatenated sketch + image)
- **Output**: 30×30×1 patch predictions
- **Architecture**: 3-layer convolutional network
- **Activation**: LeakyReLU

### Loss Functions

- **Generator Loss**: Adversarial loss + λ × L1 loss
- **Discriminator Loss**: Binary cross-entropy for real/fake classification
- **L1 Weight (λ)**: 100.0 (emphasizes pixel-wise accuracy)

## Training Tips

1. **Batch Size**: Start with 8, increase if memory allows
2. **Learning Rate**: 0.0002 works well, can be reduced for fine-tuning
3. **L1 Weight**: 100.0 is standard, lower values may produce less accurate results
4. **Data Augmentation**: Already applied to training set
5. **Validation**: Monitor validation loss to prevent overfitting

## Performance Metrics

The model tracks:
- Generator loss (adversarial + L1)
- Discriminator loss
- Learning rate progression
- Sample quality (visual inspection)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Poor Quality**: Increase L1 weight or training epochs
3. **Mode Collapse**: Check discriminator loss, adjust learning rates
4. **Slow Training**: Use GPU, increase batch size if possible

### GPU Requirements

- **Minimum**: 4GB VRAM for batch size 4
- **Recommended**: 8GB+ VRAM for batch size 8+
- **CPU**: Training possible but very slow

## Dataset Information

The FS2K dataset contains:
- **Original**: 2,000 sketch-photo pairs
- **After Augmentation**: 7,360 training pairs
- **Annotations**: Skin color, lip color, eye color, hair, gender, etc.
- **Image Size**: Variable (resized to 256×256 for training)

## License

This project is for educational purposes. Please respect the original dataset licenses.

## Citation

If you use this code, please cite the original Pix2Pix paper:

```
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
``` 