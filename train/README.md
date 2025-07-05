# Training Workflow

This folder contains the training scripts and datasets for the mitosis detection U-Net model.

## Folder Structure

```
train/
├── dataset/                    # Training dataset (user copies data here)
│   ├── united.csv             # Labels and metadata
│   └── [UUID folders]/        # Image and mask data
├── prepare_dataset.py         # Script to prepare training data
├── U_net_model_RGB.py         # Main training script
└── trained_models/            # Saved models (created after training)
```

## Workflow

### 1. Label Your Data
- Use the segmentation tools in the main project to label your data
- This creates `united.csv` and UUID folders in the `segmentation/` folder

### 2. Prepare Training Dataset
Run the dataset preparation script:
```bash
python prepare_dataset.py
```

This script will:
- Copy `united.csv` from `segmentation/` to `train/dataset/`
- Copy relevant UUID folders from `segmentation/` to `train/dataset/`
- Validate the dataset structure

### 3. Train the Model
Run the training script:
```bash
python U_net_model_RGB.py
```

This will:
- Load the dataset from `train/dataset/`
- Split data into training and testing sets (80/20 split)
- Train a U-Net model for mitosis segmentation
- Save the trained model with timestamp in the `train/` folder
- Generate predictions on the test set

## Configuration

All paths and training parameters are defined in the main `config.py` file:
- `IMG_SIZE`: Input image size (default: 100x100)
- `LEARNING_RATE`: Learning rate for training (default: 1e-4)
- `BATCH_SIZE`: Training batch size (default: 16)
- `EPOCHS`: Number of training epochs (default: 10)
- `SEED`: Random seed for reproducible splits (default: 42)

## Model Output

After training, you'll find:
- A trained model file: `unet_model_RGB_YYYYMMDD_HHMMSS.h5`
- Predicted masks for test samples in their respective UUID folders
- Training progress and validation metrics in the console output

## Dataset Requirements

Your dataset should contain:
- `united.csv` with columns: `UUID`, `Mitosis/Non-Mitosis`
- UUID folders containing:
  - `combined_image.png`: Input image
  - `mask.tif`: Ground truth segmentation mask

## Tips

1. Ensure balanced classes (mitosis vs non-mitosis) for best results
2. The model uses weighted loss to handle class imbalance
3. Check the dataset validation output before training
4. Models are saved with timestamps to avoid overwriting
5. Use the latest trained model for inference in the main project
