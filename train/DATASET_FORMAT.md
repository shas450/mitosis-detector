# Training Dataset Format

This document explains the expected format for the training dataset.

## CSV File Format

The `united.csv` file should contain the following columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| UUID | Unique identifier for each cell patch | `fca9af25-f39e-432f-8df6-32b4153088e2` |
| X | X coordinate of the cell center | `43` |
| Y | Y coordinate of the cell center | `29` |
| frame | Frame identifier | `s630009` |
| N_Positive | Number of positive detections (optional) | `0.0` |
| user_label | Manual label: 1=mitosis, 0=non-mitosis | `1` or `0` |

## Example CSV Content

```csv
UUID,X,Y,frame,N_Positive,user_label
fca9af25-f39e-432f-8df6-32b4153088e2,43,29,s630009,0.0,1
befa5044-e3a5-480a-b348-258c3df033b8,150,7,s630009,0.0,0
```

## Folder Structure

For each UUID in the CSV, there should be a corresponding folder containing:

```
segmentation/
├── s630009/
│   ├── fca9af25-f39e-432f-8df6-32b4153088e2/  # Mitosis sample
│   │   ├── combined_image.png     # Input image patch
│   │   └── pred_mask.png         # Predicted mitosis mask (for mitosis samples)
│   └── befa5044-e3a5-480a-b348-258c3df033b8/  # Non-mitosis sample
│       └── combined_image.png     # Input image patch (no mask needed)
└── s630010/
    └── ... (more UUID folders)
```

## Mask Processing During Dataset Preparation

The `prepare_dataset.py` script handles mask creation differently based on labels:

- **Mitosis samples (user_label = 1)**: 
  - Copies `pred_mask.png` from source folder
  - Saves it as `mask.tif` in training dataset
  
- **Non-mitosis samples (user_label = 0)**:
  - Creates a black mask (all zeros) matching image dimensions
  - Saves it as `mask.tif` in training dataset

This ensures that:
- Mitosis samples have actual segmentation masks for training
- Non-mitosis samples have empty masks (no mitosis regions to learn)

## Label Guidelines

- **user_label = 1**: Cell is undergoing mitosis
- **user_label = 0**: Cell is not undergoing mitosis
- **user_label = empty**: Unlabeled (will be filtered out during training)

## Training Dataset Preparation

The `prepare_dataset.py` script will:

1. Read the `united.csv` file
2. Filter only labeled samples (where `user_label` is not empty)
3. Convert `user_label` to `Mitosis/Non-Mitosis` column for training
4. Copy relevant UUID folders to the training dataset directory
5. **Handle masks based on labels:**
   - **Mitosis samples**: Copy `pred_mask.png` as `mask.tif`
   - **Non-mitosis samples**: Create black mask (all zeros) as `mask.tif`
6. Validate the dataset structure

## Required Source Files

For each UUID in the segmentation folder:

### Mitosis Samples (user_label = 1)
- `combined_image.png` (required)
- `pred_mask.png` (required - will be copied as mask.tif)

### Non-mitosis Samples (user_label = 0)  
- `combined_image.png` (required)
- No mask file needed (black mask will be created automatically)

## Minimum Requirements

- At least 10-20 labeled samples (more is better)
- Both mitosis (1) and non-mitosis (0) examples
- Valid image and mask files for each UUID
- Balanced distribution of classes when possible
