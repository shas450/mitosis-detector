#!/usr/bin/env python3
"""
Dataset Preparation Script for U-Net Training

This script helps users prepare their labeled data for training by:
1. Copying the united.csv file from segmentation folder to train/dataset folder
2. Copying the relevant UUID folders from segmentation to train/dataset folder
3. Validating the dataset structure

Usage:
    python prepare_dataset.py
"""

import os
import shutil
import pandas as pd
import sys
import cv2
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    united_csv_path, train_csv_path, train_data_dir, 
    segmentation_folder, train_dataset_folder
)


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(train_dataset_folder, exist_ok=True)
    print(f"✅ Dataset directory ready: {train_dataset_folder}")


def copy_csv_file():
    """Copy the united.csv file from segmentation to train/dataset folder and prepare it for training."""
    if not os.path.exists(united_csv_path):
        print(f"❌ ERROR: Source CSV file not found: {united_csv_path}")
        print("Please make sure you have labeled data and created the united.csv file.")
        return False
    
    try:
        # Read the original CSV
        df = pd.read_csv(united_csv_path)
        print(f"📊 Original CSV contains {len(df)} samples")
        
        # Check if it has the expected columns
        required_columns = ['UUID', 'user_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ ERROR: CSV file missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Convert user_label to binary format for training
        # Assuming user_label contains 1 for mitosis, 0 for non-mitosis, or empty for unlabeled
        df['Mitosis/Non-Mitosis'] = df['user_label'].fillna(0).astype(int)
        
        # Filter out unlabeled samples (where user_label is empty/NaN)
        labeled_df = df[df['user_label'].notna() & (df['user_label'] != '')]
        
        if len(labeled_df) == 0:
            print("❌ ERROR: No labeled samples found in CSV file")
            print("Please label some samples using the GUI first")
            return False
        
        # Save the processed CSV for training
        labeled_df.to_csv(train_csv_path, index=False)
        print(f"✅ Training CSV created with {len(labeled_df)} labeled samples")
        
        # Show label distribution
        mitosis_count = len(labeled_df[labeled_df['Mitosis/Non-Mitosis'] == 1])
        non_mitosis_count = len(labeled_df[labeled_df['Mitosis/Non-Mitosis'] == 0])
        print(f"📈 Label distribution: {mitosis_count} mitosis, {non_mitosis_count} non-mitosis")
        
        return True
    except Exception as e:
        print(f"❌ ERROR processing CSV file: {e}")
        return False


def get_uuids_from_csv():
    """Get list of UUIDs from the CSV file."""
    try:
        df = pd.read_csv(train_csv_path)
        if 'UUID' not in df.columns:
            print("❌ ERROR: CSV file must contain a 'UUID' column")
            return []
        
        uuids = df['UUID'].unique().tolist()
        print(f"📋 Found {len(uuids)} unique UUIDs in CSV file")
        return uuids
    except Exception as e:
        print(f"❌ ERROR reading CSV file: {e}")
        return []


def copy_uuid_folders(uuids):
    """Copy UUID folders from segmentation to train/dataset folder."""
    copied_count = 0
    missing_count = 0
    
    # Read the training CSV to get labels for each UUID
    try:
        df = pd.read_csv(train_csv_path)
        uuid_labels = dict(zip(df['UUID'], df['Mitosis/Non-Mitosis']))
    except Exception as e:
        print(f"❌ ERROR reading training CSV for labels: {e}")
        return False
    
    for uuid in uuids:
        # Try to find the UUID folder in all available segmentation subdirectories
        source_folder = None
        # Get all subdirectories in segmentation folder that contain UUID folders
        segmentation_subdirs = [d for d in os.listdir(segmentation_folder) 
                               if os.path.isdir(os.path.join(segmentation_folder, d)) 
                               and not d.endswith('.csv')]
        
        for subfolder in segmentation_subdirs:
            potential_source = os.path.join(segmentation_folder, subfolder, uuid)
            if os.path.exists(potential_source):
                source_folder = potential_source
                break
        
        if source_folder is None:
            print(f"⚠️  WARNING: UUID folder not found: {uuid}")
            missing_count += 1
            continue
        
        dest_folder = os.path.join(train_data_dir, uuid)
        
        try:
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)  # Remove existing folder
            
            # Create destination folder
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy combined_image.png
            img_source = os.path.join(source_folder, "combined_image.png")
            img_dest = os.path.join(dest_folder, "combined_image.png")
            
            if os.path.exists(img_source):
                shutil.copy2(img_source, img_dest)
            else:
                print(f"⚠️  WARNING: combined_image.png not found for UUID {uuid}")
                missing_count += 1
                continue
            
            # Handle mask based on label
            label = uuid_labels.get(uuid, 0)
            mask_dest = os.path.join(dest_folder, "mask.tif")
            
            if label == 1:  # Mitosis - copy pred_mask.png
                pred_mask_source = os.path.join(source_folder, "pred_mask.png")
                if os.path.exists(pred_mask_source):
                    # Read pred_mask and save as mask.tif
                    pred_mask = cv2.imread(pred_mask_source, cv2.IMREAD_GRAYSCALE)
                    if pred_mask is not None:
                        cv2.imwrite(mask_dest, pred_mask)
                        print(f"✅ Mitosis sample {uuid}: copied pred_mask.png as mask.tif")
                    else:
                        print(f"⚠️  WARNING: Could not read pred_mask.png for UUID {uuid}")
                        missing_count += 1
                        continue
                else:
                    print(f"⚠️  WARNING: pred_mask.png not found for mitosis sample {uuid}")
                    missing_count += 1
                    continue
            else:  # Non-mitosis - create black mask
                # Get image dimensions to create matching black mask
                img = cv2.imread(img_dest)
                if img is not None:
                    height, width = img.shape[:2]
                    black_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.imwrite(mask_dest, black_mask)
                    print(f"✅ Non-mitosis sample {uuid}: created black mask")
                else:
                    print(f"⚠️  WARNING: Could not read combined_image.png to create mask for UUID {uuid}")
                    missing_count += 1
                    continue
            
            copied_count += 1
            
        except Exception as e:
            print(f"❌ ERROR processing folder {uuid}: {e}")
            missing_count += 1
    
    print(f"✅ Successfully processed {copied_count} UUID folders")
    if missing_count > 0:
        print(f"⚠️  {missing_count} UUID folders were missing or had errors")
    
    return copied_count > 0


def validate_dataset():
    """Validate the prepared dataset."""
    print("\n🔍 Validating dataset...")
    
    if not os.path.exists(train_csv_path):
        print("❌ Training CSV file missing")
        return False
    
    try:
        df = pd.read_csv(train_csv_path)
        print(f"✅ Training CSV contains {len(df)} labeled samples")
        
        # Check required columns
        required_columns = ['UUID', 'Mitosis/Non-Mitosis']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Check class distribution
        if 'Mitosis/Non-Mitosis' in df.columns:
            mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 1])
            non_mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 0])
            print(f"📊 Class distribution: {mitosis_count} mitosis, {non_mitosis_count} non-mitosis")
            
            # Check for class imbalance
            if mitosis_count == 0 or non_mitosis_count == 0:
                print("⚠️  WARNING: Only one class present. Model needs both mitosis and non-mitosis samples")
            elif min(mitosis_count, non_mitosis_count) / max(mitosis_count, non_mitosis_count) < 0.1:
                print("⚠️  WARNING: Severe class imbalance detected. Consider labeling more samples of the minority class")
        
        # Check how many UUID folders exist
        uuids = df['UUID'].unique()
        existing_folders = 0
        valid_samples = 0
        mitosis_with_pred_mask = 0
        non_mitosis_with_black_mask = 0
        
        for uuid in uuids:
            uuid_path = os.path.join(train_data_dir, uuid)
            if os.path.exists(uuid_path):
                existing_folders += 1
                
                # Check if required files exist
                img_path = os.path.join(uuid_path, "combined_image.png")
                mask_path = os.path.join(uuid_path, "mask.tif")
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    valid_samples += 1
                    
                    # Check mask type based on label
                    label = df[df['UUID'] == uuid]['Mitosis/Non-Mitosis'].iloc[0]
                    if label == 1:
                        mitosis_with_pred_mask += 1
                    else:
                        non_mitosis_with_black_mask += 1
        
        print(f"📁 UUID folders found: {existing_folders}/{len(uuids)}")
        print(f"✅ Valid training samples: {valid_samples}")
        print(f"   🔴 Mitosis samples (with pred_mask): {mitosis_with_pred_mask}")
        print(f"   ⚫ Non-mitosis samples (with black mask): {non_mitosis_with_black_mask}")
        
        if valid_samples == 0:
            print("❌ No valid training samples found")
            print("Each UUID folder must contain 'combined_image.png' and 'mask.tif'")
            return False
        
        if valid_samples < len(df):
            missing_samples = len(df) - valid_samples
            print(f"⚠️  {missing_samples} samples are missing required files")
        
        success_rate = valid_samples / len(df)
        if success_rate < 0.5:
            print(f"❌ Too many invalid samples ({success_rate:.1%} success rate)")
            return False
        elif success_rate < 0.8:
            print(f"⚠️  Warning: {success_rate:.1%} of samples are valid")
        
        # Minimum dataset size check
        if valid_samples < 10:
            print("⚠️  WARNING: Very small dataset. Consider labeling more samples for better model performance")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR validating dataset: {e}")
        return False


def main():
    """Main function to prepare the dataset."""
    print("🚀 Preparing dataset for U-Net training...")
    print("="*50)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Copy CSV file
    if not copy_csv_file():
        return False
    
    # Step 3: Get UUIDs from CSV
    uuids = get_uuids_from_csv()
    if not uuids:
        return False
    
    # Step 4: Copy UUID folders
    if not copy_uuid_folders(uuids):
        print("❌ Failed to copy UUID folders")
        return False
    
    # Step 5: Validate dataset
    if not validate_dataset():
        print("❌ Dataset validation failed")
        return False
    
    print("\n" + "="*50)
    print("✅ Dataset preparation completed successfully!")
    print(f"📁 Dataset location: {train_dataset_folder}")
    print("🚀 You can now run the U-Net training script!")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Dataset preparation failed. Please check the errors above.")
        sys.exit(1)
