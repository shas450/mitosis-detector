#!/usr/bin/env python3
"""
Dataset Status Checker

This script provides information about the current state of your datasets
for both segmentation and training.

Usage:
    python check_dataset_status.py
"""

import os
import pandas as pd
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    united_csv_path, train_csv_path, train_data_dir, 
    segmentation_folder, train_dataset_folder
)


def check_segmentation_data():
    """Check the status of segmentation data."""
    print("📊 SEGMENTATION DATA STATUS")
    print("=" * 40)
    
    # Check united.csv
    if os.path.exists(united_csv_path):
        try:
            df = pd.read_csv(united_csv_path)
            print(f"✅ United CSV found: {len(df)} total samples")
            
            # Check for labeled samples
            if 'user_label' in df.columns:
                labeled_df = df[df['user_label'].notna() & (df['user_label'] != '')]
                unlabeled_count = len(df) - len(labeled_df)
                print(f"   📋 Labeled samples: {len(labeled_df)}")
                print(f"   ❓ Unlabeled samples: {unlabeled_count}")
                
                if len(labeled_df) > 0:
                    mitosis_count = len(labeled_df[labeled_df['user_label'] == 1])
                    non_mitosis_count = len(labeled_df[labeled_df['user_label'] == 0])
                    print(f"   📈 Mitosis: {mitosis_count}")
                    print(f"   📉 Non-mitosis: {non_mitosis_count}")
            
            if 'UUID' in df.columns:
                unique_uuids = df['UUID'].nunique()
                print(f"   🆔 Unique UUIDs: {unique_uuids}")
                
            # Check frame distribution
            if 'frame' in df.columns:
                frames = df['frame'].unique()
                print(f"   🎬 Frames: {', '.join(sorted(frames))}")
                
        except Exception as e:
            print(f"❌ Error reading united.csv: {e}")
    else:
        print(f"❌ United CSV not found: {united_csv_path}")
        print("   Run the main GUI and label some data first")
    
    # Check segmentation folders - dynamically find all available folders
    seg_folders = []
    if os.path.exists(segmentation_folder):
        for item in os.listdir(segmentation_folder):
            folder_path = os.path.join(segmentation_folder, item)
            # Only consider directories that don't end with .csv
            if os.path.isdir(folder_path) and not item.endswith('.csv'):
                uuid_count = len([d for d in os.listdir(folder_path) 
                                if os.path.isdir(os.path.join(folder_path, d))])
                if uuid_count > 0:  # Only include folders that actually contain UUID folders
                    seg_folders.append(f"{item}: {uuid_count} folders")
    
    if seg_folders:
        print(f"📁 Segmentation folders: {', '.join(seg_folders)}")
    else:
        print("❌ No segmentation folders with UUID data found")


def check_training_data():
    """Check the status of training data."""
    print("\n🚀 TRAINING DATA STATUS")
    print("=" * 40)
    
    # Check if dataset directory exists
    if not os.path.exists(train_dataset_folder):
        print(f"❌ Training dataset folder not found: {train_dataset_folder}")
        print("   Run 'python prepare_dataset.py' to create it")
        return
    
    # Check training CSV
    if os.path.exists(train_csv_path):
        try:
            df = pd.read_csv(train_csv_path)
            print(f"✅ Training CSV found: {len(df)} samples")
            
            # Check columns and detect data source
            if 'Mitosis/Non-Mitosis' in df.columns:
                mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 1])
                non_mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 0])
                print(f"   📈 Mitosis: {mitosis_count}")
                print(f"   📉 Non-mitosis: {non_mitosis_count}")
                print("   🔧 Data format: Training-ready (has 'Mitosis/Non-Mitosis' column)")
            elif 'user_label' in df.columns:
                # Handle old format or manually added data with user_label
                labeled_df = df[df['user_label'].notna() & (df['user_label'] != '')]
                if len(labeled_df) > 0:
                    mitosis_count = len(labeled_df[labeled_df['user_label'] == 1])
                    non_mitosis_count = len(labeled_df[labeled_df['user_label'] == 0])
                    print(f"   📈 Mitosis: {mitosis_count}")
                    print(f"   📉 Non-mitosis: {non_mitosis_count}")
                    print("   🔧 Data format: Legacy format (has 'user_label' column)")
                    print("   💡 Tip: Run 'python prepare_dataset.py' to convert to training format")
                else:
                    print("   ❓ No labeled samples found in CSV")
            else:
                print("   ⚠️  Unknown CSV format - expected 'Mitosis/Non-Mitosis' or 'user_label' column")
                print(f"   📋 Available columns: {list(df.columns)}")
            
            # Check how many UUID folders exist in training dataset
            if 'UUID' in df.columns:
                uuids = df['UUID'].unique()
                existing_folders = 0
                valid_samples = 0
                
                for uuid in uuids:
                    uuid_path = os.path.join(train_data_dir, uuid)
                    if os.path.exists(uuid_path):
                        existing_folders += 1
                        
                        # Check if it has the required files for training
                        img_path = os.path.join(uuid_path, "combined_image.png")
                        mask_path = os.path.join(uuid_path, "mask.tif")
                        
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            valid_samples += 1
                
                print(f"   📁 Available UUID folders: {existing_folders}/{len(uuids)}")
                print(f"   ✅ Valid training samples: {valid_samples}/{len(uuids)}")
                
                if existing_folders < len(uuids):
                    print(f"   ⚠️  {len(uuids) - existing_folders} UUID folders are missing")
                
                if valid_samples < existing_folders:
                    missing_files = existing_folders - valid_samples
                    print(f"   ⚠️  {missing_files} folders missing required files (combined_image.png, mask.tif)")
                
                # Detect if this is manually added data
                if existing_folders > 0:
                    sample_uuid = uuids[0]
                    sample_path = os.path.join(train_data_dir, sample_uuid)
                    if os.path.exists(sample_path):
                        files_in_folder = os.listdir(sample_path)
                        if any('manual' in f.lower() for f in files_in_folder):
                            print("   🖐️  Manual data detected")
                        elif len(files_in_folder) > 2:  # More than just image and mask
                            print("   🔍 Additional files detected in UUID folders")
                
            else:
                print("   ❌ CSV missing 'UUID' column")
                
        except Exception as e:
            print(f"❌ Error reading training CSV: {e}")
    else:
        print(f"❌ Training CSV not found: {train_csv_path}")
        print("   Options:")
        print("   1. Run 'python prepare_dataset.py' to create from segmentation data")
        print("   2. Manually place your training CSV in the train/dataset/ folder")
        
        # Check if there are manually added UUID folders without CSV
        if os.path.exists(train_data_dir):
            manual_folders = [d for d in os.listdir(train_data_dir) 
                            if os.path.isdir(os.path.join(train_data_dir, d))]
            if manual_folders:
                print(f"   🖐️  Found {len(manual_folders)} manually added UUID folders")
                print("   💡 Create a CSV file with UUID and Mitosis/Non-Mitosis columns to use them")


def check_trained_models():
    """Check available trained models."""
    print("\n🤖 TRAINED MODELS STATUS")
    print("=" * 40)
    
    train_folder_path = os.path.dirname(train_csv_path)
    model_files = []
    
    for file in os.listdir(train_folder_path):
        if file.endswith('.h5') or file.endswith('.keras'):
            file_path = os.path.join(train_folder_path, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            model_files.append(f"   📦 {file} ({file_size:.1f} MB)")
    
    if model_files:
        print("✅ Trained models found:")
        for model in model_files:
            print(model)
    else:
        print("❌ No trained models found")
        print("   Run 'python U_net_model_RGB.py' to train a model")


def provide_recommendations():
    """Provide recommendations based on current status."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 40)
    
    # Check if segmentation data exists but training data doesn't
    has_seg_data = os.path.exists(united_csv_path)
    has_train_data = os.path.exists(train_csv_path)
    has_manual_data = False
    
    # Check for manually added training data
    if os.path.exists(train_data_dir):
        manual_folders = [d for d in os.listdir(train_data_dir) 
                        if os.path.isdir(os.path.join(train_data_dir, d))]
        has_manual_data = len(manual_folders) > 0
    
    if not has_seg_data and not has_train_data and not has_manual_data:
        print("1. 📋 No data found")
        print("   Option A: Use the main GUI to label data")
        print("   → Run: python main.py")
        print("   → Make sure to save labeled data to united.csv")
        print("   Option B: Add manual training data")
        print("   → Place UUID folders in train/dataset/data/")
        print("   → Create CSV with UUID, Mitosis/Non-Mitosis columns")
    elif has_seg_data and not has_train_data:
        # Check if there are labeled samples
        try:
            df = pd.read_csv(united_csv_path)
            if 'user_label' in df.columns:
                labeled_count = len(df[df['user_label'].notna() & (df['user_label'] != '')])
                if labeled_count == 0:
                    print("1. 🏷️  Data found but no labels yet")
                    print("   → Use the GUI to label mitosis/non-mitosis samples")
                    print("   → user_label: 1 = mitosis, 0 = non-mitosis")
                elif labeled_count < 20:
                    print(f"1. ⚠️  Only {labeled_count} labeled samples found")
                    print("   → Consider labeling more samples for better training")
                    print("   → Aim for at least 50+ samples with balanced classes")
                else:
                    print(f"1. ✅ {labeled_count} labeled samples ready")
                    print("2. 🎯 Prepare training dataset")
                    print("   → Run: python prepare_dataset.py")
        except:
            pass
    elif has_train_data:
        # Check training data quality
        try:
            df = pd.read_csv(train_csv_path)
            if 'Mitosis/Non-Mitosis' in df.columns:
                print("1. ✅ Training dataset is ready")
                print("   → Run: python U_net_model_RGB.py")
            elif 'user_label' in df.columns:
                print("1. 🔄 Legacy format training data found")
                print("   → Run: python prepare_dataset.py to convert format")
            
            # Check for class balance
            if 'Mitosis/Non-Mitosis' in df.columns:
                mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 1])
                non_mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 0])
                if mitosis_count == 0 or non_mitosis_count == 0:
                    print("2. ⚠️  Class imbalance: Only one class present")
                    print("   → Add samples of the missing class")
                elif min(mitosis_count, non_mitosis_count) / max(mitosis_count, non_mitosis_count) < 0.2:
                    print("2. ⚠️  Severe class imbalance detected")
                    print("   → Consider adding more samples of the minority class")
        except:
            pass
    elif has_manual_data and not has_train_data:
        print("1. 🖐️  Manual training data detected")
        print("   → Create a CSV file: train/dataset/united.csv")
        print("   → Required columns: UUID, Mitosis/Non-Mitosis")
        print("   → Values: 1 = mitosis, 0 = non-mitosis")
        print("   → Then run: python U_net_model_RGB.py")
    
    print("\n📚 Additional Help:")
    print("   → Detailed instructions: train/README.md")
    print("   → Expected data format: train/DATASET_FORMAT.md")
    print("\n🔍 Expected Data Structure:")
    print("   → CSV: UUID, Mitosis/Non-Mitosis (1=mitosis, 0=non-mitosis)")
    print("   → Folders: train/dataset/data/[UUID]/combined_image.png")
    print("   → Folders: train/dataset/data/[UUID]/mask.tif")
    print("\n🛠️  Manual Data Setup:")
    print("   → Place UUID folders in: train/dataset/data/")
    print("   → Each folder needs: combined_image.png, mask.tif")
    print("   → Create CSV: train/dataset/united.csv with UUID and labels")


def main():
    """Main function."""
    print("🔍 MITOSIS DETECTOR - DATASET STATUS CHECK")
    print("=" * 50)
    
    check_segmentation_data()
    check_training_data()
    check_trained_models()
    provide_recommendations()
    
    print("\n" + "=" * 50)
    print("Status check complete! 🎉")


if __name__ == "__main__":
    main()
