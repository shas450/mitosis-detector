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
    
    # Check segmentation folders
    seg_folders = []
    for folder_name in ['s630009', 's630010']:
        folder_path = os.path.join(segmentation_folder, folder_name)
        if os.path.exists(folder_path):
            uuid_count = len([d for d in os.listdir(folder_path) 
                            if os.path.isdir(os.path.join(folder_path, d))])
            seg_folders.append(f"{folder_name}: {uuid_count} folders")
    
    if seg_folders:
        print(f"📁 Segmentation folders: {', '.join(seg_folders)}")
    else:
        print("❌ No segmentation folders found")


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
            
            if 'Mitosis/Non-Mitosis' in df.columns:
                mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 1])
                non_mitosis_count = len(df[df['Mitosis/Non-Mitosis'] == 0])
                print(f"   📈 Mitosis: {mitosis_count}")
                print(f"   📉 Non-mitosis: {non_mitosis_count}")
            
            # Check how many UUID folders exist in training dataset
            if 'UUID' in df.columns:
                uuids = df['UUID'].unique()
                existing_folders = 0
                for uuid in uuids:
                    uuid_path = os.path.join(train_data_dir, uuid)
                    if os.path.exists(uuid_path):
                        existing_folders += 1
                
                print(f"   📁 Available UUID folders: {existing_folders}/{len(uuids)}")
                
                if existing_folders < len(uuids):
                    print(f"   ⚠️  {len(uuids) - existing_folders} UUID folders are missing")
                
        except Exception as e:
            print(f"❌ Error reading training CSV: {e}")
    else:
        print(f"❌ Training CSV not found: {train_csv_path}")
        print("   Run 'python prepare_dataset.py' to create it")


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
    
    if not has_seg_data:
        print("1. 📋 No labeled data found")
        print("   → Use the main GUI to label some data first")
        print("   → Run: python main.py")
        print("   → Make sure to save labeled data to united.csv")
    else:
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
                    print(f"1. � Only {labeled_count} labeled samples found")
                    print("   → Consider labeling more samples for better training")
                    print("   → Aim for at least 50+ samples with balanced classes")
                else:
                    print(f"1. ✅ {labeled_count} labeled samples ready")
        except:
            pass
    
    if has_seg_data and not has_train_data:
        print("2. 🎯 Prepare training dataset")
        print("   → Run: python prepare_dataset.py")
    
    if has_train_data:
        print("3. 🚀 Training dataset is ready")
        print("   → Run: python U_net_model_RGB.py")
    
    print("\n4. 📚 For detailed instructions, see:")
    print("   → train/README.md")
    print("\n5. 🔍 Data format expected:")
    print("   → CSV columns: UUID, user_label (1=mitosis, 0=non-mitosis)")
    print("   → Folders: UUID folders with combined_image.png and mask.tif")


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
