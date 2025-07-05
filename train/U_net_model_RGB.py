import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import models

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    train_csv_path, train_data_dir, train_folder, 
    IMG_SIZE, LEARNING_RATE, SEED, BATCH_SIZE, EPOCHS
)

# ---------------------- Custom Weighted Loss ----------------------
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    weight_for_positives = 4.0  # Punish more mistakes on mitosis
    weight_for_negatives = 1.0  # Background

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * weight_for_positives + (1 - y_true) * weight_for_negatives
    weighted_bce = weight_vector * bce

    return tf.reduce_mean(weighted_bce)

# ---------------------- Check Dataset Availability ----------------------
def check_dataset():
    """Check if the dataset is available for training."""
    if not os.path.exists(train_csv_path):
        print("❌ ERROR: Training dataset not found!")
        print(f"Expected CSV file: {train_csv_path}")
        print("\n📋 To prepare the dataset:")
        print("1. Label your data using the segmentation tools")
        print("2. Copy the 'united.csv' file from the segmentation folder to the train/dataset folder")
        print("3. Copy the relevant UUID folders from segmentation to train/dataset folder")
        return False
    
    if not os.path.exists(train_data_dir):
        print(f"❌ ERROR: Training data directory not found: {train_data_dir}")
        return False
    
    # Check if CSV has data
    try:
        df = pd.read_csv(train_csv_path)
        if len(df) == 0:
            print("❌ ERROR: Training CSV file is empty!")
            return False
        
        print(f"✅ Dataset found: {len(df)} samples in {train_csv_path}")
        return True
    except Exception as e:
        print(f"❌ ERROR reading CSV file: {e}")
        return False

# Check dataset before proceeding
if not check_dataset():
    print("\n🛑 Training aborted. Please prepare the dataset first.")
    exit(1)

# ---------------------- Load and Split Data ----------------------
df = pd.read_csv(train_csv_path)
df['Label'] = df['Mitosis/Non-Mitosis']

# Split mitosis and non-mitosis
mitosis_df = df[df['Label'] == 1]
non_mitosis_df = df[df['Label'] == 0]
# Shuffle for reproducibility
mitosis_df = mitosis_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
non_mitosis_df = non_mitosis_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Total test size = 20% of all data
total_samples = len(df)
num_test_total = int(0.2 * total_samples)

# Use only 20% of mitosis for test
num_mitosis_test = int(0.2 * len(mitosis_df))

# Match with same number of non-mitosis to keep test set balanced
num_non_mitosis_test = num_mitosis_test

# Final test size (should be < 20% of total, but balanced)
test_df = pd.concat([
    mitosis_df.iloc[:num_mitosis_test],
    non_mitosis_df.iloc[:num_non_mitosis_test]
]).reset_index(drop=True)

# Remaining data for training
train_df = pd.concat([
    mitosis_df.iloc[num_mitosis_test:],
    non_mitosis_df.iloc[num_non_mitosis_test:]
]).reset_index(drop=True)

print(f"✅ Train samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"🧪 Test set: {num_mitosis_test} mitosis + {num_non_mitosis_test} non-mitosis = {len(test_df)} total")


# ---------------------- Load Images and Masks ----------------------
def load_data_from_df(dataframe):
    X, Y = [], []
    for i, row in dataframe.iterrows():
        uuid_path = os.path.join(train_data_dir, row['UUID'])  # 'UUID' column expected
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None or img.shape[:2] != mask.shape:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

            X.append(img / 255.0)
            Y.append(np.expand_dims(mask > 0, axis=-1).astype(np.float32))
    return np.array(X), np.array(Y)

# Load train and test
print("🔄 Loading training data...")
X_train, Y_train = load_data_from_df(train_df)
print("🔄 Loading test data...")
X_test, Y_test = load_data_from_df(test_df)

# Validate loaded data
if len(X_train) == 0:
    print("❌ ERROR: No training data could be loaded!")
    print("Please check that:")
    print("1. UUID folders exist in the dataset directory")
    print("2. Each folder contains 'combined_image.png' and 'mask.tif'")
    print("3. Image and mask files are valid and readable")
    exit(1)

if len(X_test) == 0:
    print("❌ ERROR: No test data could be loaded!")
    exit(1)

print(f"✅ Data loaded successfully:")
print(f"   Training samples: {len(X_train)} (images: {X_train.shape}, masks: {Y_train.shape})")
print(f"   Test samples: {len(X_test)} (images: {X_test.shape}, masks: {Y_test.shape})")

# ---------------------- U-Net Model ----------------------
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = MaxPooling2D((2, 2))(f)
    return f, p

def build_unet(input_shape=(100, 100, 3)):
    inputs = Input(input_shape)

    # Simplified encoder - only 2 levels to avoid dimension issues
    f1, p1 = encoder_block(inputs, 64)      # f1: 100x100, p1: 50x50
    f2, p2 = encoder_block(p1, 128)        # f2: 50x50, p2: 25x25

    # Bottleneck
    bottleneck = conv_block(p2, 256)       # 25x25

    # Simplified decoder
    d2 = UpSampling2D((2, 2))(bottleneck)   # 25x25 -> 50x50
    d2 = Concatenate()([d2, f2])            # Concatenate with f2: 50x50
    d2 = conv_block(d2, 128)
    
    d1 = UpSampling2D((2, 2))(d2)          # 50x50 -> 100x100
    d1 = Concatenate()([d1, f1])            # Concatenate with f1: 100x100
    d1 = conv_block(d1, 64)

    # Final output layer
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(d1)

    return models.Model(inputs, outputs)

# ---------------------- Visualization Functions ----------------------
import matplotlib.pyplot as plt

def print_test_sample_info(test_dataframe):
    """Print detailed information about test samples."""
    print("\n📋 TEST SAMPLE DETAILS")
    print("=" * 60)
    
    for idx, (_, row) in enumerate(test_dataframe.iterrows()):
        uuid = row['UUID']
        label = "Mitosis" if row['Mitosis/Non-Mitosis'] == 1 else "Non-Mitosis"
        
        uuid_path = os.path.join(train_data_dir, uuid)
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        img_exists = "✅" if os.path.exists(img_path) else "❌"
        mask_exists = "✅" if os.path.exists(mask_path) else "❌"
        
        print(f"{idx+1:2d}. UUID: {uuid[:16]}... | Label: {label:11s} | Image: {img_exists} | Mask: {mask_exists}")
        
        # Additional info if files exist
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                print(f"     Image size: {img.shape[:2]} | Mask pixels: {np.sum(mask > 127)}")
    
    print("=" * 60)

def visualize_test_samples(test_dataframe, model, num_samples=8):
    """Visualize test samples with images, ground truth masks, and predictions."""
    print(f"🖼️  Visualizing {min(num_samples, len(test_dataframe))} test samples...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(train_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select samples to visualize
    samples_to_viz = test_dataframe.head(num_samples)
    
    # Set up the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(samples_to_viz.iterrows()):
        uuid_path = os.path.join(train_data_dir, row['UUID'])
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"⚠️  Skipping {row['UUID']} - files not found")
            continue
        
        # Load original image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"⚠️  Skipping {row['UUID']} - could not load files")
            continue
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for prediction
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        
        # Resize prediction back to original size
        pred_mask_orig_size = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        # Plot original image
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'Original Image\nUUID: {row["UUID"][:8]}...\nLabel: {"Mitosis" if row["Mitosis/Non-Mitosis"] == 1 else "Non-Mitosis"}')
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth Mask')
        axes[idx, 1].axis('off')
        
        # Plot prediction mask
        axes[idx, 2].imshow(pred_mask_orig_size, cmap='gray')
        axes[idx, 2].set_title('Predicted Mask')
        axes[idx, 2].axis('off')
        
        # Plot overlay
        overlay = img_rgb.copy()
        # Make prediction red and ground truth green for comparison
        if pred_mask_orig_size.max() > 0:
            overlay[pred_mask_orig_size > 127] = [255, 0, 0]  # Red for prediction
        if mask.max() > 0:
            overlay[mask > 127] = [0, 255, 0]  # Green for ground truth
            # Yellow where they overlap
            overlap = (pred_mask_orig_size > 127) & (mask > 127)
            overlay[overlap] = [255, 255, 0]
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay\n(Green: GT, Red: Pred, Yellow: Overlap)')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = os.path.join(viz_dir, f"test_samples_visualization_{timestamp}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {viz_path}")
    
    # Try to show the plot (might not work in all environments)
    try:
        plt.show()
    except:
        print("📊 Plot saved to file (display not available in this environment)")
    
    plt.close()

def save_individual_test_samples(test_dataframe, model):
    """Save individual images of each test sample for detailed inspection."""
    print("💾 Saving individual test sample images...")
    
    # Create output directory
    individual_dir = os.path.join(train_folder, "test_samples_individual")
    os.makedirs(individual_dir, exist_ok=True)
    
    for idx, (_, row) in enumerate(test_dataframe.iterrows()):
        uuid = row['UUID']
        label = "mitosis" if row['Mitosis/Non-Mitosis'] == 1 else "non_mitosis"
        
        uuid_path = os.path.join(train_data_dir, uuid)
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Generate prediction
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        pred_mask_orig_size = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        # Save files
        sample_dir = os.path.join(individual_dir, f"{idx:02d}_{label}_{uuid[:8]}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save original image
        cv2.imwrite(os.path.join(sample_dir, "original_image.png"), img)
        
        # Save ground truth mask
        cv2.imwrite(os.path.join(sample_dir, "ground_truth_mask.png"), mask)
        
        # Save predicted mask
        cv2.imwrite(os.path.join(sample_dir, "predicted_mask.png"), pred_mask_orig_size)
        
        # Create and save overlay
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        if pred_mask_orig_size.max() > 0:
            overlay[pred_mask_orig_size > 127] = [255, 0, 0]  # Red for prediction
        if mask.max() > 0:
            overlay[mask > 127] = [0, 255, 0]  # Green for ground truth
            overlap = (pred_mask_orig_size > 127) & (mask > 127)
            overlay[overlap] = [255, 255, 0]  # Yellow for overlap
        
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sample_dir, "overlay.png"), overlay_bgr)
        
        # Save info file
        info_path = os.path.join(sample_dir, "info.txt")
        with open(info_path, 'w') as f:
            f.write(f"UUID: {uuid}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Ground Truth Pixels: {np.sum(mask > 127)}\n")
            f.write(f"Predicted Pixels: {np.sum(pred_mask_orig_size > 127)}\n")
            f.write(f"Overlap Pixels: {np.sum((pred_mask_orig_size > 127) & (mask > 127))}\n")
    
    print(f"📁 Individual samples saved in: {individual_dir}")

# ---------------------- Training Setup ----------------------
model = build_unet()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy'])

print(f"🚀 Starting training with {len(X_train)} training samples and {len(X_test)} test samples")
print(f"⚙️  Configuration: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LEARNING_RATE}")

# Train
history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, Y_test)
)

# Generate unique model name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"unet_model_RGB_{timestamp}.h5"
model_save_path = os.path.join(train_folder, model_filename)

# Save final model
model.save(model_save_path)
print(f"✅ Training complete. Model saved as: {model_filename}")
print(f"📁 Full path: {model_save_path}")

# ---------------------- Visualize and Analyze Test Results ----------------------
print("\n" + "="*60)
print("🔍 ANALYZING TEST SAMPLES")
print("="*60)

# Print test sample information
print_test_sample_info(test_df)

# Create visualizations
print("\n📊 Creating visualizations...")
visualize_test_samples(test_df, model, num_samples=min(8, len(test_df)))

# Save individual samples for detailed inspection
save_individual_test_samples(test_df, model)

print("\n💡 TO VIEW TEST SAMPLES:")
print(f"1. 📊 Combined visualization: {os.path.join(train_folder, 'visualizations')}")
print(f"2. 📁 Individual samples: {os.path.join(train_folder, 'test_samples_individual')}")
print("3. 🖼️  Each sample folder contains:")
print("   - original_image.png (input image)")
print("   - ground_truth_mask.png (expected mask)")  
print("   - predicted_mask.png (model prediction)")
print("   - overlay.png (comparison: Green=GT, Red=Pred, Yellow=Overlap)")
print("   - info.txt (detailed statistics)")

# ---------------------- Predict and Save Masks ----------------------
def predict_and_save_masks(test_dataframe):
    print("\n🔮 Generating predictions for test samples...")
    for i, row in test_dataframe.iterrows():
        uuid_path = os.path.join(train_data_dir, row['UUID'])
        img_path = os.path.join(uuid_path, "combined_image.png")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_size = (img.shape[1], img.shape[0])
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred = cv2.resize(mask_pred, orig_size)

        save_path = os.path.join(uuid_path, "predicted_mask.tif")
        cv2.imwrite(save_path, mask_pred)

    print("🧪 All test masks predicted and saved.")

predict_and_save_masks(test_df)

print("\n" + "="*60)
print("🎉 TRAINING AND ANALYSIS COMPLETE!")
print("="*60)

# ---------------------- Visualization Functions ----------------------
import matplotlib.pyplot as plt

def visualize_test_samples(test_dataframe, model, num_samples=8):
    """Visualize test samples with images, ground truth masks, and predictions."""
    print(f"🖼️  Visualizing {min(num_samples, len(test_dataframe))} test samples...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(train_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select samples to visualize
    samples_to_viz = test_dataframe.head(num_samples)
    
    # Set up the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(samples_to_viz.iterrows()):
        uuid_path = os.path.join(train_data_dir, row['UUID'])
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"⚠️  Skipping {row['UUID']} - files not found")
            continue
        
        # Load original image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"⚠️  Skipping {row['UUID']} - could not load files")
            continue
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for prediction
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        
        # Resize prediction back to original size
        pred_mask_orig_size = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        # Plot original image
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'Original Image\nUUID: {row["UUID"][:8]}...\nLabel: {"Mitosis" if row["Mitosis/Non-Mitosis"] == 1 else "Non-Mitosis"}')
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth Mask')
        axes[idx, 1].axis('off')
        
        # Plot prediction mask
        axes[idx, 2].imshow(pred_mask_orig_size, cmap='gray')
        axes[idx, 2].set_title('Predicted Mask')
        axes[idx, 2].axis('off')
        
        # Plot overlay
        overlay = img_rgb.copy()
        # Make prediction red and ground truth green for comparison
        if pred_mask_orig_size.max() > 0:
            overlay[pred_mask_orig_size > 127] = [255, 0, 0]  # Red for prediction
        if mask.max() > 0:
            overlay[mask > 127] = [0, 255, 0]  # Green for ground truth
            # Yellow where they overlap
            overlap = (pred_mask_orig_size > 127) & (mask > 127)
            overlay[overlap] = [255, 255, 0]
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay\n(Green: GT, Red: Pred, Yellow: Overlap)')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = os.path.join(viz_dir, f"test_samples_visualization_{timestamp}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {viz_path}")
    
    # Try to show the plot (might not work in all environments)
    try:
        plt.show()
    except:
        print("📊 Plot saved to file (display not available in this environment)")
    
    plt.close()

def save_individual_test_samples(test_dataframe, model):
    """Save individual images of each test sample for detailed inspection."""
    print("💾 Saving individual test sample images...")
    
    # Create output directory
    individual_dir = os.path.join(train_folder, "test_samples_individual")
    os.makedirs(individual_dir, exist_ok=True)
    
    for idx, (_, row) in enumerate(test_dataframe.iterrows()):
        uuid = row['UUID']
        label = "mitosis" if row['Mitosis/Non-Mitosis'] == 1 else "non_mitosis"
        
        uuid_path = os.path.join(train_data_dir, uuid)
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Generate prediction
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        pred_mask_orig_size = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        # Save files
        sample_dir = os.path.join(individual_dir, f"{idx:02d}_{label}_{uuid[:8]}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save original image
        cv2.imwrite(os.path.join(sample_dir, "original_image.png"), img)
        
        # Save ground truth mask
        cv2.imwrite(os.path.join(sample_dir, "ground_truth_mask.png"), mask)
        
        # Save predicted mask
        cv2.imwrite(os.path.join(sample_dir, "predicted_mask.png"), pred_mask_orig_size)
        
        # Create and save overlay
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        if pred_mask_orig_size.max() > 0:
            overlay[pred_mask_orig_size > 127] = [255, 0, 0]  # Red for prediction
        if mask.max() > 0:
            overlay[mask > 127] = [0, 255, 0]  # Green for ground truth
            overlap = (pred_mask_orig_size > 127) & (mask > 127)
            overlay[overlap] = [255, 255, 0]  # Yellow for overlap
        
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sample_dir, "overlay.png"), overlay_bgr)
        
        # Save info file
        info_path = os.path.join(sample_dir, "info.txt")
        with open(info_path, 'w') as f:
            f.write(f"UUID: {uuid}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Ground Truth Pixels: {np.sum(mask > 127)}\n")
            f.write(f"Predicted Pixels: {np.sum(pred_mask_orig_size > 127)}\n")
            f.write(f"Overlap Pixels: {np.sum((pred_mask_orig_size > 127) & (mask > 127))}\n")
    
    print(f"📁 Individual samples saved in: {individual_dir}")

def print_test_sample_info(test_dataframe):
    """Print detailed information about test samples."""
    print("\n📋 TEST SAMPLE DETAILS")
    print("=" * 60)
    
    for idx, (_, row) in enumerate(test_dataframe.iterrows()):
        uuid = row['UUID']
        label = "Mitosis" if row['Mitosis/Non-Mitosis'] == 1 else "Non-Mitosis"
        
        uuid_path = os.path.join(train_data_dir, uuid)
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        img_exists = "✅" if os.path.exists(img_path) else "❌"
        mask_exists = "✅" if os.path.exists(mask_path) else "❌"
        
        print(f"{idx+1:2d}. UUID: {uuid[:16]}... | Label: {label:11s} | Image: {img_exists} | Mask: {mask_exists}")
        
        # Additional info if files exist
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                print(f"     Image size: {img.shape[:2]} | Mask pixels: {np.sum(mask > 127)}")
    
    print("=" * 60)
