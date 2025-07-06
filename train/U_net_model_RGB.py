import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import models
import matplotlib.pyplot as plt

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
        print("2. Run: python train/prepare_dataset.py")
        return False
    
    if not os.path.exists(train_data_dir):
        print(f"❌ ERROR: Training data directory not found: {train_data_dir}")
        return False
    
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

# Use 20% of mitosis for test and match with same number of non-mitosis
num_mitosis_test = int(0.2 * len(mitosis_df))
num_non_mitosis_test = num_mitosis_test

# Create balanced test set
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
        uuid_path = os.path.join(train_data_dir, row['UUID'])
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

# Load train and test data
print("🔄 Loading training data...")
X_train, Y_train = load_data_from_df(train_df)
print("🔄 Loading test data...")
X_test, Y_test = load_data_from_df(test_df)

# Validate loaded data
if len(X_train) == 0:
    print("❌ ERROR: No training data could be loaded!")
    print("Please check that UUID folders exist with 'combined_image.png' and 'mask.tif'")
    exit(1)

if len(X_test) == 0:
    print("❌ ERROR: No test data could be loaded!")
    exit(1)

print(f"✅ Data loaded successfully:")
print(f"   Training samples: {len(X_train)} (images: {X_train.shape}, masks: {Y_train.shape})")
print(f"   Test samples: {len(X_test)} (images: {X_test.shape}, masks: {Y_test.shape})")

# ---------------------- U-Net Model ----------------------
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Resizing

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = UpSampling2D((2, 2))(x)
    target_height, target_width = x.shape[1], x.shape[2]
    skip = Resizing(target_height, target_width)(skip)
    x = Concatenate()([x, skip])
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(100, 100, 3)):
    inputs = Input(input_shape)

    # Encoder
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)

    # Bottleneck
    bottleneck = conv_block(p3, 512)

    # Decoder
    d3 = decoder_block(bottleneck, f3, 256)
    d2 = decoder_block(d3, f2, 128)
    d1 = decoder_block(d2, f1, 64)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(d1)
    outputs = Resizing(IMG_SIZE, IMG_SIZE)(outputs)

    return models.Model(inputs, outputs)

# ---------------------- Training Setup ----------------------
model = build_unet()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy'])

print(f"🚀 Starting training with {len(X_train)} training samples and {len(X_test)} test samples")
print(f"⚙️  Configuration: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LEARNING_RATE}")

# Train the model
history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, Y_test)
)

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"unet_model_RGB_{timestamp}.h5"
model_save_path = os.path.join(train_folder, model_filename)
model.save(model_save_path)

print(f"✅ Training complete. Model saved as: {model_filename}")
print(f"📁 Full path: {model_save_path}")

# ---------------------- Visualization and Analysis ----------------------
def create_visualizations(test_dataframe, model):
    """Create visualizations for test samples."""
    print("\n📊 Creating visualizations...")
    
    viz_dir = os.path.join(train_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    num_samples = min(6, len(test_dataframe))
    samples_to_viz = test_dataframe.head(num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(samples_to_viz.iterrows()):
        uuid_path = os.path.join(train_data_dir, row['UUID'])
        img_path = os.path.join(uuid_path, "combined_image.png")
        mask_path = os.path.join(uuid_path, "mask.tif")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        # Load and process
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        pred_mask_orig_size = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        # Plot images
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'Original\n{row["UUID"][:8]}...\n{"Mitosis" if row["Mitosis/Non-Mitosis"] == 1 else "Non-Mitosis"}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pred_mask_orig_size, cmap='gray')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # Overlay: Green=GT, Red=Pred, Yellow=Overlap
        overlay = img_rgb.copy()
        if pred_mask_orig_size.max() > 0:
            overlay[pred_mask_orig_size > 127] = [255, 0, 0]
        if mask.max() > 0:
            overlay[mask > 127] = [0, 255, 0]
            overlap = (pred_mask_orig_size > 127) & (mask > 127)
            overlay[overlap] = [255, 255, 0]
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay\n(G:GT, R:Pred, Y:Both)')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(viz_dir, f"test_samples_{timestamp}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {viz_path}")
    
    try:
        plt.show()
    except:
        print("📊 Plot saved to file")
    
    plt.close()

def predict_and_save_test_masks(test_dataframe, model):
    """Generate and save predictions for test samples."""
    print("🔮 Generating predictions for test samples...")
    
    for _, row in test_dataframe.iterrows():
        uuid_path = os.path.join(train_data_dir, row['UUID'])
        img_path = os.path.join(uuid_path, "combined_image.png")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Predict
        orig_size = (img.shape[1], img.shape[0])
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred = cv2.resize(mask_pred, orig_size)

        # Save prediction
        save_path = os.path.join(uuid_path, "predicted_mask.tif")
        cv2.imwrite(save_path, mask_pred)

    print("✅ All test predictions saved.")

# ---------------------- Final Analysis ----------------------
print("\n" + "="*60)
print("🔍 ANALYZING TEST RESULTS")
print("="*60)

# Show test sample info
print(f"📋 Test samples: {len(test_df)}")
for idx, (_, row) in enumerate(test_df.iterrows()):
    label = "Mitosis" if row['Mitosis/Non-Mitosis'] == 1 else "Non-Mitosis"
    print(f"{idx+1:2d}. {row['UUID'][:16]}... | {label}")

# Create visualizations
create_visualizations(test_df, model)

# Save predictions
predict_and_save_test_masks(test_df, model)

print("\n" + "="*60)
print("🎉 TRAINING AND ANALYSIS COMPLETE!")
print("="*60)
print(f"📦 Model saved: {model_filename}")
print(f"📊 Visualizations: {os.path.join(train_folder, 'visualizations')}")
print(f"🔮 Predictions saved in UUID folders as 'predicted_mask.tif'")
