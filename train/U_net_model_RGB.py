
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import models


# ---------------------- Config ----------------------
IMG_SIZE = 100
DATA_DIR = r"C:\Users\sharo\Desktop\Odd_test\Mix_data"
CSV_PATH = r"C:\Users\sharo\Desktop\Odd_test\mix_data.csv"
LEARNING_RATE = 1e-4
SEED = 42  # Fixed random split

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

# ---------------------- Load and Split Data ----------------------
df = pd.read_csv(CSV_PATH)
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
        uuid_path = os.path.join(DATA_DIR, row['UUID'])  # 'UUID' column expected
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
X_train, Y_train = load_data_from_df(train_df)
X_test, Y_test = load_data_from_df(test_df)

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

# Train
model.fit(
    X_train, Y_train,
    batch_size=16,
    epochs=10,
    validation_data=(X_test, Y_test)
)

# Save final model
model.save("unet_model_RGB_4.h5")
print("✅ Training complete. Final model saved.")

# ---------------------- Predict and Save Masks ----------------------
def predict_and_save_masks(test_dataframe):
    for i, row in test_dataframe.iterrows():
        uuid_path = os.path.join(DATA_DIR, row['UUID'])
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
