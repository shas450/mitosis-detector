# """
# train_unet_patches.py
# ---------------------
#
# Train a fully-convolutional U-Net on grayscale image patches and binary masks.
#
# • Works with any consistent patch size (e.g. 256×256).
# • Automatically pairs files whose names share the same numeric suffix:
#       img_0000_0_256.png   <->  mask_0000_0_256.png
#   Extra images or masks are reported and skipped.
#
# Outputs:
#     unet_cell_patches.keras   – best-validation-loss weights (Keras v3 format)
#
# Author: ChatGPT
# """
#
# # ---------------------------------------------------------------------
# # 1. CONFIGURATION
# # ---------------------------------------------------------------------
# IMG_DIR   = r"C:\Users\sharo\Desktop\Odd_test\s54\patches\img"
# MASK_DIR  = r"C:\Users\sharo\Desktop\Odd_test\s54\patches\mask"
# MODEL_OUT = "unet_cell_patches.keras"        # ← new extension!
#
# VAL_SPLIT   = 0.20          # 80 % train, 20 % validation
# RANDOM_SEED = 42
# EPOCHS      = 20
# BATCH       = 8
# LR          = 1e-4          # Adam learning rate
#
# # ---------------------------------------------------------------------
# # 2. IMPORTS (keep them after CONFIG so the header stays readable)
# # ---------------------------------------------------------------------
# import os
# import numpy as np
# from PIL import Image
#
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
#
#
# # ---------------------------------------------------------------------
# # 3. DATA-LOADING (robust pairing)
# # ---------------------------------------------------------------------
# def load_pairs(img_dir, mask_dir, verbose=True):
#     """
#     Reads every file in `img_dir` and `mask_dir`, matches on the numeric
#     suffix after the FIRST underscore, and returns
#
#         X, y  with shape (N, H, W, 1)   and dtype float32.
#
#     Unmatched files are skipped but listed.
#     """
#     def list_files(d):
#         return [f for f in os.listdir(d) if not f.startswith('.')]
#
#     img_dict  = { "_".join(f.split("_")[1:]) : f for f in list_files(img_dir) }
#     mask_dict = { "_".join(f.split("_")[1:]) : f for f in list_files(mask_dir) }
#
#     common_keys  = sorted(img_dict.keys() & mask_dict.keys())
#     missing_img  = sorted(mask_dict.keys() - img_dict.keys())
#     missing_mask = sorted(img_dict.keys()  - mask_dict.keys())
#
#     if verbose:
#         print(f"✓ Found {len(common_keys):,} matching patch pairs.")
#         if missing_img:
#             print(f"⚠  {len(missing_img)} masks without an image (skipped).")
#         if missing_mask:
#             print(f"⚠  {len(missing_mask)} images without a mask (skipped).")
#
#     imgs, masks = [], []
#     for k in common_keys:
#         img_path  = os.path.join(img_dir,  img_dict[k])
#         mask_path = os.path.join(mask_dir, mask_dict[k])
#
#         img  = Image.open(img_path).convert('L')   # grayscale 0-255
#         mask = Image.open(mask_path).convert('L')  # binary 0-255
#
#         img_arr  = np.asarray(img,  dtype=np.float32) / 255.0
#         mask_arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
#
#         imgs.append(img_arr[..., np.newaxis])      # add channel dim
#         masks.append(mask_arr[..., np.newaxis])
#
#     X = np.stack(imgs,  axis=0)
#     y = np.stack(masks, axis=0)
#     return X, y
#
#
# # ---------------------------------------------------------------------
# # 4. U-NET ARCHITECTURE (fully-conv; variable input size)
# # ---------------------------------------------------------------------
# def conv_block(x, n_filters):
#     x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
#     x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
#     return x
#
# def build_unet(n_filters_start=32):
#     inputs = layers.Input(shape=(None, None, 1))           # (H, W, 1)
#
#     # Encoder
#     c1 = conv_block(inputs, n_filters_start)
#     p1 = layers.MaxPooling2D()(c1)
#
#     c2 = conv_block(p1, n_filters_start * 2)
#     p2 = layers.MaxPooling2D()(c2)
#
#     c3 = conv_block(p2, n_filters_start * 4)
#     p3 = layers.MaxPooling2D()(c3)
#
#     c4 = conv_block(p3, n_filters_start * 8)
#     p4 = layers.MaxPooling2D()(c4)
#
#     # Bottleneck
#     bn = conv_block(p4, n_filters_start * 16)
#
#     # Decoder
#     u4 = layers.Conv2DTranspose(n_filters_start * 8, 2, strides=2,
#                                 padding='same')(bn)
#     u4 = layers.concatenate([u4, c4])
#     c5 = conv_block(u4, n_filters_start * 8)
#
#     u3 = layers.Conv2DTranspose(n_filters_start * 4, 2, strides=2,
#                                 padding='same')(c5)
#     u3 = layers.concatenate([u3, c3])
#     c6 = conv_block(u3, n_filters_start * 4)
#
#     u2 = layers.Conv2DTranspose(n_filters_start * 2, 2, strides=2,
#                                 padding='same')(c6)
#     u2 = layers.concatenate([u2, c2])
#     c7 = conv_block(u2, n_filters_start * 2)
#
#     u1 = layers.Conv2DTranspose(n_filters_start, 2, strides=2,
#                                 padding='same')(c7)
#     u1 = layers.concatenate([u1, c1])
#     c8 = conv_block(u1, n_filters_start)
#
#     outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)
#     return models.Model(inputs, outputs, name="U-Net")
#
#
# # ---------------------------------------------------------------------
# # 5. MAIN (load → split → train → save)
# # ---------------------------------------------------------------------
# def main():
#     # Optional: make TensorFlow deterministic & quieter
#     tf.keras.utils.set_random_seed(RANDOM_SEED)
#     tf.config.experimental.enable_tensor_float_32_execution(False)
#
#     print("Loading patches …")
#     X, y = load_pairs(IMG_DIR, MASK_DIR)
#     print(f"X: {X.shape}, y: {y.shape}")
#
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
#
#     model = build_unet()
#     model.compile(optimizer=tf.keras.optimizers.Adam(LR),
#                   loss="binary_crossentropy",
#                   metrics=["accuracy",
#                            tf.keras.metrics.MeanIoU(num_classes=2)])
#
#     model.summary()
#
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             MODEL_OUT, save_best_only=True,
#             monitor='val_loss', mode='min'),           # .keras → OK
#         tf.keras.callbacks.EarlyStopping(
#             patience=6, restore_best_weights=True)
#     ]
#
#     print("\nTraining …")
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=EPOCHS,
#         batch_size=BATCH,
#         shuffle=True,
#         callbacks=callbacks)
#
#     # Save final weights too (the checkpoint already saved best val-loss)
#     model.save(MODEL_OUT)
#     print(f"\n✓ Model saved to {MODEL_OUT}")
#
#
# if __name__ == "__main__":
#     main()

###########################3
import numpy as np
from PIL import Image
import tensorflow as tf

# --- CONFIG ---
MODEL_PATH = "unet_cell_patches.keras"
IMAGE_PATH = r"C:\Users\sharo\Desktop\Odd_test\ht1080_04_s10\Raw\040227.tif"
PATCH_SIZE = 256
OUT_PATH   =  r"C:\Users\sharo\Desktop\Odd_test\ht1080_04_s10\Raw\040227_predmask.png"

# --- Load Model ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Load Large Image ---
img = Image.open(IMAGE_PATH).convert('L')
img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
H, W = img_np.shape

# --- Prepare blank output mask ---
full_mask = np.zeros((H, W), dtype=np.uint8)

# --- Slide window over image ---
for y in range(0, H, PATCH_SIZE):
    for x in range(0, W, PATCH_SIZE):
        patch = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        pad_h = PATCH_SIZE - patch.shape[0]
        pad_w = PATCH_SIZE - patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='constant')
        patch_input = patch[np.newaxis, ..., np.newaxis]
        pred = model.predict(patch_input)
        pred_mask = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
        # Remove padding before placing
        pred_mask = pred_mask[:patch.shape[0]-pad_h if pad_h>0 else PATCH_SIZE,
                              :patch.shape[1]-pad_w if pad_w>0 else PATCH_SIZE]
        full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_mask[:H-y, :W-x]

# --- Save full mask ---
Image.fromarray(full_mask).save(OUT_PATH)
print(f"Saved mask to {OUT_PATH}")
