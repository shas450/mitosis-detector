import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

model_path = r'C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\models\unet_model_RGB_4.h5'
IMG_SIZE = 100  # Change if needed


def predict_and_save_masks(csv_path, dataset_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    df = pd.read_csv(csv_path)
    if 'N_Positive' not in df.columns:
        df['N_Positive'] = np.nan

    for i, row in df.iterrows():
        uuid_path = os.path.join(dataset_path, str(row['UUID']))
        img_path = os.path.join(uuid_path, "combined_image.png")
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_size = (img.shape[1], img.shape[0])
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        pred = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
        mask_pred = (pred > 0.5).astype(np.uint8) * 255
        mask_pred = cv2.resize(mask_pred, orig_size)

        save_path = os.path.join(uuid_path, "pred_mask.png")
        cv2.imwrite(save_path, mask_pred)

        n_positive = int(np.sum(mask_pred == 255))
        df.at[i, 'N_Positive'] = n_positive

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"{i + 1}/{len(df)} masks predicted and saved.")

    df.to_csv(csv_path, index=False)
    print(f"🧪 All test masks predicted and N_Positive updated in {csv_path}.")


# Example usage:
# predict_and_save_masks(r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation\s540001_cells.csv",
#                        r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation\s540001")
