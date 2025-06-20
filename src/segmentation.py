import re
import time
from scipy.ndimage import label, center_of_mass
import tensorflow as tf
import numpy as np
import pandas as pd
from cellpose import models, io
import os
from PIL import Image
import uuid


def infer_next_frame_path(frame0_path):
    """Given a frame0 path, return the path to the next frame (with +1 number)."""
    basename = os.path.basename(frame0_path)
    match = re.search(r'(\d+)', basename)
    if not match:
        raise ValueError("No number found in frame0 filename!")
    frame_num = int(match.group(1))
    next_num = frame_num + 1
    next_name = basename.replace(str(frame_num).zfill(len(match.group(1))),
                                 str(next_num).zfill(len(match.group(1))))
    return os.path.join(os.path.dirname(frame0_path), next_name)


def predict_mask(model, img, patch_size=256):
    """Predict segmentation mask on a large image using a sliding window."""
    img_np = img.astype(np.float32) / 255.0
    H, W = img_np.shape
    full_mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patch = img_np[y:y + patch_size, x:x + patch_size]
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            if pad_h > 0 or pad_w > 0:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='constant')
            patch_input = patch[np.newaxis, ..., np.newaxis]
            pred = model.predict(patch_input, verbose=0)
            pred_mask = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
            if pad_h > 0 or pad_w > 0:
                pred_mask = pred_mask[:patch.shape[0] - pad_h if pad_h > 0 else patch_size,
                            :patch.shape[1] - pad_w if pad_w > 0 else patch_size]
            full_mask[y:y + patch_size, x:x + patch_size] = pred_mask[:H - y, :W - x]
    return full_mask


def find_cell_centers(mask):
    """Detect cells (blobs) in a binary mask. Returns list of (cy, cx) floats."""
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_cells = label(mask > 127, structure=structure)
    centers = center_of_mass(mask, labeled, range(1, n_cells + 1))
    return centers



def extract_patch(img, cy, cx, crop_size):
    """Extracts a centered square patch with padding if needed."""
    y1 = cy - crop_size // 2
    x1 = cx - crop_size // 2
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - img.shape[0])
    pad_right = max(0, x2 - img.shape[1])

    y1 = max(y1, 0)
    x1 = max(x1, 0)
    y2 = min(y2, img.shape[0])
    x2 = min(x2, img.shape[1])

    patch = img[y1:y2, x1:x2]
    if pad_top or pad_left or pad_bottom or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return patch

def create_combined_image(patch0, patch1):
    """Creates a combined RGB image from two grayscale patches."""
    arr0 = patch0.astype(np.float32)
    arr1 = patch1.astype(np.float32)
    if arr0.max() > 0:
        arr0 = (arr0 / arr0.max()) * 255
    if arr1.max() > 0:
        arr1 = (arr1 / arr1.max()) * 255
    arr0 = arr0.astype(np.uint8)
    arr1 = arr1.astype(np.uint8)
    rgb = np.zeros((arr0.shape[0], arr0.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = arr1  # green
    rgb[..., 2] = arr0  # blue
    return rgb

def save_cell_patches_and_combined(patch0, patch1, cell_dir):
    """Saves patch0, patch1, and combined image in the given folder."""
    os.makedirs(cell_dir, exist_ok=True)
    path0 = os.path.join(cell_dir, "0.tif")
    path1 = os.path.join(cell_dir, "1.tif")
    Image.fromarray(patch0).save(path0)
    Image.fromarray(patch1).save(path1)
    combined_rgb = create_combined_image(patch0, patch1)
    combined_path = os.path.join(cell_dir, "combined_image.png")
    Image.fromarray(combined_rgb).save(combined_path)

def crop_and_save_cells(frame0, frame1, centers, output_dir, frame_name, crop_size=100):
    """Crop cell-centered patches from frames, save as UUID folders, and create combined images. Returns CSV records."""
    csv_records = []
    total_cells = len(centers)
    for idx, (cy, cx) in enumerate(centers):
        cy, cx = int(round(cy)), int(round(cx))
        patch0 = extract_patch(frame0, cy, cx, crop_size)
        patch1 = extract_patch(frame1, cy, cx, crop_size)
        cell_uuid = str(uuid.uuid4())
        cell_dir = os.path.join(output_dir, cell_uuid)
        save_cell_patches_and_combined(patch0, patch1, cell_dir)
        csv_records.append({
            "UUID": cell_uuid,
            "X": cx,
            "Y": cy,
            "frame": frame_name
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == total_cells:
            print(f"{idx + 1}/{total_cells} cells processed.")
    return csv_records


model_path = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\models\unet_cell_patches.keras"
segmentation_folder = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation"
crop_size = 100
patch_size = 256


def segment_and_crop(input_path):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(segmentation_folder, file_name)
    os.makedirs(output_dir, exist_ok=True)

    frame1_path = infer_next_frame_path(input_path)
    frame0 = np.array(Image.open(input_path).convert('L'))
    frame1 = np.array(Image.open(frame1_path).convert('L'))

    # Load model (assumes global model_path, or pass as arg)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Predict mask and find centers
    mask = predict_mask(model, frame0, patch_size=patch_size)
    centers = find_cell_centers(mask)
    print(f"Detected {len(centers)} cells.")

    # Crop, save, and record info
    csv_records = crop_and_save_cells(frame0, frame1, centers, output_dir, file_name, crop_size)

    # Save CSV
    output_csv_path = os.path.join(segmentation_folder, f"{file_name}_cells.csv")
    pd.DataFrame(csv_records).to_csv(output_csv_path, index=False)
    time.sleep(1)
    print(f"Results saved to {output_csv_path}")
    return output_csv_path, output_dir



def segment_and_crop_cellpose(input_path, crop_size=100, output_base='./output'):
    """Run Cellpose, extract and crop cells, just like custom model."""
    model = models.Cellpose(model_type='cyto3')
    images = io.imread(input_path)
    masks, flows, styles, diams = model.eval(images, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)

    # Find frame1 for combined patch
    base, ext = os.path.splitext(input_path)
    match = re.search(r'(\d+)', base)
    if not match:
        raise ValueError("No number found in filename")
    frame_num = int(match.group(1))
    next_frame_path = base.replace(str(frame_num).zfill(len(match.group(1))), str(frame_num+1).zfill(len(match.group(1)))) + ext
    if not os.path.exists(next_frame_path):
        raise FileNotFoundError(f"Next frame {next_frame_path} does not exist.")
    frame0 = np.array(Image.open(input_path).convert('L'))
    frame1 = np.array(Image.open(next_frame_path).convert('L'))

    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells > 0]
    csv_records = []
    file_name = os.path.basename(input_path).split('.')[0]
    output_dir = os.path.join(output_base, file_name)
    os.makedirs(output_dir, exist_ok=True)

    for cell_id in unique_cells:
        y, x = np.where(masks == cell_id)
        centroid_x = int(np.mean(x))
        centroid_y = int(np.mean(y))
        patch0 = extract_patch(frame0, centroid_y, centroid_x, crop_size)
        patch1 = extract_patch(frame1, centroid_y, centroid_x, crop_size)
        cell_uuid = str(uuid.uuid4())
        cell_dir = os.path.join(output_dir, cell_uuid)
        save_cell_patches_and_combined(patch0, patch1, cell_dir)
        csv_records.append({
            "UUID": cell_uuid,
            "X": centroid_x,
            "Y": centroid_y,
            "frame": file_name
        })

    output_csv_path = os.path.join(output_base, f"{file_name}_cells.csv")
    pd.DataFrame(csv_records).to_csv(output_csv_path, index=False)
    print(f"Cellpose results saved to {output_csv_path}")
    return output_csv_path, output_dir