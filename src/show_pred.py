import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from src.segmentation import infer_next_frame_path
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def pad_to_size(img, target_height, target_width):
    h, w = img.shape[:2]
    if img.ndim == 2:
        padded = np.zeros((target_height, target_width), dtype=img.dtype)
        padded[:h, :w] = img
    else:
        padded = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
        padded[:h, :w, :] = img
    return padded

def overlay_all_predicted_masks_on_image_both_frames(
    image_path, csv_path, dataset_path,
    window_size=300, step_size=300, mask_size=100, alpha=0.3
):
    # Load frames and masks
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    global_mask_current = np.zeros((H, W), dtype=np.uint8)
    global_mask_next = np.zeros((H, W), dtype=np.uint8)
    df = pd.read_csv(csv_path)
    next_frame_path = infer_next_frame_path(image_path)
    if os.path.exists(next_frame_path):
        next_image = np.array(Image.open(next_frame_path).convert("RGB"))
    else:
        next_image = np.zeros_like(image)
        print(f"Warning: Next frame {next_frame_path} not found.")

    # Compose the mask overlays
    for idx, row in df.iterrows():
        uuid, x, y = row['UUID'], int(row['X']), int(row['Y'])
        uuid_folder = os.path.join(dataset_path, str(uuid))
        mask_path = os.path.join(uuid_folder, 'pred_mask.png')
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        # Only show mitosis masks with MORE than 200 positive pixels
        if np.sum(mask) <= 20:
            continue
        cx, cy = x, y
        half = mask_size // 2
        y1 = max(cy - half, 0)
        y2 = min(cy + half, H)
        x1 = max(cx - half, 0)
        x2 = min(cx + half, W)
        my1 = half - (cy - y1)
        my2 = mask_size - (cy + half - y2)
        mx1 = half - (cx - x1)
        mx2 = mask_size - (cx + half - x2)
        mask_patch = mask[my1:my2, mx1:mx2]
        roi_current = global_mask_current[y1:y2, x1:x2]
        roi_next = global_mask_next[y1:y2, x1:x2]
        h_patch, w_patch = mask_patch.shape
        h_roi, w_roi = roi_current.shape
        h = min(h_patch, h_roi)
        w = min(w_patch, w_roi)
        if h > 0 and w > 0:
            mask_patch = mask_patch[:h, :w]
            roi_current = roi_current[:h, :w]
            roi_next = roi_next[:h, :w]
            global_mask_current[y1:y1 + h, x1:x1 + w] = np.maximum(roi_current, mask_patch)
            global_mask_next[y1:y1 + h, x1:x1 + w] = np.maximum(roi_next, mask_patch)

    def blend(image, mask, alpha=0.3):
        out = image.copy().astype(float)
        mask_indices = mask.astype(bool)
        out[mask_indices] = (1 - alpha) * out[mask_indices] + alpha * np.array([255, 255, 0])
        return out.astype(np.uint8)

    blended_current = blend(image, global_mask_current, alpha)
    blended_next = blend(next_image, global_mask_next, alpha)

    windows = []
    for y_start in range(0, H, step_size):
        for x_start in range(0, W, step_size):
            y_end = min(y_start + window_size, H)
            x_end = min(x_start + window_size, W)
            crop_current = blended_current[y_start:y_end, x_start:x_end]
            crop_next = blended_next[y_start:y_end, x_start:x_end]
            crop_current = pad_to_size(crop_current, window_size, window_size)
            crop_next = pad_to_size(crop_next, window_size, window_size)
            windows.append((crop_current, crop_next, y_start, y_end, x_start, x_end))

    # --- Interactive viewer ---
    idx = [0]  # use a mutable object so we can modify inside functions

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)

    def update():
        crop_current, crop_next, y_start, y_end, x_start, x_end = windows[idx[0]]
        axs[0].imshow(crop_current)
        axs[0].axis('off')
        axs[0].set_title(f'Current Frame [{y_start}:{y_end}, {x_start}:{x_end}]')
        axs[1].imshow(crop_next)
        axs[1].axis('off')
        axs[1].set_title(f'Next Frame [{y_start}:{y_end}, {x_start}:{x_end}]')
        fig.suptitle(f'Window {idx[0] + 1} / {len(windows)}')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ['right', ' ', 'enter']:
            if idx[0] < len(windows) - 1:
                idx[0] += 1
                update()
        elif event.key == 'left':
            if idx[0] > 0:
                idx[0] -= 1
                update()

    fig.canvas.mpl_connect('key_press_event', on_key)

    update()
    plt.show()

# Usage example:
# overlay_all_predicted_masks_on_image_both_frames(
#     image_path=r"...",
#     csv_path=r"...",
#     dataset_path=r"..."
# )
