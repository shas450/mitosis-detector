import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from src.segmentation import infer_next_frame_path
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

window_size = 300
step_size = 300
mask_size = 100
alpha = 0.3


def pad_to_size(img, target_height, target_width):
    h, w = img.shape[:2]
    if img.ndim == 2:
        padded = np.zeros((target_height, target_width), dtype=img.dtype)
        padded[:h, :w] = img
    else:
        padded = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
        padded[:h, :w, :] = img
    return padded


def load_current_and_next_frames(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    next_frame_path = infer_next_frame_path(image_path)
    if os.path.exists(next_frame_path):
        next_image = np.array(Image.open(next_frame_path).convert("RGB"))
    else:
        next_image = np.zeros_like(image)
        print(f"Warning: Next frame {next_frame_path} not found.")
    return image, next_image


def generate_combined_masks(df, dataset_path, image_shape, global_mask_current, global_mask_next):
    H, W = image_shape[:2]
    for _, row in df.iterrows():
        uuid, x, y = row['UUID'], int(row['X']), int(row['Y'])
        mask_path = os.path.join(dataset_path, str(uuid), 'pred_mask.png')
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if np.sum(mask) <= 20:
            continue

        cx, cy = x, y
        half = mask_size // 2
        y1, y2 = max(cy - half, 0), min(cy + half, H)
        x1, x2 = max(cx - half, 0), min(cx + half, W)
        my1, my2 = half - (cy - y1), mask_size - (cy + half - y2)
        mx1, mx2 = half - (cx - x1), mask_size - (cx + half - x2)
        mask_patch = mask[my1:my2, mx1:mx2]

        for global_mask in [global_mask_current, global_mask_next]:
            roi = global_mask[y1:y2, x1:x2]
            h, w = min(mask_patch.shape[0], roi.shape[0]), min(mask_patch.shape[1], roi.shape[1])
            if h > 0 and w > 0:
                global_mask[y1:y1 + h, x1:x1 + w] = np.maximum(
                    roi[:h, :w], mask_patch[:h, :w]
                )
    return global_mask_current, global_mask_next


def blend_overlay(image, mask, alpha=0.3):
    out = image.copy().astype(float)
    out[mask.astype(bool)] = (1 - alpha) * out[mask.astype(bool)] + alpha * np.array([255, 255, 0])
    return out.astype(np.uint8)


def crop_windows_from_images(img1, img2, step, window):
    H, W = img1.shape[:2]
    windows = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end, x_end = min(y + window, H), min(x + window, W)
            crop1 = pad_to_size(img1[y:y_end, x:x_end], window, window)
            crop2 = pad_to_size(img2[y:y_end, x:x_end], window, window)
            windows.append((crop1, crop2, y, y_end, x, x_end))
    return windows


def launch_interactive_viewer(windows,image_name):
    idx = [0]
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
        fig.suptitle(f'{image_name}: Window {idx[0] + 1} / {len(windows)}')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ['right', ' ', 'enter'] and idx[0] < len(windows) - 1:
            idx[0] += 1
            update()
        elif event.key == 'left' and idx[0] > 0:
            idx[0] -= 1
            update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()


def visualize_predicted_mitosis_masks(image_path, csv_path, dataset_path):
    image, next_image = load_current_and_next_frames(image_path)
    H, W = image.shape[:2]
    global_mask_current = np.zeros((H, W), dtype=np.uint8)
    global_mask_next = np.zeros((H, W), dtype=np.uint8)
    df = pd.read_csv(csv_path)

    global_mask_current, global_mask_next = generate_combined_masks(
        df, dataset_path, image.shape, global_mask_current, global_mask_next
    )

    blended_current = blend_overlay(image, global_mask_current, alpha)
    blended_next = blend_overlay(next_image, global_mask_next, alpha)

    windows = crop_windows_from_images(blended_current, blended_next, step_size, window_size)

    image_name = os.path.basename(image_path)
    launch_interactive_viewer(windows,image_name)
