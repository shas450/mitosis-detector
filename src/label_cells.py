import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

image_size = 100  # Change this if your patch size is different!

def label_mitosis(csv_path, dataset_path):
    # Load only mitosis candidates
    data = pd.read_csv(csv_path)
    data = data[data["N_Positive"] > 0].reset_index(drop=True)

    # Skip cells with small masks
    keep_indices = []
    for idx, row in data.iterrows():
        uuid_folder = row['UUID']
        mask_path = os.path.join(dataset_path, str(uuid_folder), 'pred_mask.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.sum(mask > 0) >= 20:
                keep_indices.append(idx)
        else:
            print(f"Warning: {mask_path} does not exist. Skipping.")

    data = data.loc[keep_indices].reset_index(drop=True)

    if len(data) == 0:
        print("No mitosis candidates found with mask > 20 pixels.")
        return

    data['user_label'] = None  # Initialize label column

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    image_display = ax.imshow(np.zeros((image_size, image_size * 2, 3), dtype=np.uint8))
    ax.axis('off')

    result = {'index': 0}

    def update_display(index):
        row = data.iloc[index]
        uuid_folder = row['UUID']
        folder_path = os.path.join(dataset_path, str(uuid_folder))
        images = []

        for i in range(2):
            image_path = os.path.join(folder_path, f"{i}.tif")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {image_path}. Adding a placeholder.")
                image = 128 * np.ones((image_size, image_size, 3), dtype=np.uint8)
            else:
                # Draw a green circle at the center
                cv2.circle(image, (image_size // 2, image_size // 2), 10, (0, 255, 0), 1)
                image = cv2.resize(image, (image_size, image_size))
                cv2.putText(image, f"Frame {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            images.append(image)

        concatenated = cv2.hconcat(images)
        concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
        image_display.set_data(concatenated_rgb)
        ax.set_title(f"UUID: {uuid_folder} | Cell {index+1} of {len(data)}")
        fig.canvas.draw_idle()

    def label_and_advance(label_value):
        idx = result['index']
        data.at[idx, 'user_label'] = label_value
        print(f"Labeled index {idx}: {label_value}")
        result['index'] += 1
        if result['index'] < len(data):
            update_display(result['index'])
        else:
            print("All samples labeled.")
            plt.close()

    def on_yes(event):
        label_and_advance(1)

    def on_no(event):
        label_and_advance(0)

    # Add Yes/No buttons
    ax_yes = plt.axes([0.3, 0.05, 0.1, 0.075])
    btn_yes = Button(ax_yes, 'Yes')
    btn_yes.on_clicked(on_yes)

    ax_no = plt.axes([0.6, 0.05, 0.1, 0.075])
    btn_no = Button(ax_no, 'No')
    btn_no.on_clicked(on_no)

    update_display(result['index'])
    plt.show()

    # Save user_label to the CSV (merge on UUID)
    full_data = pd.read_csv(csv_path)
    if 'user_label' in full_data.columns:
        full_data = full_data.drop(columns=['user_label'])
    full_data = full_data.merge(
        data[['UUID', 'user_label']],
        on='UUID',
        how='left'
    )
    # If original had user_label, new will be user_label_y
    if 'user_label_y' in full_data.columns:
        full_data = full_data.drop(columns=['user_label_x'])
        full_data = full_data.rename(columns={'user_label_y': 'user_label'})
    full_data.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")
