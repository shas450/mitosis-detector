import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

image_size = 200

def label_mitosis(csv_path, dataset_path):
    # Load the CSV into a DataFrame
    data = pd.read_csv(csv_path)
    data = data[data["Mitosis/Non-Mitosis"] == 1].reset_index(drop=True)

    data['user_label'] = None  # Initialize with None

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    image_display = ax.imshow(np.zeros((image_size, image_size * 2, 3), dtype=np.uint8))
    ax.axis('off')

    result = {'index': 0, 'label': None}

    def update_display(index):
        row = data.iloc[index]
        uuid_folder = row['UUID']
        cell_id = row['cell_id']
        x, y = int(row['X']), int(row['Y'])

        folder_path = os.path.join(dataset_path, uuid_folder)
        images = []

        for i in range(2):
            image_path = os.path.join(folder_path, f"{i}.tif")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Unable to load image {image_path}. Adding a placeholder.")
                image = 128 * np.ones((image_size, image_size, 3), dtype=np.uint8)
            else:
                cv2.circle(image, (x, y), 20, (0, 255, 0), 1)
                image = cv2.resize(image, (image_size, image_size))
                cv2.putText(image, f"Frame {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            images.append(image)

        concatenated = cv2.hconcat(images)
        concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
        image_display.set_data(concatenated_rgb)
        ax.set_title(f"Cell ID: {cell_id} | UUID: {uuid_folder}")
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

    # Add buttons
    ax_yes = plt.axes([0.3, 0.05, 0.1, 0.075])
    btn_yes = Button(ax_yes, 'Yes')
    btn_yes.on_clicked(on_yes)

    ax_no = plt.axes([0.6, 0.05, 0.1, 0.075])
    btn_no = Button(ax_no, 'No')
    btn_no.on_clicked(on_no)

    update_display(result['index'])
    plt.show()

    # Save the updated CSV
    full_data = pd.read_csv(csv_path)
    if 'user_label' in full_data.columns:
        full_data = full_data.drop(columns=['user_label'])

    # Merge the labeled data back in
    full_data = full_data.merge(
        data[['UUID', 'cell_id', 'user_label']],
        on=['UUID', 'cell_id'],
        how='left'
    )

    # Save to CSV
    full_data.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")
