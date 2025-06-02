import os

import matplotlib
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')  # or try 'QtAgg' if you have Qt installed


def show_images_side_by_side(image_path, csv_path):
    """Displays the original and next image side by side with mitosis points."""
    dir_path, base_name_ext = os.path.split(image_path)
    base_name, ext = os.path.splitext(base_name_ext)

    try:
        # next_image_path = os.path.join(dir_path, f"s{int(base_name[1:]) + 1:06d}{ext}")
        next_image_path =os.path.join(dir_path, f"{int(base_name) + 1:06d}{ext}")
        # Load images
        img1 = Image.open(image_path)
        img2 = Image.open(next_image_path) if os.path.exists(next_image_path) else None

        # Load mitosis data
        # df = pd.read_csv(csv_path)
        # mitotic_cells = df[df["Mitosis/Non-Mitosis"] == 1]

        df = pd.read_csv(csv_path)
        mitotic_cells = df[df["user_label"] == 1]

        # Plot both images
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Even larger figure size

        # Adjust margins to maximize image area
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0, wspace=0.05)

        # First image
        axes[0].imshow(img1, cmap='gray', aspect='auto')
        axes[0].scatter(mitotic_cells["X"], mitotic_cells["Y"], color='red', marker='o', s=5, label="Mitosis")
        axes[0].set_title(f"Mitosis Predictions: {os.path.basename(image_path)}", fontsize=16)
        axes[0].axis("off")

        # Second image (if exists)
        if img2:
            axes[1].imshow(img2, cmap='gray', aspect='auto')
            axes[1].set_title(f"Next Image: {os.path.basename(next_image_path)}", fontsize=16)
            axes[1].scatter(mitotic_cells["X"], mitotic_cells["Y"], color='red', marker='o', s=5,
                            label="Mitosis")  # Bigger dots

            axes[1].axis("off")
        else:
            axes[1].set_visible(False)  # Hide the second subplot if no next image

        plt.show()
        return   # Move to the next step when closed


    except ValueError:
        pass