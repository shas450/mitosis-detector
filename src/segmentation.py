import os
import uuid
import pandas as pd
from PIL import Image


def process_segmentation(cell_csv, image_path):
    # Load the input CSV file
    cell_data = pd.read_csv(cell_csv)

    # Ensure 'cell_id' column is numeric
    cell_data['cell_id'] = pd.to_numeric(cell_data['cell_id'], errors='coerce')

    # Determine segmentation folder
    segmentation_folder = cell_csv.split("_cells")[0]
    os.makedirs(segmentation_folder, exist_ok=True)

    # Generate second image path by incrementing the number in the filename
    dir_path, filename = os.path.split(image_path)
    base_name, ext = os.path.splitext(filename)

    # if not base_name[1:].isdigit():
    if not base_name.isdigit():
        print(f"Invalid filename format: {filename}")
        return

    # next_image_path = os.path.join(dir_path, f"s{int(base_name[1:]) + 1:06d}{ext}")
    next_image_path = os.path.join(dir_path, f"{int(base_name) + 1:06d}{ext}")

    # Process each row in cell_data.csv
    for index, row in cell_data.iterrows():
        cell_id = row['cell_id']
        if pd.isna(cell_id):
            print(f"Skipping row {index}: No cell ID found.")
            continue

        x_center = int(row['X'])
        y_center = int(row['Y'])

        # Generate UUID and create a folder
        folder_id = str(uuid.uuid4())
        cell_folder = os.path.join(segmentation_folder, folder_id)
        os.makedirs(cell_folder, exist_ok=True)
        cell_data.at[index, 'UUID'] = folder_id  # Store UUID in CSV

        # Function to process and save the image
        def crop_and_save(img_path, save_name):
            if not os.path.exists(img_path):
                print(f"Image {img_path} does not exist for cell ID {cell_id}.")
                return

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    left = max(0, x_center - 50)
                    upper = max(0, y_center - 50)
                    right = min(width, x_center + 50)
                    lower = min(height, y_center + 50)

                    cropped_img = img.crop((left, upper, right, lower))
                    padded_img = Image.new("L", (100, 100), 0)
                    paste_x = max(0, 50 - (x_center - left))
                    paste_y = max(0, 50 - (y_center - upper))
                    padded_img.paste(cropped_img, (paste_x, paste_y))

                    padded_img.save(os.path.join(cell_folder, save_name))
            except Exception as e:
                print(f"Error processing image {img_path} for cell ID {cell_id}: {e}")

        # Process both images
        crop_and_save(image_path, "0.tif")
        crop_and_save(next_image_path, "1.tif")

    # Save the modified CSV
    cell_data.to_csv(cell_csv, index=False)
    print(f"Updated CSV saved to {cell_csv}")
    return segmentation_folder
