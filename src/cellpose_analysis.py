import numpy as np
import pandas as pd
from cellpose import models, io
import os


def run_cellpose_analysis(input_path):
    # Load Cellpose model
    model = models.Cellpose(model_type='cyto3')  # Change model type if needed

    # Load image(s)
    images = io.imread(input_path)  # This handles both single images and stacks

    # Run Cellpose on the images
    masks, flows, styles, diams = model.eval(images, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)

    # Extract cell positions
    cell_data = []
    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells > 0]  # Ignore background (0)

    for cell_id in unique_cells:
        y, x = np.where(masks == cell_id)  # Get all pixels belonging to this cell
        centroid_x = int(np.mean(x))
        centroid_y = int(np.mean(y))
        cell_data.append([cell_id, centroid_x, centroid_y])

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(cell_data, columns=['cell_id', 'X', 'Y'])
    file_name = os.path.basename(input_path).split('.')[0]

    output_csv_path = f'./output/{file_name}_cells.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    return output_csv_path
