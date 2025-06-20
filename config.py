import os


# Use relative paths from the project root
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


project_root = os.path.dirname(os.path.abspath(__file__))

segmentation_folder = os.path.join(project_root, 'segmentation')

model_path = os.path.join(project_root, 'models', 'unet_cell_patches.keras')

predict_model_path = os.path.join(project_root, 'models', 'unet_model_RGB_S.h5')

united_csv_path = os.path.join(project_root, 'segmentation', 'united.csv')

PIXEL_THRESHOLD = 0
