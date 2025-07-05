import os


# Use relative paths from the project root
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


project_root = os.path.dirname(os.path.abspath(__file__))

# Segmentation paths
segmentation_folder = os.path.join(project_root, 'segmentation')
united_csv_path = os.path.join(project_root, 'segmentation', 'united.csv')

# Model paths
models_folder = os.path.join(project_root, 'models')
train_folder = os.path.join(project_root, 'train')
model_path = os.path.join(models_folder, 'unet_cell_patches.keras')
predict_model_path = os.path.join(models_folder, 'unet_model_RGB_S.h5')

# Training dataset paths
train_dataset_folder = os.path.join(train_folder, 'dataset')
train_csv_path = os.path.join(train_dataset_folder, 'united.csv')
train_data_dir = train_dataset_folder

# Training configuration
IMG_SIZE = 100
LEARNING_RATE = 1e-4
SEED = 42
BATCH_SIZE = 16
EPOCHS = 10

# Segmentation configuration
PIXEL_THRESHOLD = 0
