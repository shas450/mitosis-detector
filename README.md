# MitosisDetector

MitosisDetector is a Python application for automated detection, segmentation, and labeling of mitotic cells in microscopy images. It provides a graphical user interface (GUI) for batch processing images, supports both custom deep learning models and Cellpose for segmentation, and enables interactive review and labeling of predicted mitosis candidates.

## Features

- **Batch Image Processing:** Select and process multiple microscopy images at once.
- **Segmentation Methods:** Choose between a custom U-Net-based model or Cellpose for cell segmentation.
- **Patch Extraction:** Automatically crops cell-centered patches from consecutive frames.
- **Mitosis Prediction:** Predicts mitosis masks using a trained model.
- **Interactive Visualization:** Review segmentation and mitosis predictions with overlay and windowed navigation.
- **Manual Labeling:** Label mitosis candidates via an interactive GUI.
- **Results Export:** Saves all results and labels to CSV files for further analysis.

## Project Structure


- **config.py:** Centralized configuration for paths and constants.
- **main.py:** Entry point for launching the GUI.
- **show_ui.py:** Implements the main GUI for image selection, processing, and progress tracking.
- **src/segmentation.py:** Functions for segmenting images and extracting cell patches.
- **src/predict_mitosis.py:** Loads the mitosis prediction model and generates masks.
- **src/show_pred.py:** Visualization utilities for overlaying predictions and navigating image windows.
- **src/label_cells.py:** Interactive labeling tool for reviewing and annotating mitosis candidates.
- **frames/, models/, segmentation/:** Data, model, and output directories.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/shas450/MitosisDetector.git
   cd MitosisDetector
   ```

2. **Set up a Python environment (Windows):**
   ```sh
   python -m venv env
   env\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Launch the application:**
   ```sh
   python main.py
   ```

2. **In the GUI:**
   - Select segmentation method (`Custom Model` or `Cellpose`).
   - Click "Select Images" and choose your microscopy images.
   - Click "Run" to start processing.
   - Review segmentation and mitosis predictions interactively.
   - Label mitosis candidates as needed.

3. **Training New Models:**
   After labeling your data, you can train new models:
   
   a) **Prepare the training dataset:**
   ```sh
   cd train
   python prepare_dataset.py
   ```
   
   b) **Train the U-Net model:**
   ```sh
   python U_net_model_RGB.py
   ```
   
   This will create a new trained model in the `train/` folder with a timestamp.

## Training Workflow

The application supports continuous improvement through retraining:

1. **Label Data:** Use the GUI to label mitosis candidates in your images
2. **Export Labels:** Labels are saved in `segmentation/united.csv`
3. **Prepare Dataset:** Run `train/prepare_dataset.py` to copy data to training folder
4. **Train Model:** Run `train/U_net_model_RGB.py` to train a new model
5. **Update Config:** Update `config.py` to point to your new model for inference

See [`train/README.md`](train/README.md) for detailed training instructions.


## Configuration

Edit [`config.py`](config.py) to adjust paths and parameters as needed for your environment.

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies.

## Notes

- The application expects images to be named with frame numbers (e.g., `s550000.tif`, `s550001.tif`, ...).
- **id you analaze frame 00XX  make sure that farme 00X X+1 in the same folder!** 
- Output CSVs and segmentation results are saved in the [`segmentation`](segmentation) directory.
- For best results, ensure your models are compatible with the expected input size and format.
