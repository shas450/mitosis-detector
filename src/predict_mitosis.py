import os
import pandas as pd
import numpy as np
from PIL import Image
import pickle


def predict_mitosis(dataset_path, csv_path):
    # Paths
    model_path = r'./siamese_model.pkl'

    # Load the saved model
    with open(model_path, 'rb') as file:
        siamese_model = pickle.load(file)

    # Load the output_data.csv
    track_id_data = pd.read_csv(csv_path)

    # Function to load and preprocess an image
    def preprocess_image(image_path):
        with Image.open(image_path) as img:
            img = img.resize((100, 100))  # Resize the image to 100x100
            img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            if len(img.shape) == 2:  # If grayscale, add a channel dimension
                img = np.expand_dims(img, axis=-1)
            return img

    # Iterate through each UUID folder and predict mitosis
    for index, row in track_id_data.iterrows():
        uuid_folder = os.path.join(dataset_path, row['UUID'])

        # Ensure both images exist (0.tif and 1.tif)
        image_0_path = os.path.join(uuid_folder, '0.tif')
        image_1_path = os.path.join(uuid_folder, '1.tif')
        if not (os.path.exists(image_0_path) and os.path.exists(image_1_path)):
            print(f"Missing images in UUID folder {uuid_folder}. Skipping...")
            continue

        try:
            # Load and preprocess the images
            image_0 = preprocess_image(image_0_path)
            image_1 = preprocess_image(image_1_path)

            # Expand dimensions to match the input shape of the model
            image_0 = np.expand_dims(image_0, axis=0)
            image_1 = np.expand_dims(image_1, axis=0)

            # Predict using the model
            prediction = siamese_model.predict([image_0, image_1])
            confidence_score = prediction[0][0]  # Raw confidence score from the model
            mitosis_prediction = int(confidence_score > 0.5)  # Binary prediction (1 or 0)

            # Add predictions and confidence scores to the DataFrame
            track_id_data.at[index, 'Mitosis/Non-Mitosis'] = mitosis_prediction
            track_id_data.at[index, 'Confidence Score'] = confidence_score

        except Exception as e:
            print(f"Error processing UUID folder {uuid_folder}: {e}")

    # Save the updated output_data.csv with new columns
    track_id_data.to_csv(csv_path, index=False)
    print(f"Updated output_data.csv saved to {csv_path}.")
