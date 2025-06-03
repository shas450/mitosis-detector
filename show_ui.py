import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import threading
import matplotlib.pyplot as plt
from src.label_cells import label_mitosis
from src.segmentation import segment_and_crop
from src.predict_mitosis import predict_and_save_masks
from src.show_pred import overlay_all_predicted_masks_on_image_both_frames


def show_ui():
    file_path = None  # Store selected file path

    def process_image():
        status_label.config(text="🟡 Creating Segmentation...")
        root.update()
        # csv_path, dataset_path = segment_and_crop(file_path)

        status_label.config(text="✅ Segmentation Completed\n🟡 Predicting mitosis...")
        root.update()
        # predict_and_save_masks(csv_path,dataset_path)

        file_path = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\frames\s540001.tif"
        csv_path = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation\s540001_cells.csv"
        dataset_path = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation\s540001"

        status_label.config(text="✅ All processing completed!\n🖼️ Displaying image...")
        root.update()
        overlay_all_predicted_masks_on_image_both_frames(file_path, csv_path, dataset_path)

        # label_mitosis(csv_path, dataset_path)

    def select_image():
        nonlocal file_path
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif")])
        if file_path:
            status_label.config(text=f"📂 Selected: {file_path.split('/')[-1]}")
            run_button.config(state=tk.NORMAL)

    def start_processing():
        run_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=process_image)
        thread.start()

    # Create UI
    root = tk.Tk()
    root.title("Image Processing UI")
    root.geometry("450x250")
    root.configure(bg="#f0f0f0")

    frame = Frame(root, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, bd=2)
    frame.pack(pady=20)

    select_button = Button(frame, text="📁 Select Image", command=select_image, font=("Arial", 12), bg="#4CAF50",
                           fg="white", padx=10, pady=5)
    select_button.pack(pady=10)

    status_label = Label(frame, text="No image selected", font=("Arial", 10), bg="#ffffff", fg="#333")
    status_label.pack(pady=10)

    run_button = Button(frame, text="▶ Run", command=start_processing, font=("Arial", 12), bg="#008CBA", fg="white",
                        padx=10, pady=5, state=tk.DISABLED)
    run_button.pack(pady=10)

    root.mainloop()
