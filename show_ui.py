import os
import threading
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, OptionMenu, StringVar

from src.predict_mitosis import predict_and_save_masks
from src.segmentation import segment_and_crop, segment_and_crop_cellpose
from src.show_pred import overlay_all_predicted_masks_on_image_both_frames
from src.label_cells import label_mitosis

def show_ui():
    file_paths = []  # Store multiple file paths
    segmentation_folder = r"C:\Users\sharo\Desktop\Odd_test\Mitosis_Detector\frames\segmentation"

    root = tk.Tk()
    root.title("Image Processing UI")
    root.geometry("600x380")
    root.configure(bg="#f0f0f0")

    segmentation_method = StringVar(root)
    segmentation_method.set("Custom Model")

    frame = Frame(root, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, bd=2)
    frame.pack(pady=20)

    method_label = Label(frame, text="Choose segmentation method:", font=("Arial", 11), bg="#ffffff", fg="#333")
    method_label.pack(pady=(0, 5))
    method_menu = OptionMenu(frame, segmentation_method, "Custom Model", "Cellpose")
    method_menu.config(font=("Arial", 12), bg="#efefef", fg="#333")
    method_menu.pack(pady=5)

    status_label = Label(frame, text="No images selected", font=("Arial", 10), bg="#ffffff", fg="#333")
    status_label.pack(pady=10)

    def select_images():
        nonlocal file_paths
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif")]
        )
        if file_paths:
            display_files = [os.path.basename(p) for p in file_paths]
            status_label.config(text=f"📂 Selected: {', '.join(display_files)}")
            run_button.config(state=tk.NORMAL)
        else:
            status_label.config(text="No images selected")
            run_button.config(state=tk.DISABLED)

    def process_images():
        run_button.config(state=tk.DISABLED)
        if not file_paths:
            status_label.config(text="No images selected!")
            return

        for file_path in file_paths:
            # ====== SEGMENTATION AND PREDICTION BLOCK (comment out) ======
            # try:
            #     status_label.config(text=f"🟡 Segmenting: {os.path.basename(file_path)}")
            #     root.update()
            #     if segmentation_method.get() == "Custom Model":
            #         csv_path, dataset_path = segment_and_crop(file_path)
            #     else:
            #         csv_path, dataset_path = segment_and_crop_cellpose(
            #             file_path, crop_size=100, output_base=segmentation_folder
            #         )
            # except Exception as e:
            #     status_label.config(text=f"❌ Segmentation failed for {os.path.basename(file_path)}: {e}")
            #     continue

            # status_label.config(text="✅ Segmentation Completed\n🟡 Predicting mitosis...")
            # root.update()
            # try:
            #     predict_and_save_masks(csv_path, dataset_path)
            # except Exception as e:
            #     status_label.config(text=f"❌ Prediction failed for {os.path.basename(file_path)}: {e}")
            #     continue

            # ====== INSTEAD, use this: ======
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            csv_path = os.path.join(segmentation_folder, f"{file_name}_cells.csv")
            dataset_path = os.path.join(segmentation_folder, file_name)

            status_label.config(text="🖼️ Displaying image...")
            root.update()
            try:
                overlay_all_predicted_masks_on_image_both_frames(file_path, csv_path, dataset_path)
                status_label.config(text=f"✅ Done! {os.path.basename(file_path)}: See displayed image.")
            except Exception as e:
                status_label.config(text=f"❌ Error showing image: {e}")

            try:
                label_mitosis(csv_path, dataset_path)
            except Exception as e:
                print(f"Labeling failed for {os.path.basename(file_path)}: {e}")
                status_label.config(text=f"❌ Labeling failed: {e}")

        run_button.config(state=tk.NORMAL)
        status_label.config(text="✅ All files processed!")

    def start_processing():
        thread = threading.Thread(target=process_images)
        thread.start()

    select_button = Button(
        frame, text="📁 Select Images", command=select_images, font=("Arial", 12),
        bg="#4CAF50", fg="white", padx=10, pady=5
    )
    select_button.pack(pady=10)

    run_button = Button(
        frame, text="▶ Run", command=start_processing, font=("Arial", 12),
        bg="#008CBA", fg="white", padx=10, pady=5, state=tk.DISABLED
    )
    run_button.pack(pady=10)

    root.mainloop()

# Call show_ui() in your main file to launch the UI

