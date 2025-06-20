import os
import threading
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, OptionMenu, StringVar, ttk

import pandas as pd

from config import united_csv_path
from src.predict_mitosis import predict_and_save_masks
from src.segmentation import segment_and_crop, segment_and_crop_cellpose
from src.show_pred import visualize_predicted_mitosis_masks
from src.label_cells import label_mitosis

from tkinter import Listbox, Scrollbar, END


def select_images_dialog():
    return list(filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif")]))


def save_to_united_csv(csv_path, dataset_path, united_csv_path):
    df = pd.read_csv(csv_path)
    if os.path.exists(united_csv_path):
        df.to_csv(united_csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(united_csv_path, index=False)
    return dataset_path


def process_all_images(file_paths, method, status_label, root, progress_var):
    for idx, file_path in enumerate(file_paths, start=1):
        try:
            files_progress = f"{idx}/{len(file_paths)}"
            csv_path, dataset_path, file_name = process_single_image(file_path, files_progress, method, status_label,
                                                                     root, progress_var)
            if csv_path and os.path.exists(csv_path):
                save_to_united_csv(csv_path, dataset_path, united_csv_path)
        except Exception as e:
            update_status(status_label, root, f"❌ Failed processing image {file_path}: {e}")

    update_status(status_label, root, "🟡 Running overlay and labeling for all images...")

    for image_path in file_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_path = os.path.join("Segmentation", f"{image_name}_cells.csv")
        dataset_path = os.path.join("Segmentation", image_name)

        if not os.path.exists(csv_path):
            update_status(status_label, root, f"⚠️ CSV not found for {image_name}")
            continue

        visualize_predicted_mitosis_masks(image_path, csv_path, dataset_path)

    label_mitosis(united_csv_path)
    update_status(status_label, root, "✅ Labeling done.")


def process_single_image(file_path, files_progress, method, status_label, root, progress_var):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    update_status(status_label, root, f"🟡 [{files_progress}] Segmenting: {file_name}")

    try:
        if method == "Custom Model":
            csv_path, dataset_path = segment_and_crop(file_path)
        else:
            csv_path, dataset_path = segment_and_crop_cellpose(file_path, progress_var=progress_var, root=root)

        update_status(status_label, root, f"✅ [{files_progress}] Segmentation Completed\n🟡 Predicting mitosis...")
        predict_and_save_masks(csv_path, dataset_path, progress_var=progress_var, root=root)
    except Exception as e:
        update_status(status_label, root, f"❌ [{files_progress}] Error processing {file_name}: {e}")
        return None, None, None

    return csv_path, dataset_path, file_name


def update_status(label, root, text):
    label.config(text=text)
    root.update()


def handle_select_images(status_label, run_button, file_paths, listbox):
    selected = select_images_dialog()
    file_paths.clear()
    file_paths.extend(selected)
    listbox.delete(0, END)
    if selected:
        for p in selected:
            listbox.insert(END, os.path.basename(p))
        status_label.config(text=f"📂 {len(selected)} image(s) selected")
        run_button.config(state=tk.NORMAL)
    else:
        status_label.config(text="No images selected")
        run_button.config(state=tk.DISABLED)


def handle_process_images(file_paths, method_var, status_label, run_button, root, progress_var):
    run_button.config(state=tk.DISABLED)
    if not file_paths:
        status_label.config(text="No images selected!")
        return

    update_status(status_label, root, f"🟡 Processing {len(file_paths)} image(s)...")
    process_all_images(file_paths, method_var.get(), status_label, root, progress_var)
    run_button.config(state=tk.NORMAL)


def build_ui():
    file_paths = []
    root = tk.Tk()
    root.title("Image Processing UI")
    root.geometry("800x450")
    root.configure(bg="#f0f0f0")

    method_var = StringVar(root)
    method_var.set("Cellpose")

    frame = Frame(root, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, bd=2)
    frame.pack(pady=20)

    Label(frame, text="Choose segmentation method:", font=("Arial", 11), bg="#ffffff", fg="#333").pack(pady=(0, 5))
    method_menu = OptionMenu(frame, method_var, "Custom Model", "Cellpose")
    method_menu.config(font=("Arial", 12), bg="#efefef", fg="#333")
    method_menu.pack(pady=5)

    status_label = Label(frame, text="No images selected", font=("Arial", 10), bg="#ffffff", fg="#333")
    status_label.pack(pady=10)
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(frame, length=400, variable=progress_var, maximum=100)
    progress_bar.pack(pady=10)

    list_frame = Frame(frame)
    list_frame.pack(pady=5)

    listbox = Listbox(list_frame, width=50, height=6, font=("Arial", 10),
                      yscrollcommand=lambda *args: scrollbar.set(*args))
    scrollbar = Scrollbar(list_frame, orient="vertical", command=listbox.yview)

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    select_button = Button(
        frame, text="📁 Select Images", font=("Arial", 12), bg="#4CAF50", fg="white",
        padx=10, pady=5,
        command=lambda: handle_select_images(status_label, run_button, file_paths, listbox)
    )
    select_button.pack(pady=10)

    run_button = Button(
        frame, text="▶ Run", font=("Arial", 12), bg="#008CBA", fg="white",
        padx=10, pady=5, state=tk.DISABLED,
        command=lambda: threading.Thread(
            target=handle_process_images,
            args=(file_paths, method_var, status_label, run_button, root, progress_var)
        ).start()
    )
    run_button.pack(pady=10)

    root.mainloop()

