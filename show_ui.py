# import os
# import tkinter as tk
# from tkinter import filedialog, Label, Button, Frame
# import threading
# import matplotlib.pyplot as plt
# from src.cellpose_analysis import run_cellpose_analysis
# from src.label_cells import label_mitosis
# from src.predict_mitosis import predict_mitosis
# from src.segmentation import process_segmentation
# from src.show_images import show_images_side_by_side
#
#
# def show_ui():
#     file_path = None  # Store selected file path
#
#     def process_image():
#         status_label.config(text="🟡 Running Cellpose...")
#         root.update()
#         csv_path = run_cellpose_analysis(file_path)
#         # csv_path = "C:\\Users\\sharo\\Desktop\\Odd_test\\Mitosis_Detector\\frames\\output\\s540000_cells.csv"
#
#         status_label.config(text="✅ Cellpose Analysis Completed\n🟡 Creating Segmentation...")
#         root.update()
#         dataset_path = process_segmentation(csv_path, file_path)
#         # dataset_path = "C:\\Users\\sharo\\Desktop\\Odd_test\\Mitosis_Detector\\frames\\output\\s540000"
#
#         status_label.config(text="✅ Cellpose Analysis Completed\n✅ Segmentation Completed\n🟡 Predicting mitosis...")
#         root.update()
#         predict_mitosis(dataset_path, csv_path)
#
#         status_label.config(text="✅ All processing completed! 🖼️ Displaying image...")
#         root.update()
#
#         # Display the image with mitotic points
#         show_images_side_by_side(file_path, csv_path)
#
#         label_mitosis(csv_path, dataset_path, csv_path)
#
#
#     def select_image():
#         nonlocal file_path
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif")])
#         if file_path:
#             status_label.config(text=f"📂 Selected: {file_path.split('/')[-1]}")
#             run_button.config(state=tk.NORMAL)
#
#     def start_processing():
#         run_button.config(state=tk.DISABLED)
#         thread = threading.Thread(target=process_image)
#         thread.start()
#
#     # Create UI
#     root = tk.Tk()
#     root.title("Image Processing UI")
#     root.geometry("450x250")
#     root.configure(bg="#f0f0f0")
#
#     frame = Frame(root, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, bd=2)
#     frame.pack(pady=20)
#
#     select_button = Button(frame, text="📁 Select Image", command=select_image, font=("Arial", 12), bg="#4CAF50",
#                            fg="white", padx=10, pady=5)
#     select_button.pack(pady=10)
#
#     status_label = Label(frame, text="No image selected", font=("Arial", 10), bg="#ffffff", fg="#333")
#     status_label.pack(pady=10)
#
#     run_button = Button(frame, text="▶ Run", command=start_processing, font=("Arial", 12), bg="#008CBA", fg="white",
#                         padx=10, pady=5, state=tk.DISABLED)
#     run_button.pack(pady=10)
#
#     root.mainloop()

import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import threading
import matplotlib.pyplot as plt
from src.cellpose_analysis import run_cellpose_analysis
from src.label_cells import label_mitosis
from src.predict_mitosis import predict_mitosis
from src.segmentation import process_segmentation
from src.show_images import show_images_side_by_side


def show_ui():
    file_path = None  # Store selected file path

    def process_image():
        status_label.config(text="🟡 Running Cellpose...")
        root.update()
        #csv_path = run_cellpose_analysis(file_path)
        csv_path = "C:\\Users\\sharo\\Desktop\\Odd_test\\Mitosis_Detector\\frames\\output\\040000_cells.csv"

        status_label.config(text="✅ Cellpose Analysis Completed\n🟡 Creating Segmentation...")
        root.update()
        #dataset_path = process_segmentation(csv_path, file_path)
        dataset_path = "C:\\Users\\sharo\\Desktop\\Odd_test\\Mitosis_Detector\\frames\\output\\040000"

        status_label.config(text="✅ Cellpose Analysis Completed\n✅ Segmentation Completed\n🟡 Predicting mitosis...")
        root.update()
        #predict_mitosis(dataset_path, csv_path)

        status_label.config(text="✅ All processing completed!\n🖼️ Displaying image...")
        root.update()

        label_mitosis(csv_path, dataset_path)
        print('here')
        show_images_side_by_side(file_path, csv_path)

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
