import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os

def array_to_image(array):
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    array_rgb = np.stack((array,) * 3, axis=-1)
    return Image.fromarray(array_rgb)

def blend_with_red(image, mask, alpha=0.3):
    red_overlay = Image.new('RGBA', image.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(red_overlay)
    draw.bitmap((0, 0), mask, fill=(255, 0, 0, int(255 * alpha)))
    return Image.alpha_composite(image.convert('RGBA'), red_overlay).convert('RGB')

class SpectrogramViewer:
    def __init__(self, root, source_folder, output_folder):
        self.root = root
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.file_list = sorted([f for f in os.listdir(self.source_folder) if f.endswith('.npz')])
        self.current_file_index = 0

        self.filename_label = tk.Label(root, text="", fg="white", bg="black", font=("Courier", 12), anchor="w")
        self.filename_label.pack(side="top", fill="x")

        self.file_number_label = tk.Label(root, text="", fg="white", bg="black", font=("Courier", 12), anchor="w")
        self.file_number_label.pack(side="top", fill="x")

        self.canvas = tk.Canvas(root, bg="black", scrollregion=(0, 0, 1000, 600))
        hbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=hbar.set)
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)
        self.canvas.focus_set()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.normal_zoom_factor = 1.0
        self.zoomed_out_factor = 0.1
        self.zoom_factor = self.normal_zoom_factor  # Current zoom factor

        self.canvas.bind("<MouseWheel>", self.toggle_zoom)  # Bind mouse wheel for zooming

        self.canvas.bind("<e>", self.scroll_right)
        self.canvas.bind("<q>", self.scroll_left)
        self.canvas.bind("<Return>", self.mark_selection)
        self.canvas.bind("<w>", self.clear_annotations)
        self.canvas.bind("<Left>", self.prev_file)
        self.canvas.bind("<Right>", self.next_file)

        self.load_spectrogram()

    def on_press(self, event):
        self.selection_start = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        self.selection_end = self.selection_start
        self.update_selection()

    def on_drag(self, event):
        self.selection_end = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        self.update_selection()

    def on_release(self, event):
        self.update_selection()
        self.canvas.focus_set()

    def update_selection(self):
        # Clear the previous selection and draw new one
        self.canvas.delete("selection")
        scaled_start = int(self.selection_start * self.zoom_factor)
        scaled_end = int(self.selection_end * self.zoom_factor)
        self.canvas.create_line(scaled_start, 0, scaled_start, self.canvas.winfo_height(), fill="red", tags="selection")
        self.canvas.create_line(scaled_end, 0, scaled_end, self.canvas.winfo_height(), fill="red", tags="selection")

    def mark_selection(self, event):
        if not self.selection_mask:
            self.selection_mask = Image.new('1', self.original_image.size, 0)
        
        draw = ImageDraw.Draw(self.selection_mask)
        draw.rectangle([self.selection_start, 0, self.selection_end, self.original_image.height], fill=1)
        self.apply_selection_mask()

    def clear_annotations(self, event):
        self.selection_mask = Image.new('1', self.original_image.size, 0)
        self.apply_selection_mask()

        file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])
        npz_data = np.load(file_path, allow_pickle=True)
        npz_data['song'] = np.zeros_like(npz_data['song'])
        np.savez(file_path, **npz_data)

    def apply_selection_mask(self):
        marked_image = blend_with_red(self.original_image, self.selection_mask)
        self.update_spectrogram_display(marked_image)

    def toggle_zoom(self, event):
        self.zoom_factor = self.zoomed_out_factor if self.zoom_factor == self.normal_zoom_factor else self.normal_zoom_factor
        self.zoom_spectrogram(self.zoom_factor)

    def zoom_spectrogram(self, zoom_factor):
        new_width = int(self.original_image.width * zoom_factor)
        zoomed_image = self.original_image.resize((new_width, self.original_image.height), Image.Resampling.LANCZOS)
        self.canvas.config(scrollregion=(0, 0, new_width, self.original_image.height))
        self.update_spectrogram_display(zoomed_image)

    def load_spectrogram(self):
        file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])
        filename = os.path.basename(file_path)
        self.filename_label.config(text=filename)
        file_number_text = f"File {self.current_file_index + 1} of {len(self.file_list)}"
        self.file_number_label.config(text=file_number_text)
        
        npz_data = np.load(file_path, allow_pickle=True)
        spectrogram = npz_data['s']
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        self.original_image = array_to_image(spectrogram * 255)

        self.song_labels = npz_data.get('song', np.zeros(spectrogram.shape[-1], dtype=np.uint8))
        self.selection_mask = Image.new('1', self.original_image.size, 0)

        draw = ImageDraw.Draw(self.selection_mask)
        for i, marked in enumerate(self.song_labels):
            if marked:
                draw.rectangle([i, 0, i+1, self.original_image.height], fill=1)

        self.apply_selection_mask()

    def update_spectrogram_display(self, image):
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image

    def scroll_left(self, event):
        self.canvas.xview_scroll(-1, "units")

    def scroll_right(self, event):
        self.canvas.xview_scroll(1, "units")

    def prev_file(self, event):
        self.save_current_markings()
        self.current_file_index = max(0, self.current_file_index - 1)
        self.load_spectrogram()
        self.canvas.focus_set()

    def next_file(self, event):
        self.save_current_markings()
        self.current_file_index = min(len(self.file_list) - 1, self.current_file_index + 1)
        self.load_spectrogram()
        self.canvas.focus_set()

    def save_current_markings(self):
        if self.selection_mask:
            song_labels = np.array(self.selection_mask.getdata(), dtype=np.uint8).reshape(self.original_image.height, self.original_image.width).any(axis=0).astype(np.uint8)
            file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])

            npz_data = np.load(file_path, allow_pickle=True)
            npz_data['song'] = song_labels
            np.savez(file_path, **npz_data)

def run_app():
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    root.geometry("800x600")

    source_folder = filedialog.askdirectory(title="Select Source Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    app = SpectrogramViewer(root, source_folder, output_folder)
    root.mainloop()

run_app()
