import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import matplotlib.cm as cm
import json
import threading
import queue
import random
import tkinter.messagebox as messagebox
import traceback

def array_to_image(array):
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    array_rgb = np.stack((array,) * 3, axis=-1)
    return Image.fromarray(array_rgb)

def blend_with_color(image, mask, color, alpha=0.3):
    overlay = Image.new('RGBA', image.size, (*color, 0))
    draw = ImageDraw.Draw(overlay)
    draw.bitmap((0, 0), mask, fill=(*color, int(255 * alpha)))
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

class SpectrogramViewer:
    def __init__(self, root, source_folder, output_folder):
        self.root = root
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.file_list = sorted([f for f in os.listdir(self.source_folder) if f.endswith('.npz')])
        self.json_path = os.path.join(source_folder, 'labeled_files.json')
        self.labeled_files = self.load_labeled_files()
        self.labeled_files_order = []
        self.current_file_index = -1

        # UI setup
        self.setup_ui()

        self.downsample_factor = 8  # Adjust this value to change the downsampling level
        self.preload_queue = queue.Queue(maxsize=5)
        self.preload_thread = threading.Thread(target=self.preload_spectrograms, daemon=True)
        self.preload_thread.start()

        self.current_label = 1  # 1 for red, 2 for green
        self.label_colors = {1: "Red", 2: "Green"}
        
        # Add a label for displaying current label info
        self.label_info = tk.Label(self.root, text="", fg="white", bg="black", font=("Courier", 16, "bold"), anchor="w")
        self.label_info.pack(side="top", fill="x")

        self.selection_start = None
        self.selection_end = None

        self.total_song_timebins = 0
        self.total_non_song_timebins = 0

        self.review_mode = False
        self.review_index = 0
        self.labeled_files_list = []
        
        # Add a label to show the current mode
        self.mode_label = tk.Label(self.root, text="Normal Mode", fg="white", bg="black", font=("Courier", 16, "bold"), anchor="w")
        self.mode_label.pack(side="top", fill="x")

        self.load_next_file()
        self.update_label_info()  # Initialize label info display

    def setup_ui(self):
        # Increase font size and make bold
        font_specs = ("Courier", 16, "bold")
        bindings_font_specs = ("Courier", 14, "bold")

        # Update to use a Text widget for multi-line keyboard bindings
        self.keyboard_bindings_text = tk.Text(self.root, height=6, bg="black", fg="white", font=bindings_font_specs)
        self.keyboard_bindings_text.insert(tk.END, self.get_keyboard_bindings_text())
        self.keyboard_bindings_text.pack(side="bottom", fill="x")
        self.keyboard_bindings_text.config(state="disabled")  # Make the text widget read-only

        self.filename_label = tk.Label(self.root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.filename_label.pack(side="top", fill="x")

        self.file_number_label = tk.Label(self.root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.file_number_label.pack(side="top", fill="x")

        # Label for displaying total length
        self.total_length_label = tk.Label(self.root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.total_length_label.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)
        self.canvas.focus_set()

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<Return>", self.mark_selection)
        self.canvas.bind("\\", self.toggle_label)
        self.canvas.bind("<Left>", self.prev_file)
        self.canvas.bind("<Right>", self.next_file)
        self.canvas.bind("<Button-2>", self.next_file)  # Middle mouse button
        self.canvas.bind("<Up>", self.jump_forward)
        self.canvas.bind("<Down>", self.jump_backward)
        self.canvas.bind("<n>", self.next_unseen_file)
        self.canvas.bind("w", self.clear_annotations)
        self.canvas.bind("r", self.toggle_review_mode)

    def on_press(self, event):
        self.selection_start = self.canvas.canvasx(event.x)
        self.selection_end = self.selection_start
        self.update_selection()

    def on_drag(self, event):
        self.selection_end = self.canvas.canvasx(event.x)
        self.update_selection()

    def on_release(self, event):
        self.selection_end = self.canvas.canvasx(event.x)
        self.update_selection()

    def update_selection(self):
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self.selection_start, 0, self.selection_end, self.canvas.winfo_height(), outline="white", tags="selection")

    def on_mousewheel(self, event):
        # Scroll horizontally
        if event.delta > 0 or event.num == 4:
            self.canvas.xview_scroll(-1, "units")
        else:
            self.canvas.xview_scroll(1, "units")

    def toggle_label(self, event):
        self.current_label = 3 - self.current_label  # Toggle between 1 and 2
        self.update_label_info()
        self.canvas.focus_set()

    def update_label_info(self):
        label_text = f"Current Label: {self.current_label} ({self.label_colors[self.current_label]})"
        self.label_info.config(text=label_text)

    def mark_selection(self, event):
        if self.selection_start is not None and self.selection_end is not None:
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)
            
            # Convert canvas coordinates to image coordinates
            start = int(start / self.zoom_factor)
            end = int(end / self.zoom_factor)

            # Ensure coordinates are within bounds
            start = max(0, min(start, self.original_image.width - 1))
            end = max(0, min(end, self.original_image.width - 1))

            # Update the selection mask
            draw = ImageDraw.Draw(self.selection_mask)
            fill_value = 128 if self.current_label == 1 else 255
            draw.rectangle([start, 0, end, self.original_image.height], fill=fill_value)
            
            # Update song_labels
            self.song_labels[start:end+1] = self.current_label

            self.apply_selection_mask()
            print(f"Marked selection from {start} to {end} with label {self.current_label}")

        self.canvas.focus_set()

    def apply_selection_mask(self):
        if self.original_image.width == 0 or self.original_image.height == 0:
            print("Cannot apply selection mask: Invalid original image dimensions")
            return

        zoomed_width = max(1, int(self.original_image.width * self.zoom_factor))
        zoomed_image = self.original_image.resize((zoomed_width, self.original_image.height), Image.Resampling.LANCZOS)
        
        zoomed_mask = self.selection_mask.resize((zoomed_width, self.original_image.height), Image.Resampling.NEAREST)
        red_mask = Image.new('1', zoomed_mask.size, 0)
        green_mask = Image.new('1', zoomed_mask.size, 0)
        red_mask.paste(1, mask=zoomed_mask.point(lambda p: p == 128 and 255))
        green_mask.paste(1, mask=zoomed_mask.point(lambda p: p == 255 and 255))
        marked_image = blend_with_color(zoomed_image, red_mask, (255, 0, 0))
        marked_image = blend_with_color(marked_image, green_mask, (0, 255, 0))

        self.update_spectrogram_display(marked_image)

    def load_spectrogram(self):
        if self.current_file_index < 0 or self.current_file_index >= len(self.file_list):
            print(f"Invalid current_file_index: {self.current_file_index}")
            return

        filename = self.file_list[self.current_file_index]
        self.filename_label.config(text=filename)
        self.file_number_label.config(text=f"File {self.current_file_index + 1} of {len(self.file_list)}")

        # Load and process spectrogram
        file_path = os.path.join(self.source_folder, filename)
        try:
            npz_data = np.load(file_path, allow_pickle=True)
            spectrogram = self.downsample_spectrogram(npz_data['s'])
            
            print(f"Spectrogram shape: {spectrogram.shape}")
            
            if spectrogram.size == 0:
                print(f"Empty spectrogram in file: {filename}")
                return

            self.song_labels = npz_data.get('song', np.zeros(spectrogram.shape[-1], dtype=np.uint8))
            self.song_labels = self.song_labels[::self.downsample_factor]

            # Normalize and process spectrogram
            spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            spectrogram = np.flipud(spectrogram)
            colored_spectrogram = cm.viridis(spectrogram)
            self.original_image = Image.fromarray((colored_spectrogram[:, :, :3] * 255).astype(np.uint8))

            print(f"Original image size: {self.original_image.size}")

            if self.original_image.width == 0 or self.original_image.height == 0:
                print(f"Invalid image dimensions in file: {filename}")
                return

            self.selection_mask = Image.new('L', self.original_image.size, 0)
            draw = ImageDraw.Draw(self.selection_mask)
            for i, label in enumerate(self.song_labels):
                if label > 0:
                    fill_value = 128 if label == 1 else 255
                    draw.rectangle([i, 0, i+1, self.original_image.height], fill=fill_value)

            # Set zoom to show entire spectrogram
            self.canvas_width = self.canvas.winfo_width()
            self.zoom_factor = self.canvas_width / self.original_image.width
            
            self.apply_selection_mask()

            # Update total length label
            total_length = self.original_image.width * self.downsample_factor
            self.total_length_label.config(text=f"Total Length: {total_length} Timebins")

            # Update label info when loading a new spectrogram
            self.update_label_info()

            if self.review_mode:
                self.update_review_hud()

        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            traceback.print_exc()

    def update_spectrogram_display(self, image):
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image

    def clear_annotations(self, event):
        self.selection_mask = Image.new('L', self.original_image.size, 0)
        self.song_labels = np.zeros(self.original_image.width, dtype=np.uint8)
        self.apply_selection_mask()
        self.save_current_markings()
        print("Annotations cleared")
        self.canvas.focus_set()

    def save_current_markings(self):
        file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])
        npz_data = np.load(file_path, allow_pickle=True)
        npz_data_dict = dict(npz_data)

        # Get the original number of timebins
        original_timebins = npz_data['s'].shape[1]

        # Create an array of the original size
        full_song_labels = np.zeros(original_timebins, dtype=np.uint8)

        # Map the downsampled labels back to the original scale
        for i, label in enumerate(self.song_labels):
            start = i * self.downsample_factor
            end = min((i + 1) * self.downsample_factor, original_timebins)
            full_song_labels[start:end] = label

        npz_data_dict['song'] = full_song_labels

        np.savez(file_path, **npz_data_dict)

        self.total_song_timebins += np.sum(full_song_labels > 0)
        self.total_non_song_timebins += len(full_song_labels) - np.sum(full_song_labels > 0)

        print(f"Saved markings for file: {self.file_list[self.current_file_index]}")

    def prev_file(self, event):
        if self.review_mode:
            self.save_current_markings()  # Save changes before moving to previous file
            self.review_index = (self.review_index - 1) % len(self.labeled_files_list)
            self.load_review_file()
        else:
            self.save_current_markings()
            self.mark_file_as_labeled()
            self.load_previous_file()
        self.canvas.focus_set()

    def next_file(self, event):
        if self.review_mode:
            self.save_current_markings()  # Save changes before moving to next file
            self.review_index = (self.review_index + 1) % len(self.labeled_files_list)
            self.load_review_file()
        else:
            self.save_current_markings()
            self.mark_file_as_labeled()
            self.load_next_file()
        self.canvas.focus_set()

    def jump_forward(self, event):
        self.save_current_markings()
        self.mark_file_as_labeled()
        self.current_file_index = min(len(self.file_list) - 1, self.current_file_index + 50)
        self.load_spectrogram()
        self.canvas.focus_set()

    def jump_backward(self, event):
        self.save_current_markings()
        self.mark_file_as_labeled()
        self.current_file_index = max(0, self.current_file_index - 50)
        self.load_spectrogram()
        self.canvas.focus_set()

    def next_unseen_file(self, event=None):
        json_path = os.path.join(self.source_folder, 'summary.json')
        if not os.path.exists(json_path):
            return  # No summary file, can't determine unseen files

        with open(json_path, 'r') as file:
            summary_data = json.load(file)

        processed_files = set(summary_data.get('processed_files', []))
        for index, filename in enumerate(self.file_list):
            if filename not in processed_files:
                self.current_file_index = index
                self.load_spectrogram()
                break

    def get_keyboard_bindings_text(self):
        bindings_text = (
            "Keyboard Bindings:\n"
            "Navigate: Left/Right Arrows\n"
            "Scroll: Mouse Wheel\n"
            "Mark Selection: Enter\n"
            "Clear Annotations: W\n"
            "Jump Forward/Backward: Up/Down Arrows\n"
            "Next Unseen File: N\n"
            "Toggle Label Color: \\\n"
            "Toggle Review Mode: R\n"
        )
        return bindings_text

    def downsample_spectrogram(self, spectrogram):
        # Downsample the spectrogram
        return spectrogram[:, ::self.downsample_factor]

    def preload_spectrograms(self):
        while True:
            current_index = self.current_file_index
            for i in range(current_index, min(current_index + 5, len(self.file_list))):
                if i not in self.preload_queue.queue:
                    file_path = os.path.join(self.source_folder, self.file_list[i])
                    try:
                        npz_data = np.load(file_path, allow_pickle=True)
                        spectrogram = self.downsample_spectrogram(npz_data['s'])
                        song_labels = npz_data.get('song', np.zeros(spectrogram.shape[-1], dtype=np.uint8))
                        song_labels = song_labels[::self.downsample_factor]  # Downsample labels
                        self.preload_queue.put((i, spectrogram, song_labels))
                    except Exception as e:
                        print(f"Error preloading file {file_path}: {e}")

    def exit_app(self):
        self.save_current_markings()
        self.mark_file_as_labeled()
        self.root.destroy()

    def load_labeled_files(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                labeled_files = json.load(f)
                self.labeled_files_order = labeled_files.copy()
                return set(labeled_files)
        return set()

    def save_labeled_files(self):
        with open(self.json_path, 'w') as f:
            json.dump(list(self.labeled_files), f)

    def get_random_unlabeled_file(self):
        unlabeled_files = set(self.file_list) - self.labeled_files
        if unlabeled_files:
            return random.choice(list(unlabeled_files))
        return None

    def load_next_file(self):
        next_file = self.get_random_unlabeled_file()
        if next_file:
            self.current_file_index = self.file_list.index(next_file)
        else:
            self.current_file_index = min(len(self.file_list) - 1, self.current_file_index + 1)
        self.load_spectrogram()

    def load_previous_file(self):
        if self.labeled_files_order:
            prev_file = self.labeled_files_order.pop()
            self.current_file_index = self.file_list.index(prev_file)
        else:
            self.current_file_index = max(0, self.current_file_index - 1)
        self.load_spectrogram()

    def mark_file_as_labeled(self):
        current_file = self.file_list[self.current_file_index]
        if current_file not in self.labeled_files:
            self.labeled_files.add(current_file)
            self.labeled_files_order.append(current_file)
            self.save_labeled_files()
        if self.review_mode and current_file not in self.labeled_files_list:
            self.labeled_files_list.append(current_file)

    def toggle_review_mode(self, event):
        if self.review_mode:
            self.save_current_markings()  # Save changes when exiting review mode
        self.review_mode = not self.review_mode
        if self.review_mode:
            self.labeled_files_list = list(self.labeled_files)
            if not self.labeled_files_list:
                messagebox.showinfo("Review Mode", "No labeled files to review.")
                self.review_mode = False
                self.mode_label.config(text="Normal Mode")
            else:
                self.review_index = 0
                self.mode_label.config(text="Review Mode")
                self.load_review_file()
        else:
            self.mode_label.config(text="Normal Mode")
            self.load_next_file()
        self.canvas.focus_set()
        print(f"Review mode: {'ON' if self.review_mode else 'OFF'}")

    def load_review_file(self):
        if self.labeled_files_list:
            if 0 <= self.review_index < len(self.labeled_files_list):
                file_to_review = self.labeled_files_list[self.review_index]
                self.current_file_index = self.file_list.index(file_to_review)
                self.load_spectrogram()
                self.update_review_hud()
                print(f"Reviewing file {self.review_index + 1} of {len(self.labeled_files_list)}: {file_to_review}")
            else:
                messagebox.showinfo("Review Complete", "You've reviewed all labeled files.")
                self.review_index = 0
                self.load_review_file()
        else:
            print("No labeled files to review.")

    def update_review_hud(self):
        current_file = self.file_list[self.current_file_index]
        self.filename_label.config(text=f"Reviewing: {current_file}")
        self.file_number_label.config(text=f"File {self.review_index + 1} of {len(self.labeled_files_list)}")

def run_app():
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    root.geometry("800x600")

    source_folder = filedialog.askdirectory(title="Select Source Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    app = SpectrogramViewer(root, source_folder, output_folder)

    # Save state and perform clean-up when the application is closed
    root.protocol("WM_DELETE_WINDOW", app.exit_app)

    root.mainloop()

run_app()