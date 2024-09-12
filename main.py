import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import matplotlib.cm as cm
import json
import threading
import time
import tkinter.messagebox as messagebox
import logging
import fcntl  # For file locking
import io  # For BytesIO
import traceback  # Add this import at the top of your file
import shutil  # For file operations
import random  # For shuffling unlabeled files

logging.basicConfig(level=logging.DEBUG)

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
        
        if not self.file_list:
            messagebox.showerror("Error", "No .npz files found in the selected source folder.")
            self.root.destroy()
            return

        self.json_path = os.path.join(source_folder, 'labeled_files.json')
        self.labeled_files = self.load_labeled_files()
        
        # Initialize unlabeled files and shuffle them
        self.unlabeled_files = list(set(self.file_list) - self.labeled_files)
        random.shuffle(self.unlabeled_files)
        self.normal_mode_history = []
        self.current_filename = None

        if self.unlabeled_files:
            self.current_filename = self.unlabeled_files.pop()
            self.normal_mode_history.append(self.current_filename)
        else:
            messagebox.showinfo("Info", "All files have been labeled.")
            self.root.destroy()
            return

        # UI setup
        self.setup_ui()

        self.downsample_factor = 1  # Adjust this value to change the downsampling level

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

        self.current_npz_data = None

        self.load_spectrogram()
        self.update_file_info()  # Add this line
        self.update_label_info()  # Initialize label info display

        # Start auto-save thread
        self.auto_save_interval = 60  # Auto-save every 60 seconds
        self.auto_save_thread = threading.Thread(target=self.auto_save_worker, daemon=True)
        self.auto_save_event = threading.Event()
        self.auto_save_thread.start()

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

        # Add an exit button
        exit_button = tk.Button(self.root, text="X", command=self.exit_app, bg="red", fg="white")
        exit_button.pack(side="top", anchor="ne")

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
            self.save_current_markings()  # Save immediately after marking
            logging.debug(f"Marked selection from {start} to {end} with label {self.current_label}")

        self.canvas.focus_set()

    def apply_selection_mask(self):
        if self.original_image.width == 0 or self.original_image.height == 0:
            logging.error("Cannot apply selection mask: Invalid original image dimensions")
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
        if self.current_filename is None:
            logging.error("No current filename to load.")
            return
        filename = self.current_filename
        file_path = os.path.join(self.source_folder, filename)
        try:
            # Read the data into a dictionary
            with np.load(file_path, allow_pickle=True) as data:
                self.current_npz_data = dict(data)

            spectrogram = self.downsample_spectrogram(self.current_npz_data['s'])
            
            logging.debug(f"Spectrogram shape: {spectrogram.shape}")
            
            if spectrogram.size == 0:
                logging.error(f"Empty spectrogram in file: {filename}")
                return

            full_song_labels = self.current_npz_data.get('song', np.zeros(self.current_npz_data['s'].shape[1], dtype=np.uint8))
            # Downsample the labels
            self.song_labels = full_song_labels[::self.downsample_factor]

            # Normalize and process spectrogram
            spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            spectrogram = np.flipud(spectrogram)
            colored_spectrogram = cm.viridis(spectrogram)
            self.original_image = Image.fromarray((colored_spectrogram[:, :, :3] * 255).astype(np.uint8))

            logging.debug(f"Original image size: {self.original_image.size}")

            if self.original_image.width == 0 or self.original_image.height == 0:
                logging.error(f"Invalid image dimensions in file: {filename}")
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
            logging.error(f"Error loading file {filename}: {str(e)}")
            messagebox.showerror("Error", f"Failed to load file {filename}: {str(e)}")
            traceback.print_exc()
            # Initialize with empty data if loading fails
            self.current_npz_data = None
            self.song_labels = np.array([])

        self.update_file_info()

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
        logging.debug("Annotations cleared")
        self.canvas.focus_set()

    def save_current_markings(self):
        if self.current_npz_data is None or self.current_filename is None:
            logging.debug("No data to save")
            return

        # Upsample the labels to full resolution
        original_timebins = self.current_npz_data['s'].shape[1]
        full_song_labels = np.zeros(original_timebins, dtype=np.uint8)
        factor = self.downsample_factor
        for i, label in enumerate(self.song_labels):
            start = i * factor
            end = min((i + 1) * factor, original_timebins)
            full_song_labels[start:end] = label

        try:
            # Save the updated data without deleting the original file
            filename = self.current_filename
            file_path = os.path.join(self.source_folder, filename)
            backup_path = file_path + '.bak'
            if not os.path.exists(backup_path):
                # Create a backup if it doesn't exist
                shutil.copy2(file_path, backup_path)
            else:
                # Overwrite the backup
                shutil.copy2(file_path, backup_path)

            # Update the 'song' data in current_npz_data
            self.current_npz_data['song'] = full_song_labels

            # Save the updated data
            with open(file_path, 'wb') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                np.savez(f, **self.current_npz_data)
                fcntl.flock(f, fcntl.LOCK_UN)
            logging.debug(f"Saved markings for file: {filename}")
            self.mark_file_as_labeled()

        except Exception as e:
            logging.error(f"Error saving markings for file {filename}: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save labels for {filename}: {str(e)}")
            traceback.print_exc()

    def prev_file(self, event):
        if self.review_mode:
            self.prev_review_file()
        else:
            self.save_current_markings()
            if len(self.normal_mode_history) > 1:
                # Remove current file from history
                self.normal_mode_history.pop()
                self.current_filename = self.normal_mode_history[-1]
                self.load_spectrogram()
            else:
                messagebox.showinfo("Info", "This is the first file, cannot go back.")
        self.update_file_info()
        self.canvas.focus_set()

    def next_file(self, event):
        if self.review_mode:
            self.next_review_file()
        else:
            self.save_current_markings()
            if self.unlabeled_files:
                self.current_filename = self.unlabeled_files.pop()
                self.normal_mode_history.append(self.current_filename)
                self.load_spectrogram()
            else:
                messagebox.showinfo("Info", "All files have been labeled.")
        self.update_file_info()
        self.canvas.focus_set()

    def prev_review_file(self):
        self.save_current_markings()
        self.review_index = max(0, self.review_index - 1)
        self.load_review_file()

    def next_review_file(self):
        self.save_current_markings()
        self.review_index = min(len(self.labeled_files_list) - 1, self.review_index + 1)
        self.load_review_file()

    def update_file_info(self):
        if self.review_mode:
            self.update_review_hud()
        else:
            self.filename_label.config(text=self.current_filename)
            self.file_number_label.config(text=f"Unlabeled Files Remaining: {len(self.unlabeled_files)}")

    def jump_forward(self, event):
        # Disabled in normal mode due to random file navigation
        if self.review_mode:
            # Jump forward by 10 files in review mode
            self.review_index = min(len(self.labeled_files_list) - 1, self.review_index + 10)
            self.load_review_file()
        else:
            messagebox.showinfo("Info", "Jump Forward is disabled in random navigation mode.")
        self.canvas.focus_set()

    def jump_backward(self, event):
        # Disabled in normal mode due to random file navigation
        if self.review_mode:
            # Jump backward by 10 files in review mode
            self.review_index = max(0, self.review_index - 10)
            self.load_review_file()
        else:
            messagebox.showinfo("Info", "Jump Backward is disabled in random navigation mode.")
        self.canvas.focus_set()

    def next_unseen_file(self, event=None):
        # Disabled since normal mode already skips labeled files
        messagebox.showinfo("Info", "Next Unseen File is not applicable in random navigation mode.")
        self.canvas.focus_set()

    def get_keyboard_bindings_text(self):
        bindings_text = (
            "Keyboard Bindings:\n"
            "Navigate: Left/Right Arrows\n"
            "Scroll: Mouse Wheel\n"
            "Mark Selection: Enter\n"
            "Clear Annotations: W\n"
            "Jump Forward/Backward: Up/Down Arrows (Review Mode)\n"
            "Toggle Label Color: \\\n"
            "Toggle Review Mode: R\n"
        )
        return bindings_text

    def downsample_spectrogram(self, spectrogram):
        # Downsample the spectrogram
        factor = self.downsample_factor
        if factor == 1:
            return spectrogram
        else:
            # Handle cases where the spectrogram's timebins are not divisible by factor
            timebins = spectrogram.shape[1]
            new_timebins = timebins // factor
            spectrogram = spectrogram[:, :new_timebins * factor]
            spectrogram = spectrogram.reshape(spectrogram.shape[0], new_timebins, factor).mean(axis=2)
            return spectrogram

    def exit_app(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.save_current_markings()  # Save any unsaved changes
            self.root.quit()
            self.root.destroy()

    def load_labeled_files(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    return set(data)
            except json.JSONDecodeError:
                logging.error("Corrupted labeled_files.json file.")
                messagebox.showerror("Error", "Corrupted labeled_files.json file.")
                return set()
        return set()

    def save_labeled_files(self):
        try:
            with open(self.json_path, 'w') as f:
                json.dump(list(self.labeled_files), f)
            logging.debug(f"Saved labeled files: {self.labeled_files}")
        except Exception as e:
            logging.error(f"Error saving labeled files: {e}")
            messagebox.showerror("Error", f"Failed to save labeled files: {str(e)}")

    def mark_file_as_labeled(self):
        current_file = os.path.basename(self.current_filename)
        self.labeled_files.add(current_file)
        logging.debug(f"Marking {current_file} as labeled")
        self.save_labeled_files()
        # Remove from unlabeled files if present
        if current_file in self.unlabeled_files:
            self.unlabeled_files.remove(current_file)

    def toggle_review_mode(self, event):
        self.save_current_markings()
        self.review_mode = not self.review_mode
        if self.review_mode:
            self.labeled_files = self.load_labeled_files()  # Reload labeled files
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
            self.load_spectrogram()
        self.update_file_info()
        self.canvas.focus_set()
        logging.debug(f"Review mode: {'ON' if self.review_mode else 'OFF'}")

    def load_review_file(self):
        if self.labeled_files_list:
            if 0 <= self.review_index < len(self.labeled_files_list):
                file_to_review = self.labeled_files_list[self.review_index]
                # Find the index in self.file_list without path discrepancies
                if file_to_review in self.file_list:
                    self.current_filename = file_to_review
                    self.load_spectrogram()
                    self.update_review_hud()
                else:
                    logging.error(f"File {file_to_review} not found in file list.")
                    messagebox.showerror("Error", f"File {file_to_review} not found in file list.")
                    return
            else:
                messagebox.showinfo("Review Complete", "You've reviewed all labeled files.")
                self.review_mode = False
                self.mode_label.config(text="Normal Mode")
                self.load_spectrogram()
        else:
            logging.debug("No labeled files to review.")
            messagebox.showinfo("Review Mode", "No labeled files to review.")

    def update_review_hud(self):
        current_file = self.current_filename
        self.filename_label.config(text=f"Reviewing: {current_file}")
        self.file_number_label.config(text=f"File {self.review_index + 1} of {len(self.labeled_files_list)}")

    def auto_save_worker(self):
        while not self.auto_save_event.is_set():
            time.sleep(self.auto_save_interval)
            if not self.auto_save_event.is_set():
                self.save_current_markings()
                logging.debug("Auto-saved current markings.")

def run_app():
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    root.geometry("800x600")
    
    # Remove the default close button
    # root.overrideredirect(True)  # Commented out to prevent window from being blank
    
    source_folder = filedialog.askdirectory(title="Select Source Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    if not source_folder or not output_folder:
        messagebox.showerror("Error", "Source and Output folders must be selected.")
        root.destroy()
        return

    try:
        app = SpectrogramViewer(root, source_folder, output_folder)
    except Exception as e:
        logging.error(f"Exception during initialization: {e}")
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred during initialization: {e}")
        root.destroy()
        return

    # Center the window on the screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    root.mainloop()

if __name__ == "__main__":
    run_app()
