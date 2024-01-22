import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import matplotlib.cm as cm
import json

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

        # Increase font size and make bold
        font_specs = ("Courier", 16, "bold")
        bindings_font_specs = ("Courier", 14, "bold")

        # Update to use a Text widget for multi-line keyboard bindings
        self.keyboard_bindings_text = tk.Text(root, height=6, bg="black", fg="white", font=bindings_font_specs)
        self.keyboard_bindings_text.insert(tk.END, self.get_keyboard_bindings_text())
        self.keyboard_bindings_text.pack(side="bottom", fill="x")
        self.keyboard_bindings_text.config(state="disabled")  # Make the text widget read-only

        self.filename_label = tk.Label(root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.filename_label.pack(side="top", fill="x")

        self.file_number_label = tk.Label(root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.file_number_label.pack(side="top", fill="x")

        # Label for displaying total length
        self.total_length_label = tk.Label(root, text="", fg="white", bg="black", font=font_specs, anchor="w")
        self.total_length_label.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(root, bg="black", scrollregion=(0, 0, 1000, 600))
        hbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=hbar.set)
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)
        self.canvas.focus_set()  # Set focus to the canvas

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # Bind mouse wheel event for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # For Windows and MacOS
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # For Linux, scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # For Linux, scroll down

        self.normal_zoom_factor = 1.0
        self.zoomed_out_factor = 0.1
        self.zoom_factor = self.zoomed_out_factor  # Set initial zoom factor to zoomed out
        self.is_zoomed_out = True  # Set the initial zoom state to zoomed out

        # Bind mouse wheel event for zooming
        self.canvas.bind("<MouseWheel>", self.toggle_zoom)  # For Windows and MacOS
        self.canvas.bind("<Button-3>", self.toggle_zoom)  # For Linux, scroll up
       
        self.canvas.bind("<Up>", self.jump_forward)
        self.canvas.bind("<Down>", self.jump_backward)

        # Bind navigation keys directly to the canvas
        self.canvas.bind("<e>", self.scroll_right)
        self.canvas.bind("<q>", self.scroll_left)
        self.canvas.bind("<Return>", self.mark_selection)
        self.canvas.bind("<w>", self.clear_annotations)
        self.canvas.bind("<n>", self.next_unseen_file)
        self.canvas.bind("<Left>", self.prev_file)
        self.canvas.bind("<Right>", self.next_file)

        self.total_song_timebins = 0
        self.total_non_song_timebins = 0

        # Load the last state if available
        self.state_file = os.path.join(self.output_folder, "last_state.json")
        self.load_last_state()

        self.load_spectrogram()
        # In the __init__ method, after loading the spectrogram
        self.canvas.config(scrollregion=(0, 0, self.original_image.width, self.original_image.height))
        self.apply_selection_mask()  # Apply the initial zoomed out state to the spectrogram

    def on_press(self, event):
        x = max(0, min(self.canvas.canvasx(event.x), self.original_image.width))
        self.selection_start = x
        self.selection_end = x
        self.update_selection()

    def on_drag(self, event):
        x = max(0, min(self.canvas.canvasx(event.x), self.original_image.width))
        self.selection_end = x
        self.update_selection()

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


    def on_release(self, event):
        self.update_selection()
        self.canvas.focus_set()  # Refocus on the canvas after a release

    def update_selection(self):
        self.canvas.delete("selection")
        scaled_start = int(self.selection_start * self.zoom_factor)
        scaled_end = int(self.selection_end * self.zoom_factor)
        self.canvas.create_line(self.selection_start, 0, self.selection_start, self.canvas.winfo_height(), fill="red", tags="selection")
        self.canvas.create_line(self.selection_end, 0, self.selection_end, self.canvas.winfo_height(), fill="red", tags="selection")

        if not self.selection_mask:
            self.selection_mask = Image.new('1', self.original_image.size, 0)

    def generate_json_summary(self):
        # Check if the JSON file already exists
        json_path = os.path.join(self.source_folder, 'summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                summary_data = json.load(file)
        else:
            summary_data = {'total_song_timebins': 0, 'total_non_song_timebins': 0}

        # Update the summary data
        summary_data['total_song_timebins'] += self.total_song_timebins
        summary_data['total_non_song_timebins'] += self.total_non_song_timebins
        ratio = summary_data['total_song_timebins'] / float(summary_data['total_non_song_timebins']) if summary_data['total_non_song_timebins'] > 0 else 0
        summary_data['song_to_non_song_ratio'] = ratio

        # Write the updated summary data to the JSON file
        with open(json_path, 'w') as file:
            json.dump(summary_data, file, indent=4)

        # Reset the counters for the next file
        self.total_song_timebins = 0
        self.total_non_song_timebins = 0

    def mark_selection(self, event):
        if self.selection_start is not None and self.selection_end is not None:
            # Scale the selection coordinates according to the zoom factor
            adjusted_start = int(self.selection_start * (1 / self.zoom_factor))
            adjusted_end = int(self.selection_end * (1 / self.zoom_factor))

            # Ensure the coordinates are within the bounds of the original image
            adjusted_start = max(0, min(adjusted_start, self.original_image.width))
            adjusted_end = max(0, min(adjusted_end, self.original_image.width))

            # Correct for situations where start is greater than end
            if adjusted_start > adjusted_end:
                adjusted_start, adjusted_end = adjusted_end, adjusted_start

            # Update the selection mask
            draw = ImageDraw.Draw(self.selection_mask)
            draw.rectangle([adjusted_start, 0, adjusted_end, self.original_image.height], fill=1)
            self.apply_selection_mask()

    def clear_annotations(self, event):
        self.selection_mask = None
        self.apply_selection_mask()  # Maintain zoom level

    def apply_selection_mask(self):
        # Apply the selection mask to the zoomed image
        zoomed_width = int(self.original_image.width * self.zoom_factor)
        zoomed_image = self.original_image.resize((zoomed_width, self.original_image.height), Image.Resampling.LANCZOS)

        if self.selection_mask is not None:
            # Resize the mask to the zoomed dimensions
            zoomed_mask = self.selection_mask.resize((zoomed_width, self.original_image.height), Image.Resampling.NEAREST)
            marked_image = blend_with_red(zoomed_image, zoomed_mask)
        else:
            marked_image = zoomed_image

        self.update_spectrogram_display(marked_image)

    def on_mousewheel(self, event):
        # Calculate the clicked position relative to the current zoom level
        clicked_x = self.canvas.canvasx(event.x) * (1 / self.zoom_factor)

        # Adjust the zoom factor based on scroll direction
        if event.num == 5 or event.delta < 0:  # Scroll down or forward
            self.zoom_factor = max(0.1, self.zoom_factor - 0.1)
        elif event.num == 4 or event.delta > 0:  # Scroll up or backward
            self.zoom_factor = min(2.0, self.zoom_factor + 0.1)

        # Apply zoom
        self.zoom_spectrogram(self.zoom_factor)

        # Calculate new position after zoom
        new_x = clicked_x * self.zoom_factor

        # Adjust canvas view to center on the clicked position
        canvas_width = self.canvas.winfo_width()
        new_scroll_x = max(0, min(new_x - canvas_width / 2, self.original_image.width * self.zoom_factor - canvas_width))
        self.canvas.xview_moveto(new_scroll_x / (self.original_image.width * self.zoom_factor))

    def toggle_zoom(self, event):
        # Calculate the clicked position relative to the current zoom level
        clicked_x = self.canvas.canvasx(event.x) * (1 / self.zoom_factor)

        self.is_zoomed_out = not self.is_zoomed_out
        if self.is_zoomed_out:
            self.zoom_factor = self.initial_zoom_factor
        else:
            self.zoom_factor = self.normal_zoom_factor

        # Apply zoom
        self.zoom_spectrogram(self.zoom_factor)

        # Calculate new position after zoom
        new_x = clicked_x * self.zoom_factor

        # Adjust canvas view to center on the clicked position
        canvas_width = self.canvas.winfo_width()
        new_scroll_x = max(0, min(new_x - canvas_width / 2, self.original_image.width * self.zoom_factor - canvas_width))
        self.canvas.xview_moveto(new_scroll_x / (self.original_image.width * self.zoom_factor))

    def zoom_spectrogram(self, zoom_factor):
        new_width = int(self.original_image.width * zoom_factor)
        zoomed_image = self.original_image.resize((new_width, self.original_image.height), Image.Resampling.LANCZOS)

        if self.selection_mask is not None:
            zoomed_mask = self.selection_mask.resize((new_width, self.original_image.height), Image.Resampling.NEAREST)
            marked_image = blend_with_red(zoomed_image, zoomed_mask)
        else:
            marked_image = zoomed_image

        self.update_spectrogram_display(marked_image)

    def save_current_state(self):
        state_data = {
            'last_file_index': self.current_file_index,
            # Add any other state data you want to save here
        }
        with open(self.state_file, 'w') as file:
            json.dump(state_data, file, indent=4)

    def load_last_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as file:
                state_data = json.load(file)
            self.current_file_index = state_data.get('last_file_index', 0)
            # Load any other state data you saved here


    def save_current_markings(self):
        if self.selection_mask:
            # Resize the selection mask to match the original image dimensions
            original_size_mask = self.selection_mask.resize((self.original_image.width, self.original_image.height), Image.Resampling.NEAREST)

            # Convert the mask to a numpy array and reshape
            song_labels = np.array(original_size_mask.getdata(), dtype=np.uint8).reshape(self.original_image.height, self.original_image.width).any(axis=0).astype(np.uint8)
            
            file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])
            npz_data = np.load(file_path, allow_pickle=True)
            npz_data_dict = dict(npz_data)  # Convert to a regular dictionary
            npz_data_dict['song'] = song_labels

            np.savez(file_path, **npz_data_dict)

            self.total_song_timebins += np.sum(song_labels)
            self.total_non_song_timebins += len(song_labels) - np.sum(song_labels)

    def load_spectrogram(self):
        if self.current_file_index < 0 or self.current_file_index >= len(self.file_list):
            return

        file_path = os.path.join(self.source_folder, self.file_list[self.current_file_index])
        filename = os.path.basename(file_path)
        self.filename_label.config(text=filename)
        file_number_text = f"File {self.current_file_index + 1} of {len(self.file_list)}"
        self.file_number_label.config(text=file_number_text)

        npz_data = np.load(file_path, allow_pickle=True)
        spectrogram = npz_data['s']
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # Flip the spectrogram vertically
        spectrogram = np.flipud(spectrogram)

        # Apply a colormap to the normalized spectrogram
        colored_spectrogram = cm.viridis(spectrogram)

        # Convert the colored spectrogram to an RGB image
        self.original_image = Image.fromarray((colored_spectrogram[:, :, :3] * 255).astype(np.uint8))

        self.song_labels = npz_data.get('song', np.zeros(spectrogram.shape[-1], dtype=np.uint8))
        self.selection_mask = Image.new('1', self.original_image.size, 0)

        draw = ImageDraw.Draw(self.selection_mask)
        for i, marked in enumerate(self.song_labels):
            if marked:
                # Adjust the drawing coordinate for the flipped image
                draw.rectangle([i, 0, i+1, self.original_image.height], fill=1)

        # Calculate and display total length in timebins
        total_length = self.original_image.width  # Assuming each pixel represents one timebin
        self.total_length_label.config(text=f"Total Length: {total_length} Timebins")

        # Set the initial zoom level to fit the window size
        initial_window_width = self.canvas.winfo_width()
        self.initial_zoom_factor = initial_window_width / self.original_image.width
        self.zoom_factor = self.initial_zoom_factor

        self.apply_selection_mask()

    def get_keyboard_bindings_text(self):
        bindings_text = (
            "Keyboard Bindings:\n"
            "Navigate: Left/Right Arrows\n"
            "Zoom: Mouse Wheel/Scroll\n"
            "Scroll: Q/E\n"
            "Mark Selection: Enter\n"
            "Clear Annotations: W\n"
            "Jump Forward/Backward: Up/Down Arrows\n"
            "Next Unseen File: N\n"
        )
        return bindings_text

    def update_spectrogram_display(self, image):
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image

    def scroll_left(self, event):
        if self.canvas.xview()[0] > 0:
            self.canvas.xview_scroll(-1, "units")

    def scroll_right(self, event):
        if self.canvas.xview()[1] < 1:
            self.canvas.xview_scroll(1, "units")

    def prev_file(self, event):
        self.save_current_markings()
        self.save_current_state()  # Save current state
        self.generate_json_summary()  # Update JSON file
        self.current_file_index = max(0, self.current_file_index - 1)
        self.load_spectrogram()
        self.canvas.focus_set()

    def next_file(self, event):
        self.save_current_markings()
        self.save_current_state()  # Save current state
        self.generate_json_summary()  # Update JSON file
        self.current_file_index = min(len(self.file_list) - 1, self.current_file_index + 1)
        self.load_spectrogram()
        self.canvas.focus_set()

    def jump_forward(self, event):
        # Move 50 files forward, or to the end of the list if fewer than 50 files remain
        self.save_current_markings()
        self.current_file_index = min(len(self.file_list) - 1, self.current_file_index + 50)
        self.load_spectrogram()
        self.canvas.focus_set()

    def jump_backward(self, event):
        # Move 50 files backward, or to the start of the list if currently within the first 50 files
        self.save_current_markings()
        self.current_file_index = max(0, self.current_file_index - 50)
        self.load_spectrogram()
        self.canvas.focus_set()

    def exit_app(self):
        self.save_current_markings()
        self.generate_json_summary()  # Make sure this is before save_current_state
        self.save_current_state()
        self.root.destroy()

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