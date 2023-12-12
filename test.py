import os
import numpy as np
import matplotlib.pyplot as plt

def plot_song_data(data_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through files in the data folder
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        
        # Check if the file is a .npz file
        if file_path.endswith('.npz'):
            with np.load(file_path) as data:
                # Load 'song' array and plot it
                if 'song' in data:
                    song = data['song']
                    plt.figure()
                    plt.step(range(len(song)), song, where='mid')
                    plt.title(f'Song Plot: {file}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')

                    # Save plot to the output folder
                    output_file_path = os.path.join(output_folder, f'{os.path.splitext(file)[0]}.png')
                    plt.savefig(output_file_path)
                    plt.close()

data_folder = '/home/george-vengrovski/Documents/projects/song_detection/data'
output_folder = '/home/george-vengrovski/Documents/projects/song_detection/plots'

plot_song_data(data_folder, output_folder)
