import os
from shutil import move
from pathlib import Path


def extract_nth_files(folder, extract_nth_frame):
    # Define the path object for the folder
    folder_path = Path(folder)

    # Gather all jpg and png files
    image_files = [file for file in folder_path.iterdir() if file.suffix.lower() in ['.jpg', '.png']]

    # Sort the files to ensure consistent ordering
    image_files.sort()

    # Filter images, keeping only the first and every nth frame
    filtered_images = [image_files[i] for i in range(len(image_files)) if i == 0 or (i % extract_nth_frame == 0)]

    # Move files temporarily to avoid naming conflicts
    temp_folder = folder_path / 'temp_images'
    temp_folder.mkdir(exist_ok=True)
    for image in filtered_images:
        move(str(image), str(temp_folder / image.name))

    # Delete all other images
    for image in folder_path.glob('*'):
        if image.suffix.lower() in ['.jpg', '.png']:
            os.remove(image)

    # Move back from temp and rename
    for index, image in enumerate(sorted(temp_folder.iterdir(), key=lambda x: x.name), 1):
        new_name = folder_path / f'{index:04d}{image.suffix}'
        move(str(image), new_name)

    # Remove the temporary directory
    os.rmdir(temp_folder)