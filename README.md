# Machine Vision Course

# Project 1 - Object Color Tracking with OpenCV

This Python script utilizes the OpenCV library to track an object of a given color in a video recording. The color to be tracked is selected by pressing the left mouse button during the recording. The script allows specifying parameters such as color tolerance (hue), brightness tolerance (value), and saturation tolerance.

RUN COMMAND:
```bash
python3 object_color_tracking.py movingball.mp4 --hue_tolerance 15 --sat_tolerance 100 --val_tolerance 100
```

## Features

- **Color Selection**: Users can select the color to be tracked by clicking the left mouse button on the object of interest during video playback.
- **Adjustable Parameters**: Parameters like color tolerance, brightness tolerance, and saturation tolerance can be adjusted to fine-tune the object tracking.
- **Real-time Tracking**: The script performs real-time object tracking within the video stream.
- **Visual Feedback**: The tracked object is highlighted in the video feed, providing visual feedback to the user.

# Project 2 - Image Dataset Processing and Organization

This Python script processes and organizes a set of images into a dataset based on user-defined parameters. It supports resizing, optional grayscale conversion, and categorization of images. The script also generates statistics about the processing of images, which are saved in a CSV file.

RUN COMMAND:
```bash
python3 main.py -s /path/to/source/images -o /path/to/output/directory -sz 100x100 -f jpg -g
```

## Features

- **Image Resizing**: Users can specify the desired size for output images, which are resized accordingly.
- **Grayscale Conversion**: The script offers an option to convert images to grayscale, useful for specific image processing tasks.
- **Categorization**: Images are automatically categorized based on the original folder structure from the source directory.
- **Statistics Generation**: The script creates a CSV file containing statistics such as the number of processed and abandoned images per category.
- **Flexible Output Formats**: Supports saving processed images in either PNG or JPG formats.
- **Visual Organization**: Saves images into organized folders corresponding to their original categories in the source directory.

## Parameters

- `-s`, `--source`: Specifies the path to the directory containing the source images. This directory should contain subdirectories representing categories of images.
  
- `-o`, `--output`: Designates the output directory where the processed images and CSV statistics file will be saved. This directory will be created if it does not exist.

- `-sz`, `--size`: Sets the target size for the output images in the format WIDTHxHEIGHT. For example, `100x100` means the images will be resized to 100 pixels wide and 100 pixels high.

- `-g`, `--grayscale`: When this flag is used, the script converts all processed images to grayscale. If omitted, images are processed in their original color (RGB).

- `-f`, `--format`: Determines the format of the saved images. Possible values are `png` and `jpg`. This parameter specifies how the images are to be encoded after processing.