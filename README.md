# Machine Vision Course

# Exercise 1 - Object Color Tracking with OpenCV

This Python script utilizes the OpenCV library to track an object of a given color in a video recording. The color to be tracked is selected by pressing the left mouse button during the recording. The script allows specifying parameters such as color tolerance (hue), brightness tolerance (value), and saturation tolerance.

RUN COMMAND:
python3 object_color_tracking.py movingball.mp4 --hue_tolerance 15 --sat_tolerance 100 --val_tolerance 100

## Features

- **Color Selection**: Users can select the color to be tracked by clicking the left mouse button on the object of interest during video playback.
- **Adjustable Parameters**: Parameters like color tolerance, brightness tolerance, and saturation tolerance can be adjusted to fine-tune the object tracking.
- **Real-time Tracking**: The script performs real-time object tracking within the video stream.
- **Visual Feedback**: The tracked object is highlighted in the video feed, providing visual feedback to the user.