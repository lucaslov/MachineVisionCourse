'''
Object color tracker with cv.
'''

import argparse
import cv2 as cv
import numpy as np

class ColorTracker:
    """Class for tracking colored objects in a video"""

    def __init__(self, video_path):
        """
        Initialize the ColorTracker object.

        Args:
            video_path (str): Path to the input video file.
        """
        self.video = cv.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Unable to open video path: {video_path}")
        self.tracked_color = None
        self.hue_tolerance = 0
        self.sat_tolerance = 0
        self.val_tolerance = 0
        self.current_frame = None

    def update_frame(self):
        """
        Update the current frame from the video.

        Returns:
            numpy.ndarray or None: The current frame in HSV format if available, None otherwise.
        """
        ret, frame = self.video.read()
        if ret:
            self.current_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            return self.current_frame
        return None

    def set_color_reference(self, point, hue_tol, sat_tol, val_tol):
        """
        Sets the color reference for tracking.

        Parameters:
            x (int): The x-coordinate of the reference pixel.
            y (int): The y-coordinate of the reference pixel.
            hue_tol (int): The tolerance value for hue.
            sat_tol (int): The tolerance value for saturation.
            val_tol (int): The tolerance value for value.

        This method sets the reference color for tracking based on the provided pixel coordinates
        (x, y) from the current frame. The tolerance values for hue, saturation, and value are also
        set accordingly.
        """
        if self.current_frame is not None:
            self.tracked_color = self.current_frame[point['y'], point['x']]
            self.hue_tolerance = hue_tol
            self.sat_tolerance = sat_tol
            self.val_tolerance = val_tol

    def process_frame(self):
        """
        Process the current frame for object tracking.

        Returns:
            numpy.ndarray or None: Processed frame with tracked object
            highlighted or None if no frame available.
        """
        if self.current_frame is None or self.tracked_color is None:
            if self.current_frame is not None:
                return cv.cvtColor(self.current_frame, cv.COLOR_HSV2BGR)

        lower_bound = np.array([
            max(0, self.tracked_color[0] - self.hue_tolerance),
            max(0, self.tracked_color[1] - self.sat_tolerance),
            max(0, self.tracked_color[2] - self.val_tolerance)
        ])
        upper_bound = np.array([
            min(179, self.tracked_color[0] + self.hue_tolerance),
            min(255, self.tracked_color[1] + self.sat_tolerance),
            min(255, self.tracked_color[2] + self.val_tolerance)
        ])
        mask = cv.inRange(self.current_frame, lower_bound, upper_bound)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            (x, y), radius = cv.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            output = cv.cvtColor(self.current_frame, cv.COLOR_HSV2BGR)
            # Draw a green circle around the tracked object
            cv.circle(output, center, radius, (0, 255, 0), 2)
            return output
        return cv.cvtColor(self.current_frame, cv.COLOR_HSV2BGR)

class Display:
    """Class for displaying images in a window"""

    def __init__(self, window_name):
        """
        Initialize the Display object.

        Args:
            window_name (str): Name of the display window.
        """
        self.window_name = window_name
        cv.namedWindow(self.window_name)

    def show(self, image):
        """
        Show an image in the display window.

        Args:
            image (numpy.ndarray or None): The image to be displayed.
        """
        if image is not None:
            cv.imshow(self.window_name, image)

def mouse_callback(event, x, y, _, param):
    """
    Mouse callback function for setting the color reference point.

    Args:
        event (int): Type of mouse event.
        x (int): X-coordinate of the mouse event.
        y (int): Y-coordinate of the mouse event.
        param (dict): Dictionary containing parameters including tracker and tolerances.
    """
    if event == cv.EVENT_LBUTTONDOWN:
        tracker, hue_tol, sat_tol, val_tol = \
            param['tracker'], param['hue_tol'], param['sat_tol'], param['val_tol']
        tracker.set_color_reference({"x":x, "y":y}, hue_tol, sat_tol, val_tol)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Track an object based on color in a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--hue_tolerance', type=int, default=10, help='Hue tolerance.')
    parser.add_argument('--sat_tolerance', type=int, default=50, help='Saturation tolerance.')
    parser.add_argument('--val_tolerance', type=int, default=50, help='Value tolerance.')
    return parser.parse_args()

def main():
    """
    Main function for running the color tracking application.

    This function parses command-line arguments, initializes the ColorTracker object,
    sets up the display window, and starts the main loop for processing frames and
    displaying the results. The loop continues until the user presses the 'q' key
    or the video ends.

    Returns:
        None
    """
    args = parse_arguments()
    tracker = ColorTracker(args.video_path)
    display = Display('Color Tracker')

    cv.setMouseCallback(display.window_name, mouse_callback, {
        'tracker': tracker,
        'hue_tol': args.hue_tolerance,
        'sat_tol': args.sat_tolerance,
        'val_tol': args.val_tolerance
    })

    while True:
        frame = tracker.update_frame()
        if frame is None:
            break
        processed_frame = tracker.process_frame()
        # Show processed or original frame
        display.show(processed_frame if processed_frame is not None else frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
