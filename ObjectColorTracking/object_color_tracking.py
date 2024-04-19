import cv2
import numpy as np
import argparse

class ColorTracker:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Unable to open video path: {video_path}")
        self.tracked_color = None
        self.hue_tolerance = 0
        self.sat_tolerance = 0
        self.val_tolerance = 0
        self.current_frame = None

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return self.current_frame
        return None

    def set_color_reference(self, x, y, hue_tol, sat_tol, val_tol):
        if self.current_frame is not None:
            self.tracked_color = self.current_frame[y, x]
            self.hue_tolerance = hue_tol
            self.sat_tolerance = sat_tol
            self.val_tolerance = val_tol

    def process_frame(self):
        if self.current_frame is None or self.tracked_color is None:
            return cv2.cvtColor(self.current_frame, cv2.COLOR_HSV2BGR) if self.current_frame is not None else None

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
        mask = cv2.inRange(self.current_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            output = cv2.cvtColor(self.current_frame, cv2.COLOR_HSV2BGR)
            cv2.circle(output, center, radius, (0, 255, 0), 2)  # Draw a green circle around the tracked object
            return output
        return cv2.cvtColor(self.current_frame, cv2.COLOR_HSV2BGR)

class Display:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show(self, image):
        if image is not None:
            cv2.imshow(self.window_name, image)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker, hue_tol, sat_tol, val_tol = param['tracker'], param['hue_tol'], param['sat_tol'], param['val_tol']
        tracker.set_color_reference(x, y, hue_tol, sat_tol, val_tol)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Track an object based on color in a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--hue_tolerance', type=int, default=10, help='Hue tolerance.')
    parser.add_argument('--sat_tolerance', type=int, default=50, help='Saturation tolerance.')
    parser.add_argument('--val_tolerance', type=int, default=50, help='Value tolerance.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    tracker = ColorTracker(args.video_path)
    display = Display('Color Tracker')

    cv2.setMouseCallback(display.window_name, mouse_callback, {
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
        display.show(processed_frame if processed_frame is not None else frame)  # Show processed or original frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
