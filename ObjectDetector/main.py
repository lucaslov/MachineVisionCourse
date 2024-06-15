import numpy as np
import cv2
from typing import List

class ORBObjectDetector:
    def __init__(self, threshold: float = 0.75, nfeatures: int = 500):
        """
        Initializes the ORBObjectDetector with a specified threshold and number of features.
        
        :param threshold: The threshold for considering an object as present based on matching descriptors.
        :param nfeatures: Maximum number of features to retain (default is 500).
        """
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.threshold = threshold
        self.descriptors = []

    def train(self, img: np.ndarray, present: bool) -> None:
        """
        Trains the detector with an image and a flag indicating if the object is present.
        
        :param img: The input image.
        :param present: Boolean flag indicating if the object is present in the image.
        """
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        
        if present and descriptors is not None:
            self.descriptors.append(descriptors)

    def classify(self, img: np.ndarray) -> bool:
        """
        Classifies an image to determine if the object is present based on the trained descriptors.
        
        :param img: The input image.
        :return: Boolean indicating if the object is present.
        """
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        if descriptors is None or len(self.descriptors) == 0:
            return False
        
        # Flatten the list of descriptors
        all_train_descriptors = np.vstack(self.descriptors)
        
        # Match descriptors
        matches = self.matcher.knnMatch(descriptors, all_train_descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Calculate the sum of matching descriptors using the given formula
        y = sum(1 / (1 + m.distance) for m in good_matches)
        
        # Debug information
        print(f"Number of good matches: {len(good_matches)}")
        print(f"Sum of matching descriptors (y): {y}")
        
        # Determine if the object is present
        return y > self.threshold

# Example usage
if __name__ == "__main__":
    detector = ORBObjectDetector(threshold=0.75, nfeatures=500)
    
    # Load training images and train the detector
    train_img1 = cv2.imread('ObjectDetector/train/ball/sample1.png', cv2.IMREAD_GRAYSCALE)
    detector.train(train_img1, present=True)
    
    train_img2 = cv2.imread('ObjectDetector/train/ball/sample2.png', cv2.IMREAD_GRAYSCALE)
    detector.train(train_img2, present=True)
    
    train_img3 = cv2.imread('ObjectDetector/train/ball/sample3.png', cv2.IMREAD_GRAYSCALE)
    detector.train(train_img3, present=False)
    
    # Load a test image and classify
    test_img = cv2.imread('ObjectDetector/test/ball/test4.png', cv2.IMREAD_GRAYSCALE)
    result = detector.classify(test_img)
    print("Object present:", result)