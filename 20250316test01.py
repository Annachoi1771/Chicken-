import cv2
import numpy as np
import matplotlib.pyplot as plt

class DrumstickContourDetector:
    def __init__(self, image_path, output_path="drumstick_contour.png"):
        self.image_path = image_path
        self.output_path = output_path
        self.image = None
        self.contour_image = None
        print(f"Initialized with image_path: {self.image_path}, output_path: {self.output_path}")

    def load_image(self):
        print(f"Loading image from: {self.image_path}")
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print("Image successfully loaded and converted to RGB.")

    def preprocess_image(self):
        print("Preprocessing image: Converting to grayscale and applying Gaussian blur.")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print("Applying threshold to create a binary mask.")
        _, binary_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        print("Image preprocessing completed.")
        return binary_mask

    def detect_contour(self):
        print("Detecting contours in the image.")
        binary_mask = self.preprocess_image()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image!")

        largest_contour = max(contours, key=cv2.contourArea)
        print(f"Found {len(contours)} contours. Selecting the largest one.")

        contour_image = np.zeros_like(self.image_rgb)

        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
        self.contour_image = contour_image
        print("Contour detection completed.")

    def save_and_display_result(self):
        if self.contour_image is None:
            raise ValueError("Contour image not generated. Run detect_contour() first.")

        print(f"Saving contour result to {self.output_path}")
        cv2.imwrite(self.output_path, cv2.cvtColor(self.contour_image, cv2.COLOR_RGB2BGR))

        print("Displaying result...")
        plt.figure(figsize=(8, 8))
        plt.imshow(self.contour_image)
        plt.axis("off")
        plt.title("Detected Chicken Drumstick Contour")
        plt.show()
        print("Result display completed.")

    def run(self):
        print("Starting the drumstick contour detection process.")
        self.load_image()
        self.detect_contour()
        self.save_and_display_result()
        print("Process completed successfully.")