import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the new image
image_path = "/mnt/data/KakaoTalk_20250308_154117244.jpg"
new_image = cv2.imread(image_path)

# Convert image to HSV to segment background better
hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)

# Define the color range for the background (green screen)
lower_green = np.array([35, 40, 40])  # Lower bound of green in HSV
upper_green = np.array([90, 255, 255])  # Upper bound of green in HSV

# Create a mask for the background
mask = cv2.inRange(hsv, lower_green, upper_green)
s
# Invert the mask to keep only the object
mask_inv = cv2.bitwise_not(mask)

# Apply morphological operations to refine the mask (remove noise)
kernel = np.ones((5, 5), np.uint8)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)

# Find contours of the chicken shape (after removing the background)
contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image_cleaned = new_image.copy()
cv2.drawContours(contour_image_cleaned, contours, -1, (0, 255, 0), 2)  # Green contour

# Display the result
plt.figure(figsize=(6,6))
plt.title("Chicken Shape Contour")
plt.imshow(cv2.cvtColor(contour_image_cleaned, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()