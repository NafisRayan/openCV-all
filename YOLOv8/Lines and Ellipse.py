import cv2
import numpy as np

# Create a black image
img = np.zeros((720, 1280, 3), np.uint8)

# Draw three horizontal lines in the center
line_y = img.shape[0] // 2  # Y-coordinate for the center of the image

cv2.line(img, (img.shape[1]//4, line_y - 20), (img.shape[1]*3//4, line_y - 20), (255, 255, 255), 1)
cv2.line(img, (img.shape[1]//3, line_y), (img.shape[1]*2//3, line_y), (255, 255, 255), 1)
cv2.line(img, (img.shape[1]//4, line_y + 20), (img.shape[1]*3//4, line_y + 20), (255, 255, 255), 1)

# Define the center, axes, and angle of the ellipse
center = (640, 17)  # Adjusted center coordinates

# Calculate half the size of the image diagonally for the major axis
axes = (int(np.sqrt(1280**2 + 720**2) / 2), 50)

angle = 0

# Draw the half ellipse
cv2.ellipse(img, center, axes, angle, 0, 180, (255, 255, 255), 1)

# Display the image
cv2.imshow('Lines and Ellipse', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
