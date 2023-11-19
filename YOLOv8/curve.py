import cv2
import numpy as np

# Create a blank image
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Set the curvature variable (you can adjust this value)
curvature = 30

# Define the center, axes, and angle of the ellipse
center = (img.shape[1] // 2, img.shape[0] // 2)
axes = (img.shape[1] // 4, curvature)
angle = 0

# Draw the curved line (ellipse)
cv2.ellipse(img, center, axes, angle, 0, 180, (255, 255, 255), 2)

# Display the image
cv2.imshow("Curved Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
