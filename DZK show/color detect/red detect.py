import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the masks to detect red color
    red_mask = mask1 + mask2

    # Apply the mask to the original frame
    red_detected = cv2.bitwise_and(frame, frame, mask=red_mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Red Detector', red_detected)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
