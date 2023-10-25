import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for dark or black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])  # Adjust the V (value) range for darkness

    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Create a white background
    white_background = np.full_like(frame, (255, 255, 255), dtype=np.uint8)

    # Replace the background with white where black is not detected
    result = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask))
    result += cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Black Detector with White Background', result)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
