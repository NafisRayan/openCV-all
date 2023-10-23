import cv2
import numpy as np

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frames from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Modify the data type and apply the cornerHarris method
    gray = np.float32(gray)
    dest = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Results are marked through dilated corners
    dest = cv2.dilate(dest, None)

    # Mark the detected corners with red color
    frame[dest > 0.01 * dest.max()] = [0, 0, 255]

    # Show the live video with corners highlighted
    cv2.imshow('Real-Time Corner Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
