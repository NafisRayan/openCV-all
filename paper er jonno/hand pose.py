import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use the default camera (change to your camera source if needed)

# Initialize the 3D cube
cube_size = 0.03  # Size of the cube (adjust as needed)
cube_color = (0, 255, 0)  # Green color (adjust as needed)

# Create a cube model
cube_points = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, -cube_size],
    [cube_size, 0, -cube_size],
    [cube_size, cube_size, -cube_size],
    [0, cube_size, -cube_size]
], dtype=np.float32)

cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Initialize a variable to hold cube position
cube_position = np.array([0, 0, 0], dtype=np.float32)

while True:
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the thumb tip coordinates (change as needed)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the 3D position of the thumb tip (adjust for your setup)
            x = thumb_tip.x
            y = thumb_tip.y
            z = thumb_tip.z * cube_size  # Scale depth to cube size

            # Update the cube's position
            cube_position = np.array([x, -y, z], dtype=np.float32)

            # Draw the cube at the thumb tip position
            for edge in cube_edges:
                start = tuple(np.int32(cube_points[edge[0]] + cube_position))
                end = tuple(np.int32(cube_points[edge[1]] + cube_position))
                cv2.line(frame, start, end, cube_color, 2)

    # Draw the hand landmarks on the frame
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking with 3D Cube", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
