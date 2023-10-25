import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Loop through all pose landmarks
        for landmark in results.pose_landmarks.landmark:
            # Extract the Z-coordinate (depth) of the landmark
            landmark_depth = landmark.z
            # use landmark_depth to estimate the distance in a real-world unit (e.g., centimeters)

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
