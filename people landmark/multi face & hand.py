import cv2
import mediapipe as mp

# Initialize Mediapipe face and hand modules
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize face detection
face_detection = mp_face_detection.FaceDetection()

# Initialize hand landmark detection
hands = mp_hands.Hands(max_num_hands=20)

while True:
    # Read frames from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB and process it with Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    hand_results = hands.process(rgb)

    # Draw face landmarks
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw circles around fingertips
            for landmark in hand_landmarks.landmark:
                if landmark.HasField('visibility') and landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow('Landmark Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
