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

# Define the maximum distance between thumb_cmc and pinky_tip for snap detection
MAX_DISTANCE = 0.1

# Load the fire image
fire_img = cv2.imread('path_to_fire_image.png', cv2.IMREAD_UNCHANGED)

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

            # Get the coordinates of thumb_cmc and pinky_tip
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate the Euclidean distance between thumb_cmc and pinky_tip
            distance = ((thumb_cmc.x - pinky_tip.x) ** 2 + (thumb_cmc.y - pinky_tip.y) ** 2) ** 0.5

            # Check if the distance is less than the maximum distance for snap detection
            if distance < MAX_DISTANCE:
                # Get the coordinates of the index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert the coordinates from normalized space to pixel values
                x = int(index_finger_tip.x * frame.shape[1])
                y = int(index_finger_tip.y * frame.shape[0])

                # Overlay the fire image on the frame at the index finger tip location
                overlay = cv2.resize(fire_img, (200, 150))  # Resize the fire image to desired size
                alpha = overlay[:, :, 3] / 255.0  # Get the alpha channel values
                h, w = overlay.shape[:2]  # Get the height and width of the overlay image

                # Calculate the region of interest for overlay
                roi = frame[y:h + y, x:w + x]

                # Perform alpha blending
                for c in range(3):
                    roi[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]

    # Show the frame
    cv2.imshow('Landmark Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()