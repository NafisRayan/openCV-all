import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe face and hand modules
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize face detection
face_detection = mp_face_detection.FaceDetection()

# Initialize hand landmark detection
hands = mp_hands.Hands(max_num_hands=20)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
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

                # Extract face bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
                # Crop face region from the frame
                face_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

                # Convert the face frame to RGB and process it with Holistic model
                face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_results = holistic_model.process(face_rgb)

                # Convert the face results back to BGR for drawing
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)

                # Draw face landmarks on the face frame
                if face_results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        face_frame,
                        face_results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    )

                # Place the annotated face frame back into the original frame
                frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = face_frame

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

        cv2.imshow('Landmark Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()