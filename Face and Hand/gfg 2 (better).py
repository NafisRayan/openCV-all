import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic_model.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
            )

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(
            image,
            f"{int(fps)} FPS",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Facial and Hand Landmarks", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()