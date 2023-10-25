import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def draw_Squires_on_fingertips():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=15
    ) as hands:  # Set max_num_hands to the desired number of hands to detect
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in [
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                    ]:
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
                        cv2.putText(
                            image,
                            "Squire",
                            (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
            cv2.imshow("Squires on Fingertips", image)
            if cv2.waitKey(10) == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()

draw_Squires_on_fingertips()