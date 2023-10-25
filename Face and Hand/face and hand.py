import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  

while True:

  ret, frame = cap.read()

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Hand tracking
  hand_results = hands.process(rgb_frame)
  if hand_results.multi_hand_landmarks:
    for hand_landmarks in hand_results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

  # Face landmark detection  
  face_results = face_mesh.process(rgb_frame)
  if face_results.multi_face_landmarks:
    for face_landmarks in face_results.multi_face_landmarks:
      for landmark in face_landmarks.landmark:
        x, y = int(landmark.x*frame.shape[1]), int(landmark.y*frame.shape[0]) 
        cv2.circle(frame, (x,y), 2, (0,255,0), -1)

  cv2.imshow('Output', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()  
cv2.destroyAllWindows()
