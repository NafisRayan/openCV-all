from ultralytics import YOLO
import cv2

video_path = '2.mp4'

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    for result in results:
        for keypoint_indx, keypoint in enumerate(result.keypoints.tolist()):
            cv2.putText(frame, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
