import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")  

cap = cv2.VideoCapture('2.mp4')


while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model.predict(frame)

    detection = results[0].plot(kpt_line=True, kpt_radius=5, conf=False, boxes=True, labels=False)

    cv2.imshow('YOLOv8 Pose Detection', detection)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



#result = model(source= '2.mp4',conf=0.5,show=True,save=False)
