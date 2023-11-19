from ultralytics import YOLO
import cv2
import math
import numpy as np

x = '2.mp4'

cap = cv2.VideoCapture(x)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Draw rectangle
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # conf = round(box.conf[0].item(), 2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            name = classNames[cls]
            text_size = cv2.getTextSize(f'{name} {conf}', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

            # Find the center of the object and the center of the text
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            text_center = (max(0, x1) + text_size[0] // 2, max(35, y1) - text_size[1] // 2)

            # Draw line between object center and text center
            cv2.line(img, object_center, text_center, (0, 255, 0), 2)  # Adjusted line color to green

            # Draw a small circle at the center of the object
            cv2.circle(img, object_center, 5, (0, 0, 0), -1)  # Adjusted circle color to red

            # Draw background behind text with rounded corners
            rect_padding = 10
            cv2.rectangle(img, (max(0, x1) - rect_padding, max(35, y1) - text_size[1] - rect_padding),
                          (max(0, x1) + text_size[0] + rect_padding, max(35, y1) + rect_padding), (0, 0, 0), -1)
            cv2.rectangle(img, (max(0, x1) - rect_padding, max(35, y1) - text_size[1] - rect_padding),
                          (max(0, x1) + text_size[0] + rect_padding, max(35, y1) + rect_padding), (0, 255, 0), 2,
                          cv2.LINE_AA)

            # Draw text
            cv2.putText(img, f'{name} {conf}', (max(0, x1), max(35, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Adjusted text color to white

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
