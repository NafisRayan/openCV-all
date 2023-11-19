from ultralytics import YOLO
import cv2
import math

image_path = 'img.jpg'  # Replace with the path to your image file

img = cv2.imread(image_path)

model = YOLO("yolov8x-seg.pt")

results = model.predict(source=0, show = True)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # Find the center of the object and the center of the text
        object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        name = classNames[cls]

        textPos = (max(0, x1), max(35, y1))
        x, y = textPos
        text = f"{name} {conf}"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        text_center = (max(0, x1) + text_size[0] // 2, max(35, y1) - text_size[1] // 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        textThickness = 2
        textColor = (0, 0, 0)
        bgColor = (255, 255, 255)
        pad_x, pad_y = 15, 15
        bgOpacity = 0.5
        (t_w, t_h) = text_size

        overlay = img.copy()  # copying the image
        # Draw
        cv2.circle(overlay, object_center, 5, (0, 0, 0), -1)

        cv2.line(overlay, object_center, text_center, (0, 0, 0), 1)

        cv2.rectangle(overlay, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)  # draw rectangle

        cv2.putText(overlay, text, textPos, font, fontScale, textColor, textThickness)  # draw in text

        cv2.rectangle(overlay, (max(0, x1) - pad_x, max(35, y1) - text_size[1] - pad_y),
                      (max(0, x1) + text_size[0] + pad_x, max(35, y1) + pad_y), (0, 25, 30), 2,
                      cv2.LINE_AA)

        img = cv2.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0)  # overlaying on the image.

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
