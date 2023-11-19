from ultralytics import YOLO
import cv2
import math
import datetime
import numpy as np

v = '2.mp4'
cap = cv2.VideoCapture(v)
# Set the resolution to 1280x720
cap.set(3, 1280)  # CV_CAP_PROP_FRAME_WIDTH
cap.set(4, 720)   # CV_CAP_PROP_FRAME_HEIGHT

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

# Define the center, axes, and angle of the ellipse
center = (640, 0)  # Adjusted center coordinates
axes = (int(np.sqrt(1280**2 + 720**2) / 2), 50)
angle = 0

# Initialize status bar parameters
status_bar_height = 50
status_bar_color = (0, 0, 0)

# Set the curvature variable for the curved line
curvature = 30

while True:
    success, img = cap.read()

    results = model(img, stream=True)

    # Draw status bar with curved underline
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    status_bar = f'Time: {time_str} | Date: {date_str} | Weather: Sunny, 29c'
    text_size = cv2.getTextSize(status_bar, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    image_height, image_width, _ = img.shape

    # Draw the curved line
    cv2.ellipse(img, (image_width // 2, status_bar_height - curvature), (image_width // 2, curvature), 0, 0, 180, (255, 255, 255), 1)

    # Draw the status bar text
    cv2.putText(img, status_bar, ((image_width - text_size[0]) // 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    #cv2.line(img, (0, status_bar_height - 20), (img.shape[1], status_bar_height - 20), (255, 255, 255), 1)


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

            textPos = (max(0, x1), max(24, y1))
            x, y = textPos
            text = f"{name} {conf}"

            text_scale = 0.7  # to control all the sizes

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0]
            text_center = (max(0, x1) + text_size[0] // 2, max(24, y1) - text_size[1] // 2)

            fontScale = text_scale
            textThickness = 2 
            textColor = (0, 0, 0)
            bgColor = (255, 255, 255)
            pad_x, pad_y = int(10 * text_scale), int(10 * text_scale)
            Opacity = 0.3
            (t_w, t_h) = text_size

            overlay = img.copy()  # copying the image

            # Draw 
            cv2.circle(overlay, object_center, int(4 * text_scale), (0, 0, 0), -1)  
            cv2.line(overlay, object_center, text_center, (0, 0, 0), 1) 
            cv2.rectangle(overlay, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)  # draw rectangle
            cv2.putText(overlay, text, (max(0, x1), max(24, y1)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness)  # draw text
            cv2.rectangle(overlay, (max(0, x1) - pad_x, max(24, y1) - t_h - pad_y), (max(0, x1) + t_w + pad_x, max(24, y1) + pad_y), (0, 25, 30), int(2 * text_scale), cv2.LINE_AA)

            img = cv2.addWeighted(overlay, Opacity, img, 1 - Opacity, 0)  # overlaying on the image.

            # Draw three horizontal lines in the center
            line_y = img.shape[0] // 2  # Y-coordinate for the center of the image
            cv2.line(img, (img.shape[1]//4, line_y - 20), (img.shape[1]*3//4, line_y - 20), (255, 255, 255), 1)
            cv2.line(img, (img.shape[1]//3, line_y), (img.shape[1]*2//3, line_y), (255, 255, 255), 1)
            cv2.line(img, (img.shape[1]//4, line_y + 20), (img.shape[1]*3//4, line_y + 20), (255, 255, 255), 1)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
