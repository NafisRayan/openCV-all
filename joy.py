from ultralytics import YOLO
import cv2
import math

v = '2.mp4'

cap = cv2.VideoCapture(v)
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

# Initialize status bar parameters
status_bar_height = 50
status_bar_color = (0, 0, 0)
health = 100
ammo = 30


while True:
    success, img = cap.read()

    results = model(img, stream=True)

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
            #text = f'{name} {conf}'
            text = f"{name}"

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            text_center = ((max(0, x1) + text_size[0] // 2), (max(35, y1) - text_size[1] // 2))

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            textThickness=2 
            textColor=(0, 0, 0)
            bgColor=(255, 255, 255)
            pad_x, pad_y = 10, 10
            bgOpacity=0.5
            (t_w, t_h) = text_size


            overlay = img.copy()  # copying the image
           
           # Assuming 'overlay' already contains elements such as circles, lines, etc.

            # Define the desired horizontal and vertical distances between rectangles
            horizontal_distance = 0  # Adjust this value for horizontal space
            vertical_distance = 43# Adjust this value for vertical space

            # Define coordinates for the three rectangles with spacing
            rectB = ((x - pad_x), (max(35, y1) - text_size[1] - 2 * pad_y),
                    (x + text_size[0] + pad_x), (y + pad_y + text_size[1] + 2 * pad_y))
            rectC = ((x - pad_x - horizontal_distance), (y + pad_y + text_size[1] + pad_y + vertical_distance)-33,
                    (x + text_size[0] + pad_x + horizontal_distance), (y + pad_y + text_size[1] + 3 * pad_y + vertical_distance))

            # Draw rectangles A, B, and C on the overlay
            
            cv2.rectangle(overlay, (rectB[0], rectB[1]), (rectB[2], rectB[3]), (0, 255, 0), 2)  # Green color for B
            cv2.rectangle(overlay, (rectC[0], rectC[1]), (rectC[2], rectC[3]), (0, 255, 0), 2)  # Blue color for C


            # Put text inside rectangle B
            cv2.putText(overlay, text, textPos, font, fontScale, textColor, textThickness)



            # Draw status bar with carved underline
            # cv2.rectangle(overlay, (0, 0), (img.shape[1], status_bar_height), status_bar_color, -1)
            # cv2.line(overlay, (0, status_bar_height - 1), (img.shape[1], status_bar_height - 1), (255, 255, 255), 1)
            # cv2.line(overlay, (0, status_bar_height - 2), (img.shape[1], status_bar_height - 2), (255, 255, 255), 1)
            cv2.line(overlay, (0, status_bar_height - 3), (img.shape[1], status_bar_height - 3), (255, 255, 255), 1)

            # cv2.putText(overlay, f"Health: {health}  Ammo: {ammo}", (10, 30), font, 1, (255, 255, 255), 2)


            img = cv2.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0)  # overlaying on the image.


    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
