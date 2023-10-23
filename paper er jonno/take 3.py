import cv2
import numpy as np

# Load the pre-trained pose detection model from OpenCV
pose_net = cv2.dnn.readNetFromTensorflow('pose_estimation_model.pb')

# Create a VideoCapture object for video input
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Resize the frame for processing
    img = cv2.resize(img, (600, 400))

    # Prepare the input image for pose detection
    blob = cv2.dnn.blobFromImage(img, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Set the input to the network and get the output
    pose_net.setInput(blob)
    output = pose_net.forward()

    # Process the output to extract landmarks
    h, w = img.shape[:2]
    points = []
    for i in range(output.shape[1]):
        confidence_map = output[0, i, :, :]
        x, y = np.unravel_index(np.argmax(confidence_map), confidence_map.shape)
        points.append((int(x * w / output.shape[3]), int(y * h / output.shape[2])))

    # Draw landmarks on the image
    for point in points:
        cv2.circle(img, point, 5, (255, 0, 0), -1)

    # Display the image with drawn landmarks
    cv2.imshow("Pose Estimation", img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
