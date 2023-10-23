# import numpy as np 
# import cv2 as cv 
# from cv2 import aruco 

# marker_id = 0
# marker_size = 200

# dictionary = aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
# marker = dictionary.generateImageMarker(sidePixels = marker_size, id = marker_id)

# border_color = (255, 255, 255)
# height, width = marker.shape[:2]
# border_width = 5
# marker = cv.copyMakeBorder(marker, top=border_width, bottom=border_width, left=border_width, right=border_width, borderType= cv.BORDER_CONSTANT, value=border_color)

# cv.imshow("Marker", marker)
# cv.imwrite("marker.png", marker)
# cv.waitKey(0)
# cv.destroyAllWindows()

import numpy as np 
import cv2 as cv 
from cv2 import aruco 
import pickle 

with open("data.pkl", "rb") as file:
    mtx, dist = pickle.load(file)

marker_size_cm = 6.4

input_video = cv.VideoCapture(0)

while True:
    ret, frame = input_video.read()
    frame_copy = frame[::]

    input_img = frame
    marker_ids = []
    detector_params = aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    detector = aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(input_img)
    output_img = input_img[::]

    object_points = np.zeros((4,3), dtype=np.float32)
    object_points[0, :] = np.array([-marker_size_cm/2, marker_size_cm/2, 0])
    object_points[1, :] = np.array([marker_size_cm/2, marker_size_cm/2, 0])
    object_points[2, :] = np.array([marker_size_cm/2, -marker_size_cm/2, 0])
    object_points[3, :] = np.array([-marker_size_cm/2, -marker_size_cm/2, 0])


    for i in range(len(marker_corners)):
        rvec, tvec, _ = cv.solvePnP(object_points, marker_corners[i], mtx, dist)
        v=f"Id {marker_ids[i]} Distence: {tvec[2][0]}"
        print(v)
        cv.putText(output_img,v,(20,output_img.shape[0]-60),cv.FONT_ITALIC,0.7,(0,250,55),2)


    aruco.drawDetectedMarkers(output_img, marker_corners, marker_ids)

    cv.imshow("Feed", output_img)
    cv.waitKey(1)
    # if cv.getWindowProperty("Feed", 0) == -1:
    #     break
    if cv.waitKey(30) & 0xFF == ord('s'):
        cv.destroyAllWindows()
        break

cv.destroyAllWindows()