import cv2

# Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#*****use this for real time (camera)
#cap = cv2.VideoCapture(0)

#*****use this for video clips
cap = cv2.VideoCapture('fil/1.mp4')

#in inches
known_width = 8

#(estimate)
focal_length = 600

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Calculate the distance of the face from the frame
    for (x, y, w, h) in faces:

        # Calculate the distance using the formula d = (f * w) / p

        distance = (known_width * focal_length) / w
        distance = round(distance, 2)

        cv2.putText(frame, f"Distance: {distance} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'Press (q) to EXIT', (20,frame.shape[0]-60), cv2.FONT_ITALIC, 0.7, (0,250,255),2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()