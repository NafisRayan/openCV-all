#A python program to detect your face and deviation of your face from the center of the frame using OpenCV.

#real time

import cv2

box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#*****use this for real time (camera)
#vid = cv2.VideoCapture(0)

#*****use this for video clips
vid = cv2.VideoCapture('fil/2.mp4')



fcx=fcy=x=y=w=h=0

while True:

    _, img = vid.read()

    faces = box.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 155, 255), 2)

        fcx = img.shape[1] / 2 - (x + w / 2)
        fcy = img.shape[0] / 2 - (y + h / 2)

        # THESE ARE HERE JUST TO MARK THE CENTER
        c1 = cv2.circle(img, (x + w//2,y + h//2), 1, (0, 0, 255), 2)
        c2 = cv2.circle(img, (img.shape[1]//2,img.shape[0]//2), 1, (255, 0, 0), 2)

        # JUST TO KEEP TRACK
        print('Deviation of face from the center')
        print(f'X coordinates: {fcx}\nY coordinates: {fcy}')

        # IT IS HERE TO MAKE A LINE TO INDICATE THE DISTANCE BETWEEN CENTERS
        cv2.line(img, (x + w // 2, y + h // 2), (img.shape[1] // 2, img.shape[0] // 2), (255,255,0), 1)
    cv2.putText(img,f'Deviation from center: {(fcx**2+fcy**2)**(0.5)} units.',(20,img.shape[0]-20),cv2.FONT_ITALIC,0.7,(300,20,25),2)

    cv2.putText(img,f'Position of the face: {(x + w/2,y + h/2)}',(20,img.shape[0]-100),cv2.FONT_ITALIC,0.7,(0,0,255),2)
    cv2.putText(img,f'Press (q) to EXIT',(20,img.shape[0]-60),cv2.FONT_ITALIC,0.7,(0,250,55),2)

    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


vid.release()