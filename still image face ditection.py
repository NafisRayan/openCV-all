#A python program to detect your face and deviation of your face from the center of the frame using OpenCV.

#still image

import cv2


box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('fil/1.jpg', -1)  # USE YOUR FILE LOCATION HERE

fcx=fcy=x=y=w=h=0
lst=[]
fcs=[]

faces = box.detectMultiScale(img, 1.1, 15)

for (x, y, w, h) in faces: 
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 155, 255), 2)

    fcx = img.shape[1]/2 - (x + w/2) 
    fcy = img.shape[0]/2 - (y + h/2) 

    # THESE ARE HERE JUST TO MARK THE CENTER
    c1 = cv2.circle(img, (x + w//2,y + h//2), 1, (0, 0, 255), 2)
    c2 = cv2.circle(img, (img.shape[1]//2,img.shape[0]//2), 1, (255, 0, 0), 2)

    # IT IS HERE TO MAKE A LINE TO INDICATE THE DISTANCE BETWEEN CENTERS
    cv2.line(img, (x + w // 2, y + h // 2), (img.shape[1] // 2, img.shape[0] // 2), (255,255,0), 1)

    lst.append(round((fcx**2+fcy**2)**(0.5),2))
    fcs.append((x + w/2,y + h/2))

    # JUST TO KEEP TRACK
    print('Deviation of face from the center')
    print(f'X coordinate deviation: {fcx}\nY coordinate deviation: {fcy}.\nLinear Deviation: {(fcx**2+fcy**2)**(0.5)}')



# OUT SIDE OF THE LOOP IT WILL ONLY PROVIDE THE LAST PERSON'S DATA FOR MULTIPLE PEOPLE
# but you can un-comment the following line to see the list of datas or you can check the output box
cv2.putText(img,f'Deviation Data List: {lst}.',(20,img.shape[0]-60),cv2.FONT_ITALIC,0.7,(2,255,0),2)
cv2.putText(img,f'Positions of the faces: {fcs}.',(20,img.shape[0]-20),cv2.FONT_ITALIC,0.5,(25,39,255),2)

print(f'Deviation Data List: {lst}.\nPositions of the faces: {fcs}.')


cv2.imshow('img', img)
cv2.waitKey(0)