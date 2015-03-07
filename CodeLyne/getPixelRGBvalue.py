import cv2
import numpy as np 
import matplotlib as plt


#Read image in
imgBGR = cv2.imread('Caltech1999/image_0126.jpg',1)

#mouse callback function
def getRGBvalue(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print x , y
        print imgBGR[y,x]


#Display image
cv2.imshow('Car Picture', imgBGR)
#cv2.waitKey(0)

# Create a black image, a window and bind the function to window
#img = np.zeros((512,512,3), np.uint8)
#cv2.namedWindow('image')
cv2.setMouseCallback('Car Picture',getRGBvalue)

while(1):
    cv2.imshow('Car Picture',imgBGR)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


