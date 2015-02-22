import cv2
import numpy as np 
import matplotlib as plt


# mouse callback function
# def getRGBvalue(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print x , y
#         print imgBGR[x,y]


#Read image in
imgBGR = cv2.imread('Caltech1999/image_0126.jpg',1)

#imgRGB = imgBGR[:,:,::-1]

#Display image
cv2.imshow('Car Picture', imgBGR)
#cv2.waitKey(0)


# Create a black image, a window and bind the function to window
#img = np.zeros((512,512,3), np.uint8)
#cv2.namedWindow('image')
# cv2.setMouseCallback('Car Picture',getRGBvalue)

# while(1):
#     cv2.imshow('Car Picture',imgBGR)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()




#Classify image pixels into numColors color bins
#Dark blue, blue, light blue, 
#Dark red, red, light red
#Gray white, white, light white
#Dark Yellow, Yellow, Light yellow
#Other
#Do we need black?
# Build a vertical and horizontal histogram for each color
# input: image file
# output: vertColorHist(numColors,numVertPixels) 
#		  horColorHist(numColors,numHorPixels)

darkRed = 50;
margin = 10;
second = 42;

imgBlue = imgBGR[:,:,0]
imgGreen = imgBGR[:,:,1]
imgRed = imgBGR[:,:,2]

# Plot truly red pixels in red
#isRed = (imgRed > darkRed -margin)*(imgRed < darkRed+margin)*(imgBlue < second + margin)*(imgBlue > second - margin)*(imgGreen < second + margin)*(imgGreen > second - margin)
isRed = (imgRed > darkRed)*(imgRed > 1.7*imgBlue)*(imgRed>1.7*imgGreen)
minval = imgBGR.min(axis=2)
maxval = imgBGR.max(axis=2)
#isDarkRed =  (abs(minval-maxval) < 5)*(imgRed >60) #(imgRed<90)
redPixels = imgBGR
redPixels[:,:,2] = 255*isRed
redPixels[:,:,0] = imgBlue*(1-isRed)
redPixels[:,:,1] = imgGreen*(1- isRed)
cv2.imshow('redPixels', redPixels)

# # Plot truly blue pixels in blue
# # isBlue = (imgBlue > darkRed)*(imgBlue>1.5*imgRed)*(imgBlue>1.5*imgGreen)
# # bluePixels = imgBGR
# # bluePixels[:,:,0] = 255*isBlue
# # bluePixels[:,:,2] = imgRed*(1-isBlue)
# # bluePixels[:,:,1] = imgGreen*(1- isBlue)
# # cv2.imshow('bluePixels', bluePixels)

cv2.waitKey(0)

cv2.destroyAllWindows()