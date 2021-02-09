# import cv2

#print("Hello")

#img=cv2.imread("images/a.jpg")
#cv2.imshow("output", img)
#cv2.waitKey(1000)

#vid=cv2.VideoCapture("images/My Movie.mp4")
# vid=cv2.VideoCapture(0)
# vid.set(3,640)
# vid.set(6,480)
# vid.set(10,100)
#
# while True:
#     success, img = vid.read()
#     cv2.imshow("video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
# #faceCascade = cv2.CascadeClassifier("images/haarcascade_eye.xml")
# img = cv2.imread("images/01.png")
# imgResize = cv2.resize(img,(500,500))
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(imgGray,1.1,4)
# #faces1 = faceCascade1.detectMultiScale(imgGray,1.1,4)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#
# cv2.imshow("Result",img)
# #cv2.imshow("resize",imgResize)
# cv2.waitKey(10000)



# import cv2
#
# img = cv2.imread("images/face2.jpg")
#
# img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#
# cv2.imshow('HSV image', img_HSV)
#
# cv2.imshow('Hue Channel ', img_HSV[:, :, 0])
# cv2.imshow('Saturation ', img_HSV[:, :, 1])
# cv2.imshow('Value channel ', img_HSV[:, :, 2])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import cv2
#
# img = cv2.imread("images/face2.jpg",0)
#
# cv2.imshow("Gray",img)
# cv2.waitKey(0)
#
# ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("Binary", bw)
# # ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# # cv2.imshow("BinaryINV", bw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/face2.jpg", cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=2)
erosion = cv2.erode(mask, kernal, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()