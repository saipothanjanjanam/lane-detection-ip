import numpy as np
import cv2

	
img = cv2.imread('1.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

yellowLow = np.array([180, 180, 30], dtype="uint8") 
yellowHigh = np.array([200,200,140], dtype="uint8")
whiteLow = np.array([175, 175, 175], dtype="uint8") 
whiteHigh = np.array([200,200,200], dtype="uint8")

yellowMask = cv2.inRange(hsvImg, yellowLow, yellowHigh)
whiteMask = cv2.inRange(hsvImg,whiteLow,whiteHigh)
combMask = cv2.bitwise_or(grayImg, yellowMask, whiteMask)
yandwMaskedImg = cv2.bitwise_and(grayImg, combMask)

gaussianBlurImg = cv2.GaussianBlur(yandwMaskedImg, (5,5), 0)
edgeImg = cv2.Canny(gaussianBlurImg,50,150)

blackImg = np.zeros(edgeImg.shape, dtype="uint8")
roiPoints = np.array([[(0,270), (795,390), (550,155), (194,155)]], dtype=np.int32)
roiMask = cv2.fillPoly(blackImg, roiPoints,(255))
roiImg = cv2.bitwise_and(roiMask,edgeImg)

#line image for drawing the lines
lineImg = np.zeros(img.shape,dtype="uint8")

# For Probabilistic Hough Transform and drawing the lines
# houghLines = cv2.HoughLinesP(roiImg, 2, np.pi/180, 40, np.array([]), minLineLength=10, maxLineGap=1)
# for line in houghLines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(lineImg, (x1, y1), (x2, y2), (0,0,255), 10)

#Common Hough Transform and drawing the lines
houghLines = cv2.HoughLines(edgeImg,1,np.pi/180,200)

if houghLines is not None:
        for value in houghLines:
                rho = value[0][0]
                theta = value[0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(lineImg,(x1,y1),(x2,y2),(0,0,255),2)

alpha=0.8
beta=1
lambdaa=0
finalLineDetectImg = cv2.addWeighted(lineImg, alpha, img, beta, lambdaa)

cv2.imshow('lane', finalLineDetectImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

