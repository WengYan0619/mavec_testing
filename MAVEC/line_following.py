import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),0)
    canny = cv2.Canny(blur,50,150)
    _, binary = cv2.threshold(canny, 128, 255, cv2.THRESH_BINARY)
    
    return binary

def ROI(image):
    height = image.shape[0]
    traingle = np.array([[(60,height),(639,336),(371,158)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,traingle,255)
    masked_image = cv2.bitwise_and(image,mask)
    
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            
    return line_image
    
image = cv2.imread(r"C:\Users\wengy\Downloads\image3.png")
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = ROI(canny)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(image,0.8,line_image,1,1)
plt.imshow(canny)
plt.show()
cv2.imshow("Binary",combo_image)
cv2.waitKey(0)

