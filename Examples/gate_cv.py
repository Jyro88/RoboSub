import cv2
import sys
import numpy as np

#read the gate image
raw = cv2.imread("gate.png")
cv2.imshow("raw", raw)

final = raw

#convert to HSV
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

#only keep red-yellow values
low_H = 0
low_S = 200
low_V = 0
high_H = 255
high_S = 255
high_V = 255
threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
cv2.imshow("thresholded", threshold)

#erode to remove noise
kernel = np.ones((10, 10), np.uint8)
erode = cv2.erode(threshold, kernel)
cv2.imshow("eroded", erode)

#get the contours
contours = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
contour_area = [cv2.contourArea(x) for x in contours] # get the areas of each contour
contours_indexes = np.argsort(contour_area) # sort the indexes of the largest areas
for i in contours_indexes[-2:]: # only look at the two largest contours
    (x,y,w,h) = cv2.boundingRect(contours[i]) # get the location/dimensions of a bounding box for the contour
    cv2.rectangle(final, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=5)  # draw the bounding box on the image

    text = "gatepost"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    text_w, text_h = text_size
    cv2.rectangle(final, pt1=(x, y), pt2=(x + text_w, y - 2*text_h), color=(255, 0, 0), thickness=-1)
    cv2.putText(final, "gatepost", org=(x, y-5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)
cv2.imshow("final", final)

if cv2.waitKey(0) and 0xFF == ord('q'):
    sys.exit()

print("done")
