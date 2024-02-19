import cv2
import sys
import numpy as np

# read the buoy image
raw = cv2.imread("buoy.png")
cv2.imshow("raw", raw)

final = raw
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

low_H = 0
low_S = 0
low_V = 0
high_H = 255
high_S = 100
high_V = 50
threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
cv2.imshow("thresholded", threshold)

# dilate to combine contours
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(threshold, kernel, iterations=3)
cv2.imshow("dilated", dilate)

# erode to remove noise
kernel = np.ones((10, 10), np.uint8)
erode = cv2.erode(dilate, kernel, iterations=2)
cv2.imshow("eroded", erode)

# blur to smooth edges so circle detection is easier
blur = cv2.blur(erode, ksize=(5, 5))
cv2.imshow("blur", blur)

# detect circles
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
              param1=30,  # edge detection parameter
              param2=30,  # accumulator threshold, or how circley the circles need to be to be recognized (higher=more circlely)
              minRadius=0,
              maxRadius=100)

# draw the circles
circle_radii = [x[2] for x in circles[0]]  # get the radii of each contour
circle_indexes = np.argsort(circle_radii)  # sort by largest radius
for i in circle_indexes[-2:]:  # only contour at the largest circles
    circle = circles[0][i]  # get the largest circle
    cv2.circle(final, center=(int(circle[0]), int(circle[1])), radius=int(circle[2]), color=(0, 0, 255), thickness=2)  # draw the circle on the image
    # make the text centered
    text = "police buoy"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)  # get the text size
    text_w, text_h = text_size  # get the text width/height
    cv2.putText(final, text, org=(int(circle[0] - text_w), int(circle[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)
cv2.imshow("final", final)

# wait for "q" key before closing images
if cv2.waitKey(0) and 0xFF == ord('q'):
    sys.exit()

print("done")
