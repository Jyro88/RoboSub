import cv2
import numpy as np

#Find and compare slopes of two lines
def parallel(line1, line2):
    x1, y1, x2, y2 = line1[0]
    if (x2 - x1) != 0:
        slope1 = (y2 - y1) / (x2 - x1) 
    else: 
        slope1 = float('inf')

    x1, y1, x2, y2 = line2[0]
    if (x2 - x1) != 0:
        slope2 = (y2 - y1) / (x2 - x1)
    else:
        slope2 = float('inf')

    return slope1 == slope2 

#Draw line on image
def drawLine(line):
    x1, y1, x2, y2 = line[0]
    cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 5)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videoplayback.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Read in as grayscale
    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(gray, 50, 150, None, 3)

    final = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Use Hough Line Transform to detect lines in the image
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    # Go through array of lines and check for pairs of parallel lines
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if parallel(lines[i], lines[j]):
                    drawLine(lines[i])
                    drawLine(lines[j])

    cv2.imshow('Frame',frame)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", final)
  
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
