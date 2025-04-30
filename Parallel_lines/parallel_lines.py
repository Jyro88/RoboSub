import cv2
import numpy as np

# Find and compare slopes of two lines
def parallel(line1, line2):
    x1, y1, x2, y2 = line1[0]
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

    x1, y1, x2, y2 = line2[0]
    slope2 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

    return (abs(slope1 - slope2) < 0.5 or slope1 == slope2) and (abs(slope1) > 5 and abs(slope2) > 5)

# Draw line on image
def drawLine(line):
    x1, y1, x2, y2 = line[0]
    cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 5)
    drawn_lines.add(tuple(line[0]))  # Add the drawn line to a set

# Simplified hashing to compare similar lines
def hash_line(line):
    x1, y1, x2, y2 = line[0]
    return (round(x1 / 15), round(y1 / 15), round(x2 / 15), round(y2 / 15))

# Add a white border between images
def add_border(img):
    return cv2.copyMakeBorder(img, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('robosub 2022 gate only.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        drawn_lines = set()  # Set to store already drawn lines

        # Extract the red components from the frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70]) # light red
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70]) # dark red
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Apply the red mask to the original frame
        red_frame = cv2.bitwise_and(frame, frame, mask=red_mask)
        # cv2.imshow('red_frame', red_frame)

        # Convert the red frame to grayscale
        red_gray = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(red_gray, (5, 5), 0)
        # cv2.imshow('red_gray', red_gray)

        # Use Canny edge detector to find edges in the red frame
        edges = cv2.Canny(blurred, 50, 150, None, 3)
        # cv2.imshow('edges', edges)

        # Erosion operation to remove small noise
        kernel = np.ones((10, 10), np.uint8)
        erode = cv2.erode(edges, kernel)

        # Convert the eroded image to BGR for drawing lines
        final = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)

        # Convert edges to BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Ensure all frames are valid and same size
        height, width = frame.shape[:2]
        edges_bgr = cv2.resize(edges_bgr, (width, height))
        final = cv2.resize(final, (width, height))

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
                        line_hash = hash_line(lines[i])
                        if line_hash not in drawn_lines:
                            x1, y1, x2, y2 = lines[i][0]
                            cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            drawn_lines.add(line_hash)

        frame_with_border = add_border(frame)
        edges_with_border = add_border(edges_bgr)
        final_with_border = add_border(final)

        # Stack images horizontally
        combined = np.hstack((frame_with_border, edges_with_border, final_with_border))

        # Resize output window if needed
        screen_scale = 0.6  # Scale down for display if needed
        combined = cv2.resize(combined, (0, 0), fx=screen_scale, fy=screen_scale)

        # Show the combined frame
        cv2.imshow("Original | Edges | Detected Lines", combined)

        # Press Q on keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()