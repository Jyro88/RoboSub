import cv2
import numpy as np

# Find and compare slopes of two lines
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

    return (abs(slope1 - slope2) < 0.2 or slope1 == slope2) and (abs(slope1) > 5 and abs(slope2) > 5)


# Draw rectangle on image
def drawRectangle(line1, line2, frame_width, frame_height):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Check if vertical lines are longer than horizontal lines
    if abs(y1 - y2) > abs(x1 - x2) and abs(y3 - y4) > abs(x3 - x4):
        # Check if lines are not close to the edges of the frame
        if (
            0.1 * frame_width < x1 < 0.9 * frame_width and
            0.1 * frame_width < x2 < 0.9 * frame_width and
            0.1 * frame_height < y1 < 0.9 * frame_height and
            0.1 * frame_height < y2 < 0.9 * frame_height and
            0.1 * frame_width < x3 < 0.9 * frame_width and
            0.1 * frame_width < x4 < 0.9 * frame_width and
            0.1 * frame_height < y3 < 0.9 * frame_height and
            0.1 * frame_height < y4 < 0.9 * frame_height
        ):
            # Draw the vertical line
            cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Draw the connected horizontal line
            cv2.line(final, (x1, y1), (x3, y3), (0, 0, 255), 5)
            cv2.line(final, (x2, y2), (x4, y4), (0, 0, 255), 5)

            drawn_lines.add(tuple(line1[0]))  # Add the drawn lines to a set

# Create a VideoCapture object and read from the input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('robosub 2022 gate only.mp4')

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until the video is completed
drawn_lines = set()  # Set to store already drawn lines
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:

        frame_height, frame_width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)

        # Use Canny edge detector to find edges in the image
        edges = cv2.Canny(gray, 50, 150, None, 3)
        cv2.imshow("edges", edges)

        kernel = np.ones((10, 10), np.uint8)
        erode = cv2.erode(edges, kernel)

        final = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)

        # Use Hough Line Transform to detect lines in the image
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        # Go through the array of lines and check for pairs of parallel lines
        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    if (
                        parallel(lines[i], lines[j]) and
                        tuple(lines[i][0]) not in drawn_lines and
                        cv2.norm(np.array(lines[i][0][:2]) - np.array(lines[i][0][2:])) > min_line_length and
                        cv2.norm(np.array(lines[j][0][:2]) - np.array(lines[j][0][2:])) > min_line_length
                    ):
                        drawRectangle(lines[i], lines[j], frame_width, frame_height)


        cv2.imshow('Frame', frame)
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", final)

        # Press Q on the keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything is done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
