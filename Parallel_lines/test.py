import cv2
import numpy as np

# Check if two lines are roughly vertical and parallel
def parallel(line1, line2):
    x1, y1, x2, y2 = line1[0]
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

    x1, y1, x2, y2 = line2[0]
    slope2 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

    return (abs(slope1 - slope2) < 0.5 or slope1 == slope2) and (abs(slope1) > 5 and abs(slope2) > 5)

# Simplified hashing to compare similar lines
def hash_line(line):
    x1, y1, x2, y2 = line[0]
    return (round(x1 / 10), round(y1 / 10), round(x2 / 10), round(y2 / 10))

# Open video file
cap = cv2.VideoCapture('robosub 2022 gate only.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space for robust red detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Apply mask to get only red regions
    red_only = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Grayscale and blur to reduce noise
    red_gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(red_gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Erode to remove small noise
    kernel = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(edges, kernel)

    # Copy of the original frame to draw on
    output = frame.copy()
    drawn_lines = set()

    # Detect lines
    lines = cv2.HoughLinesP(eroded, 1, np.pi / 180, threshold=30,
                            minLineLength=50, maxLineGap=20)

    # Draw parallel lines
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if parallel(lines[i], lines[j]):
                    line_hash = hash_line(lines[i])
                    if line_hash not in drawn_lines:
                        x1, y1, x2, y2 = lines[i][0]
                        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        drawn_lines.add(line_hash)

    # Show frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Edges', edges)
    cv2.imshow('Detected Lines', output)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
