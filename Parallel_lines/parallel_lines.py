import cv2
import numpy as np

# === CONFIG ===
POSITION_THRESHOLD = 20  # pixels of change before redrawing

# === FUNCTIONS ===
def parallel(line1, line2):
    x1, y1, x2, y2 = line1[0]
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    x1, y1, x2, y2 = line2[0]
    slope2 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    return (abs(slope1 - slope2) < 0.5 or slope1 == slope2) and (abs(slope1) > 5 and abs(slope2) > 5)

def hash_line(line):
    x1, y1, x2, y2 = line[0]
    return (round(x1 / 15), round(y1 / 15), round(x2 / 15), round(y2 / 15))

def is_similar(prev, curr):
    return all(abs(p - c) < POSITION_THRESHOLD for p, c in zip(prev, curr))

def add_border(img):
    return cv2.copyMakeBorder(img, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

# === INIT ===
cap = cv2.VideoCapture('robosub 2022 gate only.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")

drawn_lines = {}  # hash → [x1, y1, x2, y2]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === RED MASKING (HSV) ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    red_frame = cv2.bitwise_and(frame, frame, mask=red_mask)

    # === EDGE DETECTION ===
    red_gray = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(red_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, None, 3)
    kernel = np.ones((10, 10), np.uint8)
    erode = cv2.erode(edges, kernel)

    final = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)

    # === HOUGH LINE DETECTION ===
    rho, theta = 1, np.pi / 180
    threshold, min_line_length, max_line_gap = 30, 50, 20
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if parallel(lines[i], lines[j]):
                    h = hash_line(lines[i])
                    x1, y1, x2, y2 = lines[i][0]
                    new_line = [x1, y1, x2, y2]

                    if h in drawn_lines and is_similar(drawn_lines[h], new_line):
                        continue  # Skip drawing if similar

                    # Update and draw
                    drawn_lines[h] = new_line
                    cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # === DISPLAY FRAMES ===
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    edges_bgr = cv2.resize(edges_bgr, (width, height))
    final_resized = cv2.resize(final, (width, height))

    combined = np.hstack((
        add_border(frame),
        add_border(edges_bgr),
        add_border(final_resized)
    ))
    combined = cv2.resize(combined, (0, 0), fx=0.6, fy=0.6)
    cv2.imshow("Original | Edges | Smoothed Final", combined)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
