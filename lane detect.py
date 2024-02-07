import cv2
import numpy as np

def apply_line_detection(frame):
    # first we convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # then apply GaussianBlur to reduce noise and improve line detection overall
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # then we can use Canny edge detector to find edges in the frame
    edges = cv2.Canny(blurred, 50, 150)

    # lastly we use HoughLinesP to detect lines in the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw only a limited amount of detected lines
    line_frame = frame.copy()
    for i, line in enumerate(lines):
        if i < 100:  # Draw only the first 10ish lines
            x1, y1, x2, y2 = line[0]
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # merge the original frame with the line-drawn frame
    overlay_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)

    # lastly, just return the overlay frame
    return overlay_frame

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply the line detection function to the frame
    overlay = apply_line_detection(frame)

    # Display the resulting frame with the overlay
    cv2.imshow('Video Overlay', overlay)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
