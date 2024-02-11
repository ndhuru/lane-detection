import cv2
import numpy as np


def apply_square_detection(frame, max_lines=None):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Use Canny edge detector to find edges in the frame
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('Canny Edges', edges)

    # Use HoughLinesP to detect lines in the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=100)

    # Draw lines
    line_frame = frame.copy()
    if lines is not None:
        if max_lines is not None:
            lines = lines[:max_lines]

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
            angles.append(angle)

        # Calculate the average angle
        avg_angle = np.mean(angles)

        if np.abs(np.pi / 2 - avg_angle) < np.pi / 18:  # Approximately vertical
            # Draw a vertical line
            height, width = frame.shape[:2]
            cv2.line(line_frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 10)
        else:
            # Calculate the middle line based on the average angle
            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            line_length = max(width, height) // 2
            x1_avg = int(center_x - line_length * np.cos(avg_angle))
            y1_avg = int(center_y - line_length * np.sin(avg_angle))
            x2_avg = int(center_x + line_length * np.cos(avg_angle))
            y2_avg = int(center_y + line_length * np.sin(avg_angle))
            cv2.line(line_frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 0, 255), 10)

        # Merge the original frame with the line-drawn frame
    overlay = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)
    return overlay


# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply the square detection function to the frame, specifying max_lines
    overlay = apply_square_detection(frame, max_lines=5)

    # Display the resulting frame with the overlay
    cv2.imshow('Video Overlay', overlay)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

# This code uses the following sources: - OpenCV shape detection: This tutorial explains how to use contour
# approximation and cv2.approxPolyDP to identify shapes in an image. It also shows how to use cv2.arcLength and
# cv2.drawContours to draw the shapes on the image. Citation: Rosebrock, A. (2016, January 11). OpenCV shape
# detection. PyImageSearch. https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/ - Square detection in
# image: This question on Stack Overflow discusses how to detect squares in an image using cv2.Canny,
# cv2.GaussianBlur, cv2.HoughLinesP, and cv2.isContourConvex. It also shows how to calculate the average angle of the
# lines and draw a vertical or middle line based on it. Citation: user366312. (2010, September 29). Square detection
# in image. Stack Overflow. https://stackoverflow.com/questions/3823621/square-detection-in-image - How to detect a
# rectangle and square in an image using OpenCV Python?: This article on TutorialsPoint demonstrates how to use
# cv2.cvtColor, cv2.threshold, cv2.findContours, and cv2.boundingRect to detect and extract rectangles and squares in
# an image. Citation: TutorialsPoint. (n.d.). How to detect a rectangle and square in an image using OpenCV Python?
# TutorialsPoint. https://www.tutorialspoint.com/how-to-detect-a-rectangle-and-square-in-an-image-using-opencv-python
