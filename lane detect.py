import cv2
import numpy as np


def apply_square_detection(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Create a mask that only allows a certain region of the frame to be processed
    mask = np.zeros_like(blurred)
    height, width = frame.shape[:2]
    polygon = np.array([[
        (int(width * 0.2), int(height * 0.2)),  # Top left point
        (int(width * 0.8), int(height * 0.2)),  # Top right point
        (int(width * 0.8), int(height * 0.8)),  # Bottom right point
        (int(width * 0.2), int(height * 0.8))  # Bottom left point
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)  # Use 0 instead of 255 to fill the polygon with black
    # mask = cv2.bitwise_not(mask)  # Remove this line
    masked = cv2.bitwise_and(blurred, mask)  # Apply the mask to the blurred frame

    # Use Canny edge detector to find edges in the frame
    edges = cv2.Canny(masked, 50, 150)

    cv2.namedWindow('Edges')

    # Show the edges image in the new window
    cv2.imshow('Edges', edges)

    # Use HoughLinesP to detect lines in the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

    # Draw only a limited amount of detected lines
    line_frame = frame.copy()
    slopes = []  # List to store the slopes of the parallel lines
    intercepts = []  # List to store the intercepts of the parallel lines
    try:
        for i, line in enumerate(lines):
            if i < 8:  # Draw only the first 10ish lines
                x1, y1, x2, y2 = line[0]
                cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
                # Calculate the slope and the intercept of the line using the formula m = (y2 - y1) / (x2 - x1) and b = y1 - mx1
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                # Append the slope and the intercept to the lists
                slopes.append(slope)
                intercepts.append(intercept)
    except TypeError:
        pass

    # Find the center line equation using the average of the slopes and the intercepts
    tol = 5  # Tolerance value for parallel lines
    for i in range(len(slopes)):  # Loop over the slopes
        for j in range(i + 1, len(slopes)):  # Loop over the slopes after the current one
            # Check if the slopes and the intercepts are close enough to be parallel
            if not np.isnan(slopes[i]) and not np.isnan(slopes[j]) and not np.isnan(intercepts[i]) and not np.isnan(
                    intercepts[j]):  # Check if the slopes and the intercepts are not NaN
                if abs(slopes[i] - slopes[j]) < tol and abs(intercepts[i] - intercepts[j]) < tol:
                    # Find the average of the slopes and the intercepts of the parallel lines
                    center_slope = (slopes[i] + slopes[j]) / 2
                    center_intercept = (intercepts[i] + intercepts[j]) / 2
                    # Find the endpoints of the center line using the equation y = mx + b
                    y1 = int(center_slope * 0 + center_intercept)  # y-coordinate of the left endpoint
                    y2 = int(center_slope * width + center_intercept)  # y-coordinate of the right endpoint
                    cv2.line(line_frame, (0, y1), (width, y2), (255, 0, 0), 10)  # Draw a blue line

    # Merge the original frame with the line-drawn frame
    overlay = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)

    # Return the overlay frame
    return overlay


# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply the square detection function to the frame
    overlay = apply_square_detection(frame)

    # Display the resulting frame with the overlay
    cv2.imshow('Video Overlay', overlay)  # Change the variable name to overlay

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
