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
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(blurred, mask)

    # Use Canny edge detector to find edges in the frame
    edges = cv2.Canny(masked, 50, 150)

    # Use HoughLinesP to detect lines in the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

    # Draw only a limited amount of detected lines
    line_frame = frame.copy()
    try:
        for i, line in enumerate(lines):
            if i < 10:  # Draw only the first 10ish lines
                x1, y1, x2, y2 = line[0]
                cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
    except TypeError:
        pass

    # Merge the original frame with the line-drawn frame
    overlay_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)

    # Return the overlay frame
    return overlay_frame

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply the square detection function to the frame
    overlay = apply_square_detection(frame)

    # Display the resulting frame with the overlay
    cv2.imshow('Video Overlay', overlay)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
