# Import the required modules
import cv2
import numpy as np


# Define a function to draw the lane lines on the original image
def draw_lane_lines(image, lines):
    # Create a blank image with the same shape as the original image
    line_image = np.zeros_like(image)
    # Check if there are any lines detected
    if lines is not None:
        # Loop over the lines
        for line in lines:
            # Get the coordinates of the line endpoints
            x1, y1, x2, y2 = line.reshape(4)
            # Draw the line on the blank image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    # Return the line image
    return line_image


def draw_centerline(image, lines):
    # Create a blank image with the same shape as the original image
    center_image = np.zeros_like(image)
    # Check if there are any lines detected
    if lines is not None:
        # Initialize lists to store the coordinates of the outer lines
        outer_lines = []
        # Loop over the lines
        for line in lines:
            # Get the coordinates of the line endpoints
            x1, y1, x2, y2 = line.reshape(4)
            # Calculate the slope of the line
            slope = (y2 - y1) / (x2 - x1)
            # Ignore the vertical lines
            if np.abs(slope) > 100:  # Avoid division by zero
                continue
            # Classify the lines into outer lines based on slope magnitude
            outer_lines.append((x1, y1, x2, y2, slope))

        # Check if there are enough outer lines to calculate the centerline
        if len(outer_lines) >= 2:
            # Sort the outer lines based on slope magnitude
            sorted_outer_lines = sorted(outer_lines, key=lambda x: np.abs(x[4]), reverse=True)
            # Take the first two outer lines
            outer1, outer2 = sorted_outer_lines[:2]
            # Calculate the midpoint between the endpoints of the outer lines
            mid_x = int((outer1[0] + outer2[0]) / 2)
            mid_y = int((outer1[1] + outer2[1]) / 2)
            # Calculate the slope of the centerline
            center_slope = (outer1[4] + outer2[4]) / 2
            # Calculate the y-intercept of the centerline using the midpoint
            center_intercept = mid_y - center_slope * mid_x
            try:
                # Calculate the endpoints of the centerline based on image height and slope
                height, width = image.shape[:2]
                center_x1 = int((height - center_intercept) / center_slope)
                center_x2 = int((height / 2 - center_intercept) / center_slope)
                # Draw the centerline on the blank image
                cv2.line(center_image, (center_x1, height), (center_x2, int(height / 2)), (0, 255, 0), 10)
            except OverflowError:
                # Handle overflow error gracefully
                pass
    # Return the center image
    return center_image


# Define a function to mask the region of interest on the image
def region_of_interest(image):
    # Get the height and width of the image
    height, width = image.shape[:2]
    # Define the vertices of the square to mask
    top_left = (int(width * 0.25), int(height * 0.25))
    bottom_right = (int(width * 0.75), int(height * 0.75))
    # Create a blank image with the same shape as the original image
    mask = np.zeros_like(image)
    # Fill the square with white color
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
    # Apply bitwise and operation to the original image and the mask
    masked_image = cv2.bitwise_and(image, mask)
    # Draw the region of interest outline on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2)
    # Return the masked image
    return masked_image


# Capture the video from the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Error opening the camera")

# Loop until the user presses 'q' or the video ends
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    # Check if the frame is valid
    if ret:
        # Draw region of interest outline on the frame
        region_of_interest(frame)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection to find the edges
        canny = cv2.Canny(blur, 50, 150)
        # Apply the region of interest mask to the edge image
        cropped = region_of_interest(canny)
        # Apply Hough transform to find the lines
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=100)
        # Draw the lane lines on the original frame
        line_image = draw_lane_lines(frame, lines)
        # Draw the centerline on the original frame
        center_image = draw_centerline(frame, lines)
        # Combine the original frame with the line images
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        combo_image = cv2.addWeighted(combo_image, 0.8, center_image, 1, 1)
        # Show the result on the screen
        cv2.imshow("Result", combo_image)
        # Wait for 1 millisecond for user input
        key = cv2.waitKey(1)
        # If the user presses 'q', break the loop
        if key == ord('q'):
            break
    else:
        break

# Release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()
