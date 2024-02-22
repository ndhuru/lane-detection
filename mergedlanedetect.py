# Import the required modules
import cv2
import numpy as np
import math

# Define the source of the video
# You can change this to 0 if you want to use your device camera
video_source = 0

# Define the region of interest for lane detection
def region_of_interest(img, vertices):
    # Define a blank mask to start with
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Define the Canny edge detection function
def canny_edge(img, low_threshold, high_threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    return edges


# Define the Hough transform function
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # Apply Hough transform on the edge detected image
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # Create an empty black image to draw the lines on
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Draw the lines on the image
    draw_lines(line_img, lines)

    return line_img


# Define the function to draw the lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # Iterate over the lines and draw them on the image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Define the function to draw the centerline
def draw_centerline(img, lines, color=[0, 255, 0], thickness=2):
    # Initialize the lists to store the coordinates and slopes of the left and right lane lines
    left_x = []
    left_y = []
    left_slope = []
    right_x = []
    right_y = []
    right_slope = []

    # Iterate over the lines and separate them into left and right lane lines based on their slope and position

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if slope < 0 and x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
                    # Left lane line
                    left_x.append(x1)
                    left_x.append(x2)
                    left_y.append(y1)
                    left_y.append(y2)
                    left_slope.append(slope)
                elif slope > 0 and x1 > img.shape[1] / 2 and x2 > img.shape[1] / 2:
                    # Right lane line
                    right_x.append(x1)
                    right_x.append(x2)
                    right_y.append(y1)
                    right_y.append(y2)
                    right_slope.append(slope)
    except ValueError:
        pass

    # If there are no left or right lane lines, return without drawing the centerline
    if len(left_x) == 0 or len(right_x) == 0:
        return

    # Find the average slope and intercept of the left and right lane lines
    left_slope_mean = np.mean(left_slope)
    left_intercept = np.mean(left_y) - left_slope_mean * np.mean(left_x)
    right_slope_mean = np.mean(right_slope)
    right_intercept = np.mean(right_y) - right_slope_mean * np.mean(right_x)

    # Find the coordinates of the intersection point of the left and right lane lines
    x_intersect = (right_intercept - left_intercept) / (left_slope_mean - right_slope_mean)
    y_intersect = left_slope_mean * x_intersect + left_intercept

    # Find the coordinates of the center point of the image
    x_center = img.shape[1] / 2
    y_center = img.shape[0] / 2

    # Find the slope and intercept of the centerline
    center_slope = (y_center - y_intersect) / (x_center - x_intersect)
    center_intercept = y_center - center_slope * x_center

    # Find the coordinates of the endpoints of the centerline
    y1 = img.shape[0]
    x1 = (y1 - center_intercept) / center_slope
    y2 = y_intersect
    x2 = (y2 - center_intercept) / center_slope

    # Draw the centerline on the image
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


# Define the function to overlay the lines on the original image
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    # Return the result of adding the two images with the given weights
    return cv2.addWeighted(initial_img, α, img, β, γ)


# Create a video capture object to read the video
cap = cv2.VideoCapture(video_source)

# Loop until the end of the video
while (cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read correctly, break the loop
    if not ret:
        break

    # Resize the frame to a smaller size for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Define the vertices of the region of interest
    vertices = np.array([[(100, 540), (460, 320), (520, 320), (900, 540)]], dtype=np.int32)

    # Apply the region of interest function
    roi = region_of_interest(frame, vertices)

    # Apply the Canny edge detection function
    edges = canny_edge(roi, 50, 150)

    # Apply the Hough transform function
    lines = hough_lines(edges, 1, np.pi / 180, 15, 40, 20)

    # Draw the centerline on the image
    draw_centerline(lines, lines)

    # Overlay the lines on the original image
    result = weighted_img(lines, frame)

    # Display the result
    cv2.imshow('Lane Detection', result)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

