# Import the required modules
import cv2
import numpy as np


video_source = 0

# Define the region of interest for lane detection
def region_of_interest(img, vertices):
    # Define a blank mask to start with
    # Creates a mask around desired area
    roi = np.zeros(img.shape[:2], dtype="uint8")
    # Calculate the center coordinates of the screen
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    # Calculate the width and height of the ROI
    roi_width = 350
    roi_height = 350
    # Calculate the top-left and bottom-right coordinates of the ROI
    roi_tl_x = center_x - roi_width // 2
    roi_tl_y = center_y - roi_height // 2
    roi_br_x = center_x + roi_width // 2
    roi_br_y = center_y + roi_height // 2
    cv2.rectangle(roi, (roi_tl_x, roi_tl_y), (roi_br_x, roi_br_y), 1, -1)
    mask = cv2.bitwise_and(img, img, mask=roi)
    cv2.rectangle(img, (roi_tl_x, roi_tl_y), (roi_br_x, roi_br_y), (255, 0, 255), 5)
    return mask



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
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    # Iterate over the lines and draw them on the image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Define the function to draw the centerline
def draw_centerline(img, lines, color=[0, 255, 0], thickness=2):
    if lines is not None:
        # Variables needed to find the centerline
        slope_arr = []
        lines_list = []
        for line in lines:
            try:
                # Creates array of lines
                x1, y1, x2, y2 = line[0]
                lines_list.append(line[0])

                # Calculates the slopes of the lines
                slope = 0
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                slope_arr.append(slope)
            except:
                pass

        # Loops through the slope array to calculate the centerline
        for i in range(len(slope_arr)):
            for j in range(len(slope_arr)):
                x1, y1, x2, y2 = lines_list[i]
                x3, y3, x4, y4 = lines_list[j]
                # Calculates and displays the centerline
                cv2.line(img, ((x1 + x3) // 2, (y1 + y3) // 2), ((x2 + x4) // 2, (y2 + y4) // 2), (251, 13, 136), 10)


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
