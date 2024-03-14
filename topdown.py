import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture("/Users/pl252491/PycharmProjects/lanedetection/venv/Car drive.mp4")

# Define the parameters for Canny edge detector
low_threshold = 50
high_threshold = 150
# Define the parameters for Gaussian blur
kernel_size = 5
# Define the parameters for Hough transform
rho = 1
theta = np.pi / 180
threshold = 2
min_line_length = 15
max_line_gap = 50
# Define the color for the lane lines
line_color = (255, 0, 0)  # Blue color for lines
# Define the thickness for the lane lines
line_thickness = 10

# Loop until the user presses 'q' or the video ends
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply Canny edge detector
        edges = cv2.Canny(blur, low_threshold, high_threshold)

        # Create a mask image with a trapezoidal region
        mask = np.zeros_like(edges)
        mask_h, mask_w = mask.shape

        # Define the vertices of the trapezoid
        bottom_left = (int(mask_w * 0.1), mask_h-100)  # 10% from the left of the screen
        top_left = (int(mask_w * 0.4), int(mask_h * 0.6)+50)  # 40% from the left and 60% from the top
        top_right = (int(mask_w * 0.6), int(mask_h * 0.6)+50)  # 60% from the left and 60% from the top
        bottom_right = (int(mask_w * 0.9), mask_h-100)  # 90% from the left of the screen

        # Create an array of points in the order of bottom_left, top_left, top_right, bottom_right
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        # Fill the polygon
        cv2.fillPoly(mask, vertices, 255)

        # Apply the mask
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough transform to find the lines
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # Draw the lines on the original frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

        # Define the color for the centerline
        center_color = (0, 255, 250)  # Green color for centerline
        # Define the thickness for the centerline
        center_thickness = 5

        # Initialize the lists to store the coordinates of the two lines
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []

        # Loop through the lines and append the coordinates to the lists
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
        except:
            pass

        try:
            x1_avg = sum(x1_list) / len(x1_list)
            y1_avg = sum(y1_list) / len(y1_list)
            x2_avg = sum(x2_list) / len(x2_list)
            y2_avg = sum(y2_list) / len(y2_list)
        except:
            pass

        # Draw the centerline on the original frame
        cv2.line(frame, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), center_color, center_thickness)

        # Draw the outline of the mask
        mask_outline = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        mask_outline = cv2.polylines(mask_outline, vertices, True, (0, 255, 0), 2)

        # Combine the original frame with the mask outline
        result = cv2.addWeighted(frame, 1, mask_outline, 0.5, 0)
        # Show the combined frame
        cv2.imshow('Lane Detection', result)

        # Perspective transform
        src_points = np.float32([bottom_left, top_left, top_right, bottom_right])
        dst_points = np.float32([[int(mask_w * 0.2), mask_h], [int(mask_w * 0.2), 0], [int(mask_w * 0.8), 0], [int(mask_w * 0.8), mask_h]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, M, (mask_w, mask_h))
        cv2.imshow('Top Down View', warped)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
    else:
       # If the video ends, exit the loop
        break

# Release the video capture object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
