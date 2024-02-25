# import necessary libraries
import cv2
import numpy as np

# initialize video capture
cap = cv2.VideoCapture(0)

# set thresholds for edge detection
low_threshold = 50
high_threshold = 150
# set kernel size for Gaussian blur
kernel_size = 5
# parameters for Hough transform
rho = 1
theta = np.pi / 180
threshold = 2
min_line_length = 15
max_line_gap = 50
# define line properties
line_color = (255, 0, 0)
line_thickness = 10
# main loop for video capture
while cap.isOpened():
    # read frame from video capture
    ret, frame = cap.read()
    if ret:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        # detect edges using Canny edge detection
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        # create a mask for region of interest
        mask = np.zeros_like(edges)
        mask_h, mask_w = mask.shape
        square_size = min(mask_h, mask_w) // 2
        center_x, center_y = mask_w // 2, mask_h // 2
        cv2.rectangle(mask, (center_x - square_size // 2, center_y - square_size // 2),
                      (center_x + square_size // 2, center_y + square_size // 2), (255), -1)
        # apply mask to edges
        masked_edges = cv2.bitwise_and(edges, mask)
        # detect lines using Hough transform
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        if lines is not None:
            # draw detected lines on frame
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)
        # find center line
        center_color = (0, 255, 0)
        center_thickness = 5
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
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
        # draw center line on frame
        cv2.line(frame, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), center_color, center_thickness)
        # draw mask outline on frame
        mask_outline = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        mask_outline = cv2.rectangle(mask_outline, (center_x - square_size // 2, center_y - square_size // 2),
                                     (center_x + square_size // 2, center_y + square_size // 2), (0, 255, 0), 2)
        # combine frame with mask outline
        result = cv2.addWeighted(frame, 1, mask_outline, 0.5, 0)
        # display result
        cv2.imshow('Lane Detection', result)
        # check for 'q' key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break
# release video capture and close windows
cap.release()
cv2.destroyAllWindows()
