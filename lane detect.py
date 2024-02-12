import cv2
import numpy as np

# global variable to store previous lines
prev_lines = []
# allows for motion blur
def apply_square_detection(frame, max_lines=None):
    # global var on previous lines, set to list
    global prev_lines

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # detect edges using canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('canny edges', edges)  # show the canny edges

    # apply hough transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=100)

    # create a copy of the frame to draw lines on
    line_frame = frame.copy()

    if lines is not None:
        # limit the number of detected lines if specified
        if max_lines is not None:
            lines = lines[:max_lines]

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # calculate the angle of the line
            angle = np.arctan2(y2 - y1, x2 - x1)
            # draw the detected line
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
            angles.append(angle)

        # calculate the average angle of detected lines
        avg_angle = np.mean(angles)

        # calculate the midpoint between two parallel lines
        if len(lines) >= 2:
            x1_avg, y1_avg, x2_avg, y2_avg = np.mean(lines[:, 0, :], axis=0).astype(int)
            # store the current line coordinates
            current_line = (x1_avg, y1_avg), (x2_avg, y2_avg)

            # append current line to previous lines
            prev_lines.append(current_line)

            # if we have more than 5 previous lines, remove the oldest one
            if len(prev_lines) > 5:
                prev_lines.pop(0)

            # calculate the average line coordinates from previous lines
            avg_line = np.mean(prev_lines, axis=0).astype(int)
            x1_avg, y1_avg = avg_line[0]
            x2_avg, y2_avg = avg_line[1]

            # draw the smoothed line
            cv2.line(line_frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (255, 0, 0), 10)

    # create an overlay by blending the original frame with the frame containing detected lines
    overlay = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)
    return overlay

# open the default camera
cap = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # apply square detection on the frame
    overlay = apply_square_detection(frame, max_lines=4)

    # display the resulting frame with overlays
    cv2.imshow('video overlay', overlay)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# destroy windows when done
cap.release()
cv2.destroyAllWindows()



# sources
# OpenCV-Python: https://github.com/opencv/opencv-python
# How to install cv2: https://stackoverflow.com/questions/57883178/how-can-i-install-cv2
# OpenCV-Python in Windows: https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html
# OpenCV's perspectiveTransform in Python: https://stackoverflow.com/questions/53861636/how-can-i-implement-opencvs-perspectivetransform-in-python 
# Drawing Shapes on Images with Python OpenCV Library: https://wellsr.com/python/drawing-shapes-on-images-with-python-opencv-library/
